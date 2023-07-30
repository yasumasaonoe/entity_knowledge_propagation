import copy
import random
import os
import torch
import types
import yaml
from collections import defaultdict
from alive_progress import alive_bar
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from .metrics import compute_perplexity_gpt, compute_perplexity_t5
from .metrics import compute_dist_over_labels_gpt, compute_dist_over_labels_t5
from .trainer import finetuning
from .edit_func import ft_gpt_entity_inferences, ft_t5_entity_inferences
from .edit_func import ft_gpt_ecbd, ft_t5_ecbd
from .edit_func import prepend_def_t5, prepend_def_gpt
from .edit_func import mend_gpt_ecbd, mend_t5_ecbd
from .edit_func import mend_gpt_entity_inferences, mend_t5_entity_inferences
from .data_utils import to_tsr_gpt_ecbd, to_tsr_t5_ecbd, load_json
from .data_utils import format_gpt_data, format_gpt_data_entity_inferences, \
    format_gpt2_data, format_gpt2_data_entity_inferences
from .data_utils import to_tsr_gpt_entity_inference, to_tsr_t5_entity_inference
from .data_utils import MEND_DIR, MEND_MODEL_DIR, SPECIFICITY_DATA_PATH
from .mend.mend import MEND


def convert_dict_to_namespace(d):
    ns = types.SimpleNamespace()
    for k, v in d.items():
        if 'lr' in k:
            v = float(v)
        setattr(ns, k, v)
    return ns


def run_edit_entity_inferences(data,
                              dataset_name,
                              edit_method,
                              device,
                              train_params,
                              model_name=None,
                              random_def=None):

    # Load a raw model and tokenizer.
    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_entity_inference
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not param_name.startswith('transformer.h.23'):  # Last layer
                    param.requires_grad = False
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_entity_inference
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not any(param_name.startswith(p)
                           for p in ['encoder.block.23.layer',
                                     'decoder.block.23.layer']):  # Last layer
                    param.requires_grad = False
    elif train_params['BASE_MODEL'] == 'gpt2-xl':
        model_raw = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_entity_inference
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not param_name.startswith('transformer.h.47'):  # Last layer
                    param.requires_grad = False
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
    model_raw = model_raw.to(device)

    # Check num of params to update.
    param_update = 0
    param_freeze = 0
    total_param_num = 0
    for param_name, param in model_raw.named_parameters():
        total_param_num += 1
        if param.requires_grad:
            param_update += 1
        else:
            param_freeze += 1

    print(f'#Paramas to update: {param_update}')
    print(f'#Params to freeze: {param_freeze}')

    all_outputs = []

    edit_func = None
    model_ft = None
    # Select edit function.
    if edit_method == 'ft':  # Finetuned on all examples together.
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/'\
                     'model_files'
        print(model_name)
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt_entity_inferences
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = ft_t5_entity_inferences
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        else:
            raise NotImplementedError(
                'Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method == 'ft_per_ex':
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt_entity_inferences
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = ft_t5_entity_inferences
    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def',
                         'sanity_check']:
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = prepend_def_t5
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt_entity_inferences

        elif train_params['BASE_MODEL'] == 't5-large':
            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(MEND_MODEL_DIR,
                                     't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5_entity_inferences

    else:
        raise NotImplementedError

    with alive_bar(total=len(data)) as bar:
        for i, ex in enumerate(data):
            output = {'ex_id': ex['ex_id']}
            label = ex['label']
            batch = to_tsr(tokenizer, ex, device)
            bleu_score = batch['bleu_score']
            bert_score = batch['bert_score']
            bleurt_score = batch['bleurt_score']
            meteor_score = batch['meteor_score']

            specificity_batches = None
            if train_params['COMPUTE_SPECIFICITY']:
                specificity_data = [ex for j, ex in enumerate(data) if i != j]
                specificity_batches = [
                    to_tsr(tokenizer, ex, device) for ex in specificity_data]

            if edit_method == 'ft':  # Finetuned on all examples together.
                _, _, \
                pre_edit_dict, post_edit_dict, \
                post_loc_dict, pre_loc_dict = edit_func(batch,
                                                        model_ft,
                                                        model_raw=model_raw)
            elif edit_method == 'ft_per_ex':
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(model_ft,
                                            tokenizer,
                                            ex,
                                            train_params,
                                            device)  # single instance

                # n_probe_labels = batch['edit_inner'][0]['labels'][
                #     'input_ids'].size(0)
                # for i in range(n_probe_labels):
                #     label_tsr = batch["edit_inner"][0]["labels"]['input_ids'][
                #         i].unsqueeze(0)
                #     print('Label:', tokenizer.convert_ids_to_tokens(label_tsr[
                #                                                         0]))

                _, _, \
                pre_edit_dict, post_edit_dict, \
                _, _, \
                pre_loc_dicts, post_loc_dicts = edit_func(
                    batch,
                    model_ft,
                    model_raw=model_raw,
                    specificity_batches=specificity_batches)

            elif edit_method == 'prepend_def':
                batch_prepended_def = to_tsr(tokenizer,
                                             ex,
                                             device,
                                             prepend_def=True,
                                             prepend_sent=False,
                                             random_def=None)
                _, _, \
                pre_edit_dict, post_edit_dict, \
                post_loc_dict, pre_loc_dict = edit_func(
                    batch,
                    batch_prepended_def,
                    model_raw,
                    dataset_name=dataset_name)
            elif edit_method == 'random_def':
                batch_prepended_def = to_tsr(tokenizer,
                                             ex,
                                             device,
                                             prepend_def=False,
                                             prepend_sent=False,
                                             random_def=random_def)
                _, _, \
                pre_edit_dict, post_edit_dict, \
                post_loc_dict, pre_loc_dict = edit_func(
                    batch,
                    batch_prepended_def,
                    model_raw,
                    dataset_name=dataset_name)
            elif edit_method == 'sanity_check':
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(model_ft,
                                            tokenizer,
                                            ex,
                                            train_params,
                                            device)
                _, _, \
                pre_edit_dict, post_edit_dict, \
                post_loc_dict, pre_loc_dict = edit_func(batch,
                                                        model_ft,
                                                        model_raw=model_raw)
            elif edit_method == 'mend':
                _, _, \
                pre_edit_dict, post_edit_dict, \
                _, _, \
                pre_loc_dicts, post_loc_dicts = edit_func(
                    batch,
                    mend_model,
                    specificity_batches=specificity_batches,
                    dataset_name=dataset_name)

            else:
                raise

            assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

            j = 0
            # Assuming only 1 probe sentence.
            if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:

                labels, pre_probs, pre_lls = compute_dist_over_labels_gpt(
                    tokenizer,
                    pre_edit_dict,
                    ex['probe_sentences'][f'template_{j}']['labels'],
                    batch["edit_inner"][j]['labels'],
                    batch["edit_inner"][j]['left_context_ps'],
                    batch["edit_inner"][j]['right_context_ps']
                )

                if edit_method in ['prepend_def', 'random_def']:
                    labels, post_probs, post_lls = compute_dist_over_labels_gpt(
                        tokenizer,
                        post_edit_dict,
                        ex['probe_sentences'][f'template_{j}']['labels'],
                        batch_prepended_def["edit_inner"][j]['labels'],
                        batch_prepended_def["edit_inner"][j]['left_context_ps'],
                        batch_prepended_def["edit_inner"][j]['right_context_ps']
                    )
                else:
                    labels, post_probs, post_lls = compute_dist_over_labels_gpt(
                        tokenizer,
                        post_edit_dict,
                        ex['probe_sentences'][f'template_{j}']['labels'],
                        batch["edit_inner"][j]['labels'],
                        batch["edit_inner"][j]['left_context_ps'],
                        batch["edit_inner"][j]['right_context_ps']
                    )

                # Release GPU memory.
                pre_edit_dict = None
                post_edit_dict = None

                results_specificity = None
                if train_params['COMPUTE_SPECIFICITY']:
                    results_specificity = []
                    assert len(specificity_batches) == len(pre_loc_dicts) \
                           == len(post_loc_dicts)
                    for k in range(len(specificity_batches)):

                        s_batch = specificity_batches[k]
                        s_labels, s_pre_probs, s_pre_lls = \
                        compute_dist_over_labels_gpt(
                            tokenizer,
                            pre_loc_dicts[k],
                            specificity_data[k]['probe_sentences'][
                                                'template_0']['labels'],
                            s_batch["edit_inner"][0]['labels'],
                            s_batch["edit_inner"][0]['left_context_ps'],
                            s_batch["edit_inner"][0]['right_context_ps']
                        )

                        s_labels, s_post_probs, s_post_lls = \
                        compute_dist_over_labels_gpt(
                            tokenizer,
                            post_loc_dicts[k],
                            specificity_data[k]['probe_sentences'][
                                                'template_0']['labels'],
                            s_batch["edit_inner"][0]['labels'],
                            s_batch["edit_inner"][0]['left_context_ps'],
                            s_batch["edit_inner"][0]['right_context_ps']
                        )
                        s_label = specificity_data[k]['label']
                        s_result = [p for p in
                                  zip(s_labels, s_pre_lls, s_post_lls,
                                      s_pre_probs, s_post_probs)
                                  if p[0] == s_label][0]
                        s_pred_dist = [
                            list(zip(s_labels, s_pre_lls, s_post_lls,
                                     s_pre_probs, s_post_probs)), s_label]
                        results_specificity.append(
                            {'results': s_result, 'probs': s_pred_dist})

                pre_loc_dicts = None
                post_loc_dicts = None

            elif train_params['BASE_MODEL'] == 't5-large':

                labels, pre_probs, pre_lls = compute_dist_over_labels_t5(
                    tokenizer,
                    pre_edit_dict,
                    ex['probe_sentences'][f'template_{j}']['labels'],
                    batch["edit_inner"][j]['labels']
                )

                labels, post_probs, post_lls = compute_dist_over_labels_t5(
                    tokenizer,
                    post_edit_dict,
                    ex['probe_sentences'][f'template_{j}']['labels'],
                    batch["edit_inner"][j]['labels']
                )
                # Release GPU memory.
                pre_edit_dict = None
                post_edit_dict = None

                results_specificity = None
                if train_params['COMPUTE_SPECIFICITY']:
                    results_specificity = []
                    assert len(specificity_batches) == len(pre_loc_dicts) \
                           == len(post_loc_dicts)
                    for k in range(len(specificity_batches)):

                        s_batch = specificity_batches[k]
                        s_labels, s_pre_probs, s_pre_lls = \
                        compute_dist_over_labels_t5(
                            tokenizer,
                            pre_loc_dicts[k],
                            specificity_data[k]['probe_sentences'][
                                                'template_0']['labels'],
                            s_batch["edit_inner"][0]['labels']
                        )

                        s_labels, s_post_probs, s_post_lls = \
                        compute_dist_over_labels_t5(
                            tokenizer,
                            post_loc_dicts[k],
                            specificity_data[k]['probe_sentences'][
                                                'template_0']['labels'],
                            s_batch["edit_inner"][0]['labels']
                        )

                        s_label = specificity_data[k]['label']
                        s_result = [p for p in
                                  zip(s_labels, s_pre_lls, s_post_lls,
                                      s_pre_probs, s_post_probs)
                                  if p[0] == s_label][0]
                        s_pred_dist = [
                            list(zip(s_labels, s_pre_lls, s_post_lls,
                                     s_pre_probs, s_post_probs)), s_label]
                        results_specificity.append(
                            {'results': s_result, 'probs': s_pred_dist})

                pre_loc_dicts = None
                post_loc_dicts = None

            else:
                raise NotImplementedError

            result = None
            pred_dist = None
            if label in labels:
                result = [p for p in
                     zip(labels, pre_lls, post_lls, pre_probs, post_probs)
                     if p[0] == label][0]
                pred_dist = [list(zip(labels, pre_lls, post_lls, pre_probs,
                              post_probs)), label]
            elif isinstance(label, list):
                label_scores = []
                all_scores = []
                for p in zip(labels, pre_lls, post_lls, pre_probs,
                             post_probs):
                    all_scores.append(p)
                    if p[0] in label:
                        label_scores.append(p)
                result = label_scores
                pred_dist = [all_scores, label]
            else:
                print('-' * 60)
                print('Probe Sentence {}: {}'.format(j,
                                                     ex['probe_sentences'][
                                                         f'template_{j}'][
                                                         'probe_sentence']))
                print('WARNING: Label not found! {}'.format(label))
                print('         Labels {}'.format(labels))
                for p in zip(labels, pre_lls, post_lls, pre_probs,
                             post_probs):
                    print(p)

            if train_params['COMPUTE_SPECIFICITY']:
                assert len(results_specificity) == len(data) - 1, \
                    (len(results_specificity), len(data))

            output['results'] = result
            output['probs'] = pred_dist
            output['sim_scores'] = {
                'bleu_score': bleu_score,
                'bert_score': bert_score,
                'bleurt_score': bleurt_score,
                'meteor_score': meteor_score,
            }
            output['specificity'] = results_specificity
            all_outputs.append(output)
            bar()

    return all_outputs


def run_edit_ecbd(data,
                  dataset_name,
                  edit_method,
                  device,
                  train_params,
                  model_name=None,
                  random_def=None,
                  oracle_ft=False,
                  specificity_data=None):

    # Load a raw model and tokenizer.
    if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
        model_raw = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not param_name.startswith('transformer.h.23'):  # Last layer
                    param.requires_grad = False
    elif train_params['BASE_MODEL'] in ['t5-large', 't5-base']:
        model_raw = T5ForConditionalGeneration.from_pretrained(
            train_params['BASE_MODEL'])
        tokenizer = T5Tokenizer.from_pretrained(train_params['BASE_MODEL'])
        to_tsr = to_tsr_t5_ecbd
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not any(param_name.startswith(p)
                           for p in ['encoder.block.23.layer',
                                     'decoder.block.23.layer']):  # Last layer
                    param.requires_grad = False
    elif train_params['BASE_MODEL'] == 'gpt2-xl':
        model_raw = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        to_tsr = to_tsr_gpt_ecbd
        if train_params['FREEZE_LAYERS']:
            for param_name, param in model_raw.named_parameters():
                if not param_name.startswith('transformer.h.47'):  # Last layer
                    param.requires_grad = False
    else:
        raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
    model_raw = model_raw.to(device)

    # Check num of params to update.
    param_update = 0
    param_freeze = 0
    total_param_num = 0
    for param_name, param in model_raw.named_parameters():
        total_param_num += 1
        if param.requires_grad:
            param_update += 1
        else:
            param_freeze += 1

    print(f'#Paramas to update: {param_update}')
    print(f'#Params to freeze: {param_freeze}')

    # Finetuned model.
    model_ft = None

    # Select edit function.
    if edit_method == 'ft':
        assert model_name is not None, 'FT: Finetuned model must be provided.'
        # Load a finetuned model.
        checkpoint = f'/mnt/data1/yasu/newent/ft_outputs/{model_name}/'\
                     'model_files'
        print(model_name)
        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            edit_func = ft_gpt_ecbd
            model_ft = GPTNeoForCausalLM.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = ft_t5_ecbd
            model_ft = T5ForConditionalGeneration.from_pretrained(checkpoint)
        elif train_params['BASE_MODEL'] == 'gpt2-xl':
            edit_func = ft_gpt_ecbd
            model_ft = GPT2LMHeadModel.from_pretrained(checkpoint)
        else:
            raise NotImplementedError('Currently, we use either GPT-Neo or T5.')
        model_ft = model_ft.to(device)
    elif edit_method in ['ft_per_ex', 'sanity_check']:
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = ft_gpt_ecbd
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = ft_t5_ecbd
    elif edit_method in ['prepend_def', 'prepend_sent', 'random_def']:
        if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:
            edit_func = prepend_def_gpt
        elif train_params['BASE_MODEL'] == 't5-large':
            edit_func = prepend_def_t5
    elif edit_method == 'mend':

        # Mend yaml
        with open(os.path.join(MEND_DIR, 'mend.yaml'), 'r') as f:
            mend_cfg = yaml.safe_load(f)
        _config = convert_dict_to_namespace(mend_cfg)
        _config.mend = convert_dict_to_namespace(mend_cfg['mend'])

        if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
            with open(os.path.join(MEND_DIR, 'gptneo13.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)
            model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 'gpt-neo-1.3B.2022-12-04_12-59-44_5903457054')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_gpt_ecbd

        elif train_params['BASE_MODEL'] == 't5-large':

            # Model yaml
            with open(os.path.join(MEND_DIR, 't5large_gen.yaml'), 'r') as f:
                model_cfg = yaml.safe_load(f)
            _config.model = convert_dict_to_namespace(model_cfg)

            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            mend_model = MEND(model, _config, lambda: copy.deepcopy(model))
            mend_path = os.path.join(
                MEND_MODEL_DIR, 't5-large.2022-02-12_15-17-56_1732139287')
            archive = torch.load(mend_path, map_location="cpu")
            mend_model.load_state_dict(archive["model"])
            mend_model.to(device)

            edit_func = mend_t5_ecbd

    else:
        raise NotImplementedError

    if specificity_data is not None:
        specificity_batches = [
            to_tsr(tokenizer, _ex, device) for _ex in
            specificity_data]

    all_outputs = []

    with alive_bar(total=len(data)) as bar:

        for i, ent in enumerate(data[:]):

            # specificity_batches = None
            # if train_params['COMPUTE_SPECIFICITY']:
            #     other_entities = [ent for j, ent in enumerate(data) if i != j]
            #     specificity_batches = []
            #     for _ent in random.sample(other_entities, 10):
            #         ent_id, examples = _ent
            #         specificity_batches += [
            #             to_tsr(tokenizer, ex, device) for ex in examples]

            # ex contains multiple probe sentences.
            ent_id, examples = ent
            ex_for_finetuning = examples[0]  # Take the first one

            if oracle_ft:
                random.shuffle(examples)
                n_ex = len(examples) // 2
                if n_ex:
                    ex_for_training = examples[:n_ex]
                    ex_for_testing = examples[n_ex:]
                    ex_for_finetuning = ex_for_training[0]  # Dummy
                    ex_for_finetuning['definition'] = \
                    ex_for_finetuning['probe_sentences']['template_0'][
                        'probe_sentence']
                    ex_for_finetuning['def_target'] = \
                    ex_for_finetuning['probe_sentences']['template_0']['label']
                    ex_for_finetuning['additional_sentences'] = []
                    for _ex in ex_for_training[1:]:
                        ex_for_finetuning['additional_sentences'].append(
                            [
                                _ex['probe_sentences']['template_0'][
                                    'probe_sentence'],
                                _ex['probe_sentences']['template_0']['label']
                            ]
                        )
                    examples = ex_for_testing
                else:
                    continue

            if edit_method == 'ft_per_ex':
                # Finetune a model
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
                model_ft, loss = finetuning(
                    model_ft,
                    tokenizer,
                    ex_for_finetuning,
                    train_params,
                    device,
                    specificity_batches=specificity_batches,
                    alpha=train_params['ALPHA'],
                    reg_type=train_params['REG_TYPE']
                )  # single instance
            elif edit_method in ['prepend_def', 'random_def']:
                model_ft = copy.deepcopy(model_raw)
                model_ft = model_ft.to(device)
            elif edit_method in ['mend', 'sanity_check']:
                pass
            else:
                raise

            for ex in examples[:]:
                output = {'ex_id': ex['ex_id']}

                batch = to_tsr(tokenizer, ex, device)
                bleu_score = batch['bleu_score']
                bert_score = batch['bert_score']
                bleurt_score = batch['bleurt_score']
                meteor_score = batch['meteor_score']

                if edit_method == 'ft_per_ex':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches)
                elif edit_method == 'prepend_def':
                    batch_prepended_def = to_tsr(tokenizer,
                                                 ex,
                                                 device,
                                                 prepend_def=True,
                                                 prepend_sent=False,
                                                 random_def=None)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    post_loc_dict, pre_loc_dict = edit_func(
                        batch,
                        batch_prepended_def,
                        model_ft,
                        dataset_name=dataset_name)
                elif edit_method == 'random_def':
                    batch_prepended_def = to_tsr(tokenizer,
                                                 ex,
                                                 device,
                                                 prepend_def=False,
                                                 prepend_sent=False,
                                                 random_def=random_def)
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    post_loc_dict, pre_loc_dict = edit_func(
                        batch,
                        batch_prepended_def,
                        model_ft,
                        dataset_name=dataset_name)
                elif edit_method == 'sanity_check':

                    ex_for_finetuning = copy.deepcopy(ex)
                    ex_for_finetuning['definition'] = \
                        ex['probe_sentences']['template_0']['probe_sentence']
                    ex_for_finetuning['def_target'] = \
                        ex['probe_sentences']['template_0']['label']

                    model_ft = copy.deepcopy(model_raw)
                    model_ft = model_ft.to(device)
                    model_ft, loss = finetuning(model_ft, tokenizer,
                                                ex_for_finetuning,
                                                train_params, device)

                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        model_ft,
                        model_raw=model_raw,
                        specificity_batches=specificity_batches)
                elif edit_method == 'mend':
                    pre_edit_logits, post_edit_logits, \
                    _, _, \
                    pre_loc_logits, post_loc_logits, \
                    _, _ = edit_func(
                        batch,
                        mend_model,
                        specificity_batches=specificity_batches,
                        dataset_name=dataset_name)
                else:
                    raise

                assert len(batch["edit_inner"]) == 1, len(batch["edit_inner"])

                j = 0
                # Assuming only 1 probe sentence.
                if train_params['BASE_MODEL'] in ['gpt-neo-1.3B', 'gpt2-xl']:

                    results_specificity = None

                    if edit_method in ['prepend_def', 'random_def']:
                        pre_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['probe_sentence'][
                                'input_ids'],
                            batch["edit_inner"][j]['probe_sentence'][
                                'attention_mask'],
                            batch["edit_inner"][j]['probe_sentence'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            post_edit_logits,
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['input_ids'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence']['attention_mask'],
                            batch_prepended_def["edit_inner"][j][
                                'probe_sentence'],
                            batch_prepended_def["edit_inner"][j][
                                'left_context_ps'],
                            batch_prepended_def["edit_inner"][j][
                                'right_context_ps']
                        )

                    else:
                        pre_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            pre_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        post_perp_loss = compute_perplexity_gpt(
                            tokenizer,
                            post_edit_logits,
                            batch["edit_inner"][j]['labels']['input_ids'],
                            batch["edit_inner"][j]['labels']['attention_mask'],
                            batch["edit_inner"][j]['labels'],
                            batch["edit_inner"][j]['left_context_ps'],
                            batch["edit_inner"][j]['right_context_ps']
                        )

                        pre_edit_logits = None
                        post_edit_logits = None

                        if train_params['COMPUTE_SPECIFICITY']:
                            results_specificity = []
                            assert len(specificity_batches) == len(
                                pre_loc_logits) \
                                   == len(post_loc_logits)
                            for k in range(len(specificity_batches)):
                                s_batch = specificity_batches[k]
                                s_pre_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    pre_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                s_post_perp_loss = compute_perplexity_gpt(
                                    tokenizer,
                                    post_loc_logits[k],
                                    s_batch["edit_inner"][0]['labels'][
                                        'input_ids'],
                                    s_batch["edit_inner"][0]['labels'][
                                        'attention_mask'],
                                    s_batch["edit_inner"][0]['labels'],
                                    s_batch["edit_inner"][0]['left_context_ps'],
                                    s_batch["edit_inner"][0]['right_context_ps']
                                )

                                results_specificity.append(
                                    {'pre': s_pre_perp_loss[0],
                                     'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None

                elif train_params['BASE_MODEL'] == 't5-large':
                    label_ids = batch["edit_inner"][0]['labels']['input_ids']
                    label_attention_mask = batch["edit_inner"][0]['labels'][
                        'attention_mask']
                    pre_perp_loss = compute_perplexity_t5(tokenizer,
                                                          pre_edit_logits,
                                                          label_ids,
                                                          label_attention_mask)
                    post_perp_loss = compute_perplexity_t5(tokenizer,
                                                           post_edit_logits,
                                                           label_ids,
                                                           label_attention_mask)

                    pre_edit_logits = None
                    post_edit_logits = None

                    results_specificity = None
                    if train_params['COMPUTE_SPECIFICITY']:
                        results_specificity = []
                        assert len(specificity_batches) == len(pre_loc_logits) \
                               == len(post_loc_logits)
                        for k in range(len(specificity_batches)):
                            s_batch = specificity_batches[k]
                            s_pre_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                pre_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            s_post_perp_loss = compute_perplexity_t5(
                                tokenizer,
                                post_loc_logits[k],
                                s_batch["edit_inner"][0]['labels'][
                                    'input_ids'],
                                s_batch["edit_inner"][0]['labels'][
                                    'attention_mask']
                            )

                            results_specificity.append(
                                {'pre': s_pre_perp_loss[0],
                                 'post': s_post_perp_loss[0]})

                    pre_loc_logits = None
                    post_loc_logits = None

                else:
                    raise NotImplementedError

                output['pre'] = pre_perp_loss[0]
                output['post'] = post_perp_loss[0]
                output['sim_scores'] = {
                    'bleu_score': bleu_score,
                    'bert_score': bert_score,
                    'bleurt_score': bleurt_score,
                    'meteor_score': meteor_score
                }
                output['specificity'] = results_specificity
                all_outputs.append(output)
                if not pre_perp_loss[0][1]:
                    print('WARNING')
                    print('Example:', ex)
                    print('pre_perp_loss', pre_perp_loss)
                    print('post_perp_loss', post_perp_loss)

            bar()

    return all_outputs


def group_examples(data):
    data_d = defaultdict(list)
    for ex in data:
        ent_id = '_'.join(ex['ex_id'].split('_')[:2])
        data_d[ent_id].append(ex)
    return list(data_d.items())


def run_experiment(ki_method,
                   ft_model_name,
                   dataset_name,
                   data_files,
                   device,
                   train_params,
                   random_def=None,
                   oracle_ft=False):

    outputs_d = {}
    for data_file in data_files:
        data = load_json(data_file)
        print(data_file, len(data))

        if dataset_name == 'ecbd':
            specificity_data = None
            if train_params["COMPUTE_SPECIFICITY"]:
                specificity_data = load_json(SPECIFICITY_DATA_PATH)
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                data = [format_gpt_data(ex) for ex in data]
                if train_params["COMPUTE_SPECIFICITY"]:
                    specificity_data = [
                        format_gpt_data(ex) for ex in specificity_data]
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                data = [format_gpt2_data(ex) for ex in data]
                if train_params["COMPUTE_SPECIFICITY"]:
                    specificity_data = [
                        format_gpt2_data(ex) for ex in specificity_data]
            # For ECBD, we group examples by entities and finetune only once per
            # entity. This is unnecessary for specificity data.
            data = group_examples(data)
            all_outputs = run_edit_ecbd(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                model_name=ft_model_name,
                random_def=random_def,
                oracle_ft=oracle_ft,
                specificity_data=specificity_data)

        else:  # Entity Inferences
            if train_params['BASE_MODEL'] == 'gpt-neo-1.3B':
                data = [format_gpt_data_entity_inferences(ex) for ex in data]
            elif train_params['BASE_MODEL'] == 'gpt2-xl':
                data = [format_gpt2_data_entity_inferences(ex) for ex in data]
            all_outputs = run_edit_entity_inferences(
                data,
                dataset_name,
                ki_method,
                device,
                train_params,
                model_name=ft_model_name,
                random_def=random_def)

        # Aggregate results
        outputs_d[data_file] = all_outputs

        torch.cuda.empty_cache()

    return outputs_d