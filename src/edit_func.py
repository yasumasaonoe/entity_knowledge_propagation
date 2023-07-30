from typing import Dict, List, Any

import torch
import torch.nn.functional as F

'''
https://github.com/eric-mitchell/mend
'''

def get_log_probs(pred, targ, shift=False):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(
        -1, targ.unsqueeze(-1)).squeeze(-1)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)  # We want to get the whole sequence right
    acc = correct.float().mean()

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    log_prob_all = unmasked_log_probs * mask.float()
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob,
        "log_prob_all": log_prob_all
    }


def compute_specificity_entity_inferences(model_raw, model_ft,
                                          specificity_batches, shift=False):
    pre_loc_dicts = []
    post_loc_dicts = []
    is_mend = hasattr(model_raw, 'model')
    name_or_path = model_raw.model.name_or_path if is_mend else \
        model_raw.name_or_path
    for s_batch in specificity_batches:
        if 't5' in name_or_path:
            ex = s_batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['decoder_attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
        pre_loc_logits = model_raw(**ex) if is_mend else model_raw(**ex).logits
        post_loc_logits = model_ft(**ex) if is_mend else model_ft(**ex).logits
        pre_loc_dict = []
        post_loc_dict = []
        n_probe_labels = s_batch['edit_inner'][0]['labels'][
            'input_ids'].size(0)
        for i in range(n_probe_labels):
            label = s_batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_loc_dict.append(get_log_probs(
                pre_loc_logits,label, shift=shift))
            post_loc_dict.append(get_log_probs(
                post_loc_logits, label, shift=shift))
        pre_loc_dicts.append(pre_loc_dict)
        post_loc_dicts.append(post_loc_dict)
    return pre_loc_dicts, post_loc_dicts


def compute_specificity_ecbd(model_raw, model_ft, specificity_batches):
    pre_loc_logits = []
    post_loc_logits = []
    is_mend = hasattr(model_raw, 'model')
    name_or_path = model_raw.model.name_or_path if is_mend else \
        model_raw.name_or_path
    i = 0
    for s_batch in specificity_batches:
        if 't5' in name_or_path :
            ex = s_batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['decoder_attention_mask'] = s_batch["edit_inner"][0]['labels'][
               'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = s_batch["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = s_batch["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
        pre_loc_logits.append(
            model_raw(**ex) if is_mend else model_raw(**ex).logits)
        post_loc_logits.append(
            model_ft(**ex) if is_mend else model_ft(**ex).logits)
        i += 1
    return pre_loc_logits, post_loc_logits


def ft_t5_entity_inferences(
        batch, model_ft, model_raw=None, specificity_batches=None):

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):

            ex = batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)
            ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
                'attention_mask'][i].unsqueeze(0)

            pre_edit_logits = model_raw(**ex).logits
            post_edit_logits = model_ft(**ex).logits

            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, ex['labels'], shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, ex['labels'], shift=False))

    pre_loc_logits = None
    post_loc_logits = None
    pre_loc_dicts, post_loc_dicts = \
    compute_specificity_entity_inferences(model_raw, model_ft,
                                          specificity_batches,
                                          shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def ft_t5_ecbd(batch, model_ft, model_raw=None, specificity_batches=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
        model_raw, model_ft, specificity_batches)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def ft_gpt_entity_inferences(batch, model_ft, model_raw=None,
                             specificity_batches=None):

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):

            ex = {}
            ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)
            ex['attention_mask'] = batch["edit_inner"][0]['labels'][
                'attention_mask'][i].unsqueeze(0)
            ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)

            pre_edit_logits = model_raw(**ex).logits
            post_edit_logits = model_ft(**ex).logits

            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, ex['labels'], shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, ex['labels'], shift=True))

    pre_loc_logits = None
    post_loc_logits = None
    pre_loc_dicts, post_loc_dicts = \
    compute_specificity_entity_inferences(model_raw, model_ft,
                                          specificity_batches,
                                          shift=True)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def ft_gpt_ecbd(
        batch, model_ft, model_raw=None, specificity_batches=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model_raw(**ex).logits

    with torch.set_grad_enabled(False):
        post_edit_logits = model_ft(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, label, shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
        model_raw, model_ft, specificity_batches)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def prepend_def_t5(batch_pre, batch_post, model, dataset_name=None):
    # No Def
    with torch.no_grad():
        ex = batch_pre["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['decoder_attention_mask'] = \
        batch_pre["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)
        pre_edit_logits= model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):
        ex = batch_post["edit_inner"][0]['probe_sentence']
        ex['labels'] = batch_post["edit_inner"][0]['labels']['input_ids'][
            0].unsqueeze(0)  # Dummy label
        ex['decoder_attention_mask'] = \
        batch_post["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)
        post_edit_logits = model(**ex).logits

    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            pre_label = batch_pre["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=False))
            post_label = batch_post["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=False))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, \
           post_loc_dict, pre_loc_dict


def prepend_def_gpt(batch_pre, batch_post, model, dataset_name=None):

    # No def
    with torch.no_grad():
        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_pre["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)

        pre_edit_logits = model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):

        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_post["edit_inner"][0][
                'probe_sentence']['attention_mask'][0].unsqueeze(0)

        else:
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['labels'][
                'input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_post["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)

        post_edit_logits = model(**ex).logits


    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(
            0)
        pre_edit_dict = []
        post_edit_dict = []

        for i in range(n_probe_labels):
            if dataset_name == 'ecbd':
                pre_label = \
                batch_pre["edit_inner"][0]["probe_sentence"]['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['probe_sentence']['input_ids'][
                    i].unsqueeze(0)
            else:
                pre_label = \
                batch_pre["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)

            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=True))

            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=True))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, \
           post_loc_dict, pre_loc_dict


def mend_gpt_ecbd(batch, model, specificity_batches=None, dataset_name=None):

    ex = {}
    ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)
    ex['attention_mask'] = batch["edit_inner"][0]['labels'][
        'attention_mask'][0].unsqueeze(0)
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model(**ex)

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.set_grad_enabled(False):
        post_edit_logits = edited_model(**ex)

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, label, shift=True))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, label, shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None

    if specificity_batches is not None and dataset_name is not None:
        pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
            model, edited_model, specificity_batches)

    del edited_model

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def mend_t5_ecbd(batch, model, specificity_batches=None, dataset_name=None):

    ex = batch["edit_inner"][0]['probe_sentence']
    ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
        0].unsqueeze(0)  # Dummy label
    ex['decoder_attention_mask'] = \
        batch["edit_inner"][0]['labels']['attention_mask'][0].unsqueeze(0)

    # Before edit
    with torch.no_grad():
        pre_edit_logits = model(**ex)

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.set_grad_enabled(False):
        post_edit_logits = edited_model(**ex)

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            label = batch["edit_inner"][0]["labels"]['input_ids'][
                i].unsqueeze(0)
            pre_edit_dict.append(get_log_probs(pre_edit_logits, label))
            post_edit_dict.append(get_log_probs(post_edit_logits, label))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        pre_loc_logits, post_loc_logits = compute_specificity_ecbd(
            model, edited_model, specificity_batches)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def mend_gpt_entity_inferences(
        batch, model, specificity_batches=None, dataset_name=None):

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            ex = {}
            ex['input_ids'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)
            ex['attention_mask'] = batch["edit_inner"][0]['labels'][
                'attention_mask'][i].unsqueeze(0)
            ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)

            pre_edit_logits = model(**ex)
            post_edit_logits = edited_model(**ex)

            pre_edit_dict.append(get_log_probs(
                pre_edit_logits, ex['labels'], shift=True))
            post_edit_dict.append(get_log_probs(
                post_edit_logits, ex['labels'], shift=True))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None

    if specificity_batches is not None and dataset_name is not None:
        pre_loc_dicts, post_loc_dicts = \
        compute_specificity_entity_inferences(model, edited_model,
                                              specificity_batches,
                                              shift=True)

    del edited_model

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)


def mend_t5_entity_inferences(
        batch, model, specificity_batches=None, dataset_name=None):

    # Edit a model
    edited_model, model_info = model.edit(batch["definition"])  # definition

    with torch.no_grad():
        n_probe_labels = batch['edit_inner'][0]['labels']['input_ids'].size(0)
        pre_edit_dict = []
        post_edit_dict = []
        for i in range(n_probe_labels):
            ex = batch["edit_inner"][0]['probe_sentence']
            ex['labels'] = batch["edit_inner"][0]['labels']['input_ids'][
                i].unsqueeze(0)
            ex['decoder_attention_mask'] = batch["edit_inner"][0]['labels'][
                'attention_mask'][i].unsqueeze(0)

            pre_edit_logits = model(**ex)
            post_edit_logits = edited_model(**ex)

            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, ex['labels'], shift=False))
            post_edit_dict.append(
                get_log_probs(post_edit_logits, ex['labels'], shift=False))

    pre_loc_dicts = None
    post_loc_dicts = None
    pre_loc_logits = None
    post_loc_logits = None
    if specificity_batches is not None and dataset_name is not None:
        pre_loc_dicts, post_loc_dicts = \
        compute_specificity_entity_inferences(model, edited_model,
                                              specificity_batches,
                                              shift=False)

    return (pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict,
            pre_loc_logits, post_loc_logits, pre_loc_dicts, post_loc_dicts)