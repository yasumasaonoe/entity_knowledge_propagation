import os
import json
import numpy as np
import torch
import sys
import pickle
from scipy import stats

sys.path.append('/mnt/data1/yasu/knowledge_injection/code')
from transformers import set_seed
from src import metrics
from src import run_edit
from src import data_utils

ROOT_DIR = '/mnt/data1/yasu/knowledge_injection/code'

def main(data_file):

    # Choose from 'ft_per_ex', 'mend', 'prepend_def'
    ki_method = 'mend'

    # Pretrained model. Use this together with 'ft'.
    ft_model_name = None
    model_info = '_ei'

    # Choose a unique experiment name
    exp_name = f'entity_inference/gpt/final/mend_fixed/{ki_method}_{model_info}'

    exp_dir = os.path.join(ROOT_DIR, 'output', exp_name)

    os.makedirs(exp_dir, exist_ok=False)

    data_dir = os.path.join(ROOT_DIR, 'data/entity_inferences')

    data_files = [
        os.path.join(data_dir, 'disaster_explicit_attribute_independent.json'),
        os.path.join(data_dir, 'disaster_implicit_attribute_independent.json'),
        os.path.join(data_dir, 'fake_person_explicit_attribute_dependent.json'),
        os.path.join(
            data_dir,
            'fake_person_implicit_attribute_dependent_adjective.json'),
        os.path.join(data_dir, 'tv_show_explicit_attribute_dependent.json'),
        os.path.join(data_dir,
                     'tv_show_implicit_attribute_dependent_adjective.json'),
    ]

    train_params = {
        "EXP_NAME": exp_name,
        "EXP_DIR": exp_dir,
        "KI_METHOD": ki_method,
        "BASE_MODEL": "gpt-neo-1.3B",  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 1,  # training batch size
        "VALID_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 1,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 5e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
        "AUGMENT": False,
        "NUM_AUGMENT": 1,
        "AUGMENT_GENERIC": False,
        "AUGMENT_SPECIFIC": False,
        "X_IS_Y": False,
        "MASKING_STRATEGY": 'random',
        "USE_NLL_MASKS": False,
        "TOPK": 1,
        "MEMORY_RETRIEVAL": False,
        "TRAIN_ON_PROBE": False,
        "COMPUTE_SPECIFICITY": True,
        "FREEZE_LAYERS": False,
        "REG_TYPE": '',
        "ALPHA": 0.0
    }

    # Print params
    for k, v in train_params.items():
        print('{:>24}: {}'.format(k, v))

    device = torch.device("cuda:0")
    set_seed(train_params['SEED'])

    results_dict = run_edit.run_experiment(ki_method,
                                           ft_model_name,
                                           'entity_inferences',
                                           data_files,
                                           device,
                                           train_params,
                                           random_def=None,
                                           oracle_ft=False)

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    with open(os.path.join(exp_dir, 'train_params.json'), 'w') as f:
        json.dump(results_dict, f)

    all_results = []
    all_probs = []
    all_sim_scores = []
    all_specificity_pre_acc = []
    all_specificity_post_acc = []
    for data_name, output in results_dict.items():
        results = [res['results']for res in output]
        probs = [res['probs'] for res in output]
        sim_scores = [res['sim_scores'] for res in output]
        scores = metrics.aggregate_results(results)
        pre_acc, post_acc, n_examples = metrics.compute_top1_accuracy(probs)
        all_results.extend(results)
        all_probs.extend(probs)
        all_sim_scores.extend(sim_scores)

        gold_probs = []
        for prob, label in probs:
            prob_d = {p[0]:p for p in prob}
            gold_probs.append(prob_d[label])

        diff = [p[-1] - p[-2] for p in gold_probs]

        bleu_scores = [s['bleu_score']['bleu'] for s in sim_scores]
        bert_scores = [s['bert_score']['f1'][0] for s in sim_scores]
        bleurt_scores = [s['bleurt_score']['scores'][0] for s in sim_scores]
        meteor_scores = [s['meteor_score']['meteor'] for s in sim_scores]

        corr_bleu = stats.pearsonr(diff, bleu_scores)[0]
        corr_bert = stats.pearsonr(diff, bert_scores)[0]
        corr_bleurt = stats.pearsonr(diff, bleurt_scores)[0]
        corr_meteor = stats.pearsonr(diff, meteor_scores)[0]

        print('{}:\nAccuracy: Pre = {:.4f}, Post = {:.4f}'.format(
            data_name, pre_acc, post_acc))
        print(scores)
        print(
            'Correlation with gains: BLEU = {:.4f}, BERTScore = {:.4f}, '
            'BLEURT = {:.4f}, METEOR = {:.4f}'.format(
            corr_bleu, corr_bert, corr_bleurt, corr_meteor))

        if train_params['COMPUTE_SPECIFICITY']:

            specificity = [[r['probs'] for r in res['specificity']] for res in
                           output]
            specificity_acc = [metrics.compute_top1_accuracy(sp) for sp in
                                   specificity]
            specificity_pre_acc = [s[0] for s in specificity_acc]
            specificity_post_acc = [s[1] for s in specificity_acc]
            all_specificity_pre_acc.extend(specificity_pre_acc)
            all_specificity_post_acc.extend(specificity_post_acc)

            assert len(output) == len(specificity_acc) == len(specificity_post_acc)
            for idx, o in enumerate(output):
                o['specificity_pre_acc'] = specificity_pre_acc[idx]
                o['specificity_post_acc'] = specificity_post_acc[idx]

        result_dict = {res['ex_id']: res for res in output}

        data_utils.write_results_entity_inferences(
            result_dict,
            data_name,
            os.path.join(
                exp_dir, data_name.split('/')[-1].rstrip('.json') + '.txt'),
            train_params['BASE_MODEL'])

    total_pre_acc, total_post_acc, total_n_examples  \
        = metrics.compute_top1_accuracy(all_probs)

    print('Total Accuracy: Pre = {:.4f}, Post = {:.4f}'.format(
        total_pre_acc, total_post_acc))

    print('Total Specificity: Pre {:.4f}, Post = {:.4f}'.format(
        np.mean(all_specificity_pre_acc), np.mean(all_specificity_post_acc)))

    all_gold_probs = []
    for prob, label in all_probs:
        prob_d = {p[0]: p for p in prob}
        all_gold_probs.append(prob_d[label])

    all_diff = [p[-1] - p[-2] for p in all_gold_probs]

    bleu_scores = [s['bleu_score']['bleu'] for s in all_sim_scores]
    bert_scores = [s['bert_score']['f1'][0] for s in all_sim_scores]
    bleurt_scores = [s['bleurt_score']['scores'][0] for s in all_sim_scores]
    meteor_scores = [s['meteor_score']['meteor'] for s in all_sim_scores]
    corr_bleu = stats.pearsonr(all_diff, bleu_scores)[0]
    corr_bert = stats.pearsonr(all_diff, bert_scores)[0]
    corr_bleurt = stats.pearsonr(all_diff, bleurt_scores)[0]
    corr_meteor = stats.pearsonr(all_diff, meteor_scores)[0]

    print(
        'Correlation with gains: BLEU = {:.4f}, BERTScore = {:.4f}, '
        'BLEURT = {:.4f}, METEOR = {:.4f}'.format(
            corr_bleu, corr_bert, corr_bleurt, corr_meteor))


if __name__ == '__main__':

    data_files = [
        '{}/disaster_explicit_attribute_independent.json',
        '{}/disaster_implicit_attribute_independent.json',
        '{}/fake_person_explicit_attribute_dependent.json',
        '{}/fake_person_implicit_attribute_dependent_adjective.json',
        '{}/tv_show_explicit_attribute_dependent.json',
        '{}/tv_show_implicit_attribute_dependent_adjective.json'
    ]
    for data_file in data_files:
        main(data_file)
