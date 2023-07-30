import os
import json
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


def main():

    # Choose from 'ft_per_ex', 'prepend_def', 'mend'
    ki_method = 'mend'

    # Pretrained model. Use this together with 'ft'.
    ft_model_name = None
    model_info = 'ecbd'

    # Choose a unique experiment name
    exp_name = f'ecbd/t5/final/{ki_method}_{model_info}'

    exp_dir = os.path.join(ROOT_DIR, 'output', exp_name)

    os.makedirs(exp_dir, exist_ok=False)

    data_dir = os.path.join(ROOT_DIR, 'data')

    data_files = [
        os.path.join(data_dir, 'ecbd/all_ent_2020_2021_np_easy.json'),
        os.path.join(data_dir, 'ecbd/all_ent_2020_2021_random_easy.json')
    ]

    train_params = {
        "EXP_NAME": exp_name,
        "EXP_DIR": exp_dir,
        "KI_METHOD": ki_method,
        "BASE_MODEL": "t5-large",  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 1,  # training batch size
        "VALID_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 20,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-5,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 32,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
        "AUGMENT": False,
        "NUM_AUGMENT": 1,
        "CONCEPTNET": False,
        "X_IS_Y": False,
        "MASKING_STRATEGY": 'random',
        "USE_NLL_MASKS": True,
        "TOPK": 1,
        "MEMORY_RETRIEVAL": False,
        "TRAIN_ON_PROBE": False,
        "COMPUTE_SPECIFICITY": True,
        "FREEZE_LAYERS": False,
        "REG_TYPE": None,
        "ALPHA": 0.0
    }

    # Print params
    for k, v in train_params.items():
        print('{:>24}: {}'.format(k, v))

    device = torch.device("cuda:1")
    set_seed(train_params['SEED'])

    results_dict = run_edit.run_experiment(ki_method,
                                           ft_model_name,
                                           'ecbd',
                                           data_files,
                                           device,
                                           train_params,
                                           random_def=None,
                                           oracle_ft=False)

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    with open(os.path.join(exp_dir, 'train_params.json'), 'w') as f:
        json.dump(train_params, f)

    all_pre_results = []
    all_post_results = []
    all_sim_scores = []
    all_specificity_pre_res = []
    all_specificity_post_res = []
    for data_name, results in results_dict.items():
        pre_results = [res['pre'] for res in results]
        post_results = [res['post'] for res in results]
        sim_scores = [res['sim_scores'] for res in results]
        pre_perp = metrics.compute_total_perplexity(pre_results)
        post_perp = metrics.compute_total_perplexity(post_results)
        all_pre_results.extend(pre_results)
        all_post_results.extend(post_results)
        all_sim_scores.extend(sim_scores)

        diff = [
            post[0] - pre[0] for pre, post in zip(pre_results, post_results)]

        bleu_scores = [s['bleu_score']['bleu'] for s in sim_scores]
        bert_scores = [s['bert_score']['f1'][0] for s in sim_scores]
        bleurt_scores = [s['bleurt_score']['scores'][0] for s in sim_scores]
        meteor_scores = [s['meteor_score']['meteor'] for s in sim_scores]

        corr_bleu = stats.pearsonr(diff, bleu_scores)[0]
        corr_bert = stats.pearsonr(diff, bert_scores)[0]
        corr_bleurt = stats.pearsonr(diff, bleurt_scores)[0]
        corr_meteor = stats.pearsonr(diff, meteor_scores)[0]

        print('{}:\nPerplexity: Pre = {:.4f}, Post = {:.4f}'.format(
            data_name, pre_perp, post_perp))
        print(
            'Correlation with gains: BLEU = {:.4f}, BERTScore = {:.4f}, '
            'BLEURT = {:.4f}, METEOR = {:.4f}'.format(
            corr_bleu, corr_bert, corr_bleurt, corr_meteor))

        if train_params['COMPUTE_SPECIFICITY']:

            specificity_pre = [[r['pre'] for r in res['specificity']] for res
                               in results]
            specificity_post = [[r['post'] for r in res['specificity']] for res
                               in results]
            specificity_pre_perp = [metrics.compute_total_perplexity(s) for s
                                    in specificity_pre]
            specificity_post_perp =  [metrics.compute_total_perplexity(s) for s
                                    in specificity_post]
            all_specificity_pre_res.extend(specificity_pre)
            all_specificity_post_res.extend(specificity_post)

            # assert len(results) == len(all_specificity_pre_res) == len(
            #     all_specificity_post_res)
            for idx, o in enumerate(results):
                o['specificity_pre_acc'] = specificity_pre_perp[idx]
                o['specificity_post_acc'] = specificity_post_perp[idx]

        result_dict = {res['ex_id']: res for res in results}

        data_utils.write_results_ecbd(
            result_dict,
            data_name,
            os.path.join(
                exp_dir, data_name.split('/')[-1].rstrip('.json') + '.txt'),
            train_params['BASE_MODEL'])

    total_pre_perplexity = metrics.compute_total_perplexity(all_pre_results)
    total_post_perplexity = metrics.compute_total_perplexity(all_post_results)

    print('Total Perplexity: Pre = {:.4f}, Post = {:.4f}'.format(
        total_pre_perplexity, total_post_perplexity))

    total_specificity_pre_perplexity = metrics.compute_total_perplexity(
        [y for x in all_specificity_pre_res for y in x])
    total_specificity_post_perplexity = metrics.compute_total_perplexity(
        [y for x in all_specificity_post_res for y in x])

    print('Total Specificity: Pre {:.4f}, Post = {:.4f}'.format(
        total_specificity_pre_perplexity, total_specificity_post_perplexity))

    all_diff = [post[0] - pre[0] for pre, post in zip(all_pre_results,
                                                      all_post_results)]

    all_numbers = {
        'pre_perplexity': total_pre_perplexity,
        'post_perplexity': total_post_perplexity,
        'pre_specificity': total_specificity_pre_perplexity,
        'post_specificity': total_specificity_post_perplexity
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(all_numbers, f)

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
    main()
