import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import utils

import argparse

def formulate_params(params):
    return float("%.1f" % round(params, 1))


def get_all_samples(relations, path_relations):
    print('======== Getting all samples from {} relations ========'.format(len(relations)))
    all_samples = []
    all_queries = []
    len_sample = 0
    one_hit = 0
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    FN_N = 0
    FN_P = 0
    max_span_score = -10000
    min_span_score = 10000
    max_null_odd = -10000
    min_null_odd = 10000

    for relation in tqdm(relations):

        if args.phase == 'known':
            with open(os.path.join('./data/splits/known/', '{}_test.json'.format(relation)), 'r') as f:
                queries = json.load(f)

        elif args.phase == 'zero_shot':
            with open(os.path.join('./data/splits/zero_shot/', '{}_test.json'.format(relation)), 'r') as f:
                queries = json.load(f)
        all_queries.extend(queries)

        with open(os.path.join(path_relations, '{}-predictions.json'.format(
                relation)), 'r') as f:
            for index, line in enumerate(f):
                result = json.loads(line)
                # null_odds = 1000 => Bert predict nothing for this sample
                pred = result['prediction']
                len_sample += 1
                all_samples.append(pred)

                if pred['null_odds'] != 1000:
                    if pred['null_odds']>max_null_odd:
                        max_null_odd = pred['null_odds']
                    if pred['null_odds']<min_null_odd:
                        min_null_odd = pred['null_odds']
                    if pred['span_score'] > max_span_score:
                        max_span_score = pred['span_score']
                    if pred['span_score'] < min_span_score:
                        min_span_score = pred['span_score']


                if pred['null_odds'] == 1000:  # => NEGATIVE
                    if queries[index]['is_impossible'] == True:  # FALSE
                        FN += 1
                        FN_N += 1
                    elif queries[index]['is_impossible'] == False:
                        TN += 1
                    continue

                if pred['target'] == 1:  # => POSITIVE
                    if queries[index]['is_impossible']:  # ==> FALSE
                        FP += 1
                    else:  # => TRUE
                        TP += 1
                    one_hit += 1
                # Make wrong preds
                elif pred['target'] == 0:  # ==> Negative
                    if queries[index]['is_impossible'] == True and pred[
                        'text'] == "":  # ==> False + Negative
                        FN += 1
                        FN_N += 1
                    elif queries[index]['is_impossible'] == True and pred['text'] != "":
                        FN += 1
                        FN_P += 1
                    elif queries[index]['is_impossible'] == False:  # ==> True + Negative
                        TN += 1

    stats = dict()
    stats['max_sc'], stats['min_sc'], stats['max_no'], stats['min_no'] = formulate_params(max_span_score), \
        formulate_params(min_span_score), formulate_params(max_null_odd), formulate_params(min_null_odd)
    print('[INFO] TWO PARAMS: ', stats)
    statistics = dict()
    statistics['H@1'], statistics['TP'], statistics['TN'], statistics['FP'], statistics['FN'], statistics['FN_N'], \
        statistics['FN_P'], statistics['F1'], statistics['LEN'] = TP / (TN + TP + FN + FP), TP, TN, FP, FN, FN_N, FN_P,\
        (TP + FN_N) / (TN + TP + FN + FP), (TN + TP + FN + FP)
    return all_samples, all_queries, statistics



def get_frequencies(relations, path_features, type='test'):
    with open(os.path.join('data', 'train', 'known', 'train_entities_stats.json'), 'r') as f:
        frequencies = json.load(f)
    frequency_of_relation = []
    for i, relation in enumerate(relations):
        file = os.path.join(path_features, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, type))
        if not os.path.exists(file):
            print(" == skipping non existing file {} ==".format(file))
            continue
        with open(file, 'r') as f:
            i = 0
            for line in f:
                i += 1
            for _ in range(i):
                frequency_of_relation.append(frequencies[relation])

    return frequency_of_relation


def print_metrics(frequencies, all_samples, all_queries):
    range_max = [99999999, 99999999, 10, 1]
    range_min = [-1, 100, 2, 1]
    name_of_round = ['FULL', 'FREQUENT', 'FEW_SHOT', 'ONE-SHOT']
    known_pos_stats = dict()
    for i in range(4):
        TP, TN, FP, FN, FN_N, FN_P = 0, 0, 0, 0, 0, 0
        len_effective = 0
        for idx, sample in enumerate(all_samples):
            if frequencies[idx] < range_min[i]:
                continue
            if frequencies[idx] > range_max[i]:
                continue
            len_effective += 1
            # args.threshold
            if sample['null_odds'] < args.threshold:
                pass
            else:
                if all_queries[idx]['is_impossible']:
                    FN += 1
                    FN_N += 1
                    continue
                else:
                    TN += 1
                    continue

            if sample['null_odds'] == 1000:  # => NEGATIVE
                if all_queries[idx]['is_impossible']:  # FALSE
                    FN += 1
                    FN_N += 1
                elif not all_queries[idx]['is_impossible']:
                    TN += 1
                continue

            if sample['target'] == 1:  # => POSITIVE
                if all_queries[idx]['is_impossible']:  # ==> FALSE
                    FP += 1
                else:  # => TRUE
                    TP += 1
            # Make wrong preds
            elif sample['target'] == 0:  # ==> Negative
                if all_queries[idx]['is_impossible'] and sample['text'] == "":  # ==> False + Negative
                    FN += 1
                    FN_N += 1
                elif all_queries[idx]['is_impossible'] and sample['text'] != "":
                    FN += 1
                    FN_P += 1
                elif not all_queries[idx]['is_impossible']:  # ==> True + Negative
                    TN += 1

        assert len_effective == (TP + TN + FP + FN)
        assert FN == FN_P + FN_N
        if args.type != 'both':
            print(
                f' == INFO({name_of_round[i]}) == length: {len_effective}, TP: {TP}, FN_N: {FN_N}, F1: {(TP+FN_N)/(TP+TN+FP+FN)}',
                end='\n')
        else:
            known_pos_stats[name_of_round[i]] = dict()
            known_pos_stats[name_of_round[i]]['TP'], known_pos_stats[name_of_round[i]]['FN_N'], \
                known_pos_stats[name_of_round[i]]['F1'], known_pos_stats[name_of_round[i]]['LEN'] = TP, FN_N, (TP+FN_N)/(TP+TN+FP+FN), (TP+TN+FP+FN)

    if args.type == 'both':
        return known_pos_stats
    else:
        return ''


def print_metrics_zero_shot(relations, all_samples):
    name_of_round = ['FULL', 'UNSEEN_HEAD', 'UNSEEN_TAIL', 'UNSEEN_HEAD_TAIL', 'INFREQUENT']
    data_path = os.path.join('data', 'splits', 'zero_shot')
    with open(os.path.join('data', 'train', 'known', 'seen_entities.json'), 'r') as f:
        ent = json.load(f)
        all_heads = ent['heads']
        all_tails = ent['tails']

    _, is_rare, seen_head, seen_tail = utils.load_full_zs(relations, os.path.join('./out/features/trainonknown', args.phase), data_path, all_heads,
                                                                    all_tails)
    all_queries = []
    for relation in relations:
        with open(os.path.join(data_path, '{}_test.json'.format(relation)), 'r') as f:
            queries = json.load(f)
            all_queries.extend(queries)
    zero_shot_pos_stats = dict()
    for i in range(5):
        len_effective = 0
        hit_one = 0
        TP, TN, FP, FN, FN_N, FN_P = 0, 0, 0, 0, 0, 0
        for idx, sample in enumerate(all_samples):
            if name_of_round[i] == 'FULL':
                pass
            if name_of_round[i] == 'INFREQUENT' and not is_rare[idx]:
                continue
            if name_of_round[i] == 'UNSEEN_HEAD' and seen_head[idx]:
                continue
            if name_of_round[i] == 'UNSEEN_TAIL' and seen_tail[idx]:
                continue
            if name_of_round[i] == 'UNSEEN_HEAD_TAIL' and (seen_head[idx] or seen_tail[idx]):
                continue
            len_effective += 1

            if sample['null_odds'] < args.threshold:
                pass
            else:
                if all_queries[idx]['is_impossible']:
                    FN += 1
                    FN_N += 1
                    continue
                else:
                    TN += 1
                    continue

            if sample['null_odds'] == 1000:  # => NEGATIVE
                if all_queries[idx]['is_impossible']:  # FALSE
                    FN += 1
                    FN_N += 1
                elif not all_queries[idx]['is_impossible']:
                    TN += 1
                continue

            if sample['target'] == 1:  # => POSITIVE
                if all_queries[idx]['is_impossible']:  # ==> FALSE
                    FP += 1
                else:  # => TRUE
                    TP += 1
            # Make wrong preds
            elif sample['target'] == 0:  # ==> Negative
                if all_queries[idx]['is_impossible'] and sample['text'] == "":  # ==> False + Negative
                    FN += 1
                    FN_N += 1
                elif all_queries[idx]['is_impossible'] and sample['text'] != "":
                    FN += 1
                    FN_P += 1
                elif not all_queries[idx]['is_impossible']:  # ==> True + Negative
                    TN += 1

            if sample['target'] == 1:  # => POSITIVE
                hit_one += 1

        assert len_effective == (TP + TN + FP + FN)
        assert FN == FN_P + FN_N
        if args.type != 'both':
            print(f' == INFO({name_of_round[i]}) == LENGTH: {len_effective} , H@1: {hit_one / len_effective} , '
                  f' TP: {TP}, FN_N: {FN_N}, '
                  f'F1: {(TP+FN_N)/(TP+TN+FP+FN)}', end='\n')
        else:
            zero_shot_pos_stats[name_of_round[i]] = dict()
            zero_shot_pos_stats[name_of_round[i]]['TP'], zero_shot_pos_stats[name_of_round[i]]['FN_N'], \
                zero_shot_pos_stats[name_of_round[i]]['F1'],  zero_shot_pos_stats[name_of_round[i]]['LEN'] = TP, FN_N, (TP+FN_N)/(TP+TN+FP+FN), (TP+TN+FP+FN)

    if args.type == 'both':
        return zero_shot_pos_stats
    else:
        return ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='pos', help='pos/both')
    parser.add_argument('--phase', type=str, default='known', help='known/zero_shot part')
    parser.add_argument('--threshold', type=float, default=0.0, help='null_odd')

    args = parser.parse_args()
    val_dir = os.path.join('./out/predictions', args.phase)

    val_dir_old = os.path.join('./out/features/trainonknown', args.phase)

    relations = sorted(
        list(set([f[:-len('-predictions.json')] for f in os.listdir(val_dir) if f.endswith('-predictions.json')])))

    all_samples, all_queries, stats = get_all_samples(relations, val_dir)

    print(
        f'=== INFO({args.phase})=== threshold:{args.threshold} TP: {stats["TP"]} FN_N: {stats["FN_N"]} '
        f'F1: {stats["F1"]} LENGTH: {stats["LEN"]}')

    if args.phase == 'known':
        print('[EVALUATION MATRIX] FOR KNOWN PART')
        freqs = get_frequencies(relations, val_dir_old, 'test')
        stats_1 = print_metrics(freqs, all_samples, all_queries)
        name_of_round = ['FULL', 'FREQUENT', 'FEW_SHOT', 'ONE-SHOT']
        df = pd.read_csv('before_known.csv')

        for i in range(4):

            TP = stats_1[name_of_round[i]]['TP']
            FN_N = stats_1[name_of_round[i]]['FN_N'] + df['FN_N'][i]
            LENGTH = stats_1[name_of_round[i]]['LEN'] + df['FN_N'][i] + df['FN_P'][i]
            F1_POS = stats_1[name_of_round[i]]['F1']
            F1_NEG = df['F1'][i]
            F1_ALL = (TP + FN_N) / LENGTH

            print(f' == INFO({name_of_round[i]}) == LENGTH: {LENGTH}({stats_1[name_of_round[i]]["LEN"]} + {df["FN_N"][i] + df["FN_P"][i]}) F1_POS: {F1_POS}, F1_NEG: {F1_NEG} F1_ALL: {F1_ALL}')

    elif args.phase == 'zero_shot':
        print('[EVALUATION MATRIX] FOR ZERO_SHOT PART')
        stats_2 = print_metrics_zero_shot(relations, all_samples)

        name_of_round = ['FULL', 'UNSEEN_HEAD', 'UNSEEN_TAIL', 'UNSEEN_HEAD_TAIL', 'INFREQUENT']
        df = pd.read_csv('after_known.csv')
        for i in range(5):

            TP = stats_2[name_of_round[i]]['TP']
            FN_N = stats_2[name_of_round[i]]['FN_N'] + df['FN_N'][i]
            LENGTH = stats_2[name_of_round[i]]['LEN'] + df['FN_N'][i] + df['FN_P'][i]
            F1_POS = stats_2[name_of_round[i]]['F1']
            F1_NEG = df['F1'][i]
            F1_ALL = (TP + FN_N) / LENGTH

            print(f' == INFO({name_of_round[i]}) == LENGTH: {LENGTH}({stats_2[name_of_round[i]]["LEN"]} + {df["FN_N"][i] + df["FN_P"][i]}) F1_POS: {F1_POS}, F1_NEG: {F1_NEG} F1_ALL: {F1_ALL}')



    # for i in np.arange(stats['min_sc'], stats['max_sc'], 0.1):
    #     for j in np.arange(stats['max_no'], stats['min_no'], -0.1):
    #         len_filtered_samples, one_hit_filtered = get_onehit(all_samples, i , j)
    #         data.append([i, j, len_filtered_samples, one_hit_filtered])
    # df = pd.DataFrame(data=data, columns=['span_score', 'null_odd', 'doc_left', 'H@1'])
    # # path output
    # df.to_csv('./after_rankerNet.csv')
