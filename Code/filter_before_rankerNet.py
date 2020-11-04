import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from evaluation_on_confusionM import get_f1, compute_p, compute_r, compute_acc
import utils
import argparse
import csv


def get_onehit(samples, span_score=-10000, null_odds=10000):
    filter_sample = []
    one_hit = 0
    num_neg = 0
    five_hit = 0
    p = 0
    r = 0
    for sample in samples:
        if sample[0]['span_score'] > span_score and sample[0]['null_odds'] < null_odds:
            filter_sample.append(sample)

    for sample in filter_sample:

        if len(sample) == 1 and sample[0]['null_odds'] == 1000:
            num_neg += 1
            continue

        if sample[0]['target'] == 1:
            one_hit += 1
        for res in sample[:5]:
            if res['target'] == 1:
                five_hit += 1
                break

    if len(filter_sample) == 0:
        return 0, 0, 0, 0
    if one_hit == 0 and num_neg == 0:
        return len(filter_sample), one_hit / len(filter_sample), 0, 0
    p = compute_p(one_hit, len(filter_sample) - num_neg - one_hit)
    r = compute_r(one_hit, num_neg)
    return len(filter_sample), one_hit / len(filter_sample), p, r


def formulate_params(params):
    return float("%.1f" % round(params, 1))


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


def get_all_samples(relations, path_relations, args_type, args_phase):
    print('======== Getting all samples from {} relations ========'.format(len(relations)))
    all_samples = []
    all_queries = []
    len_sample = 0
    one_hit = 0
    TP = 0
    FN = 0
    FN_N = 0
    FN_P = 0
    TN = 0
    FP = 0
    five_hit = 0
    three_hit = 0
    max_span_score = -10000
    min_span_score = 10000
    max_null_odd = -10000
    min_null_odd = 10000
    negative_sample = 0
    stats = dict()
    metric_stats = dict()

    for relation in relations:
        if args_type == 'neg':
            queries = []

        elif args_type == 'pos':
            if args_phase == 'known':
                with open(os.path.join('./data/splits/known/', '{}_test.json'.format(relation)), 'r') as f:
                    queries = json.load(f)
                test4Json = 'test'
            elif args_phase == 'zero_shot':
                with open(os.path.join('./data/splits/zero_shot/', '{}_test.json'.format(relation)), 'r') as f:
                    queries = json.load(f)
                test4Json = 'test'
        elif args_type == 'all':
            queries = []

        all_queries.extend(queries)

        if args_type == 'pos':
            if os.path.exists(
                    os.path.join(path_relations, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, test4Json))):
                with open(os.path.join(path_relations, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, test4Json)),
                          'r') as f:
                    for index, line in enumerate(f):
                        len_sample += 1
                        result = json.loads(line.strip())
                        all_samples.append(result)
                        if len(result) == 1 and result[0]['null_odds'] == 1000:  # => NEGATIVE
                            if queries[index]['is_impossible']:  # FALSE
                                FN += 1
                                FN_N += 1
                            elif not queries[index]['is_impossible']:
                                TN += 1
                            negative_sample += 1
                            continue
                        # get the max/min of span score and the same for null_odds
                        for pred in result:
                            if (pred['null_odds'] != 1000):
                                if (pred['null_odds'] > max_null_odd):
                                    max_null_odd = pred['null_odds']
                                if (pred['null_odds'] < min_null_odd):
                                    min_null_odd = pred['null_odds']
                                if (pred['span_score'] > max_span_score):
                                    max_span_score = pred['span_score']
                                if (pred['span_score'] < min_span_score):
                                    min_span_score = pred['span_score']

                        # if the first prediction has the right answer add 1 to len_H@1
                        # Make one or more preds and first target => 1
                        if result[0]['target'] == 1:  # => POSITIVE
                            if queries[index]['is_impossible']:  # ==> FALSE
                                FP += 1
                            else:  # => TRUE
                                TP += 1
                            one_hit += 1
                        # Make wrong preds
                        elif result[0]['target'] == 0:  # ==> Negative
                            if queries[index]['is_impossible'] and result[0]['text'] == "":  # ==> False + Negative
                                FN += 1
                                FN_N += 1
                            elif queries[index]['is_impossible'] and result[0]['text'] != "":
                                FN += 1
                                FN_P += 1
                            elif not queries[index]['is_impossible']:  # ==> True + Negative
                                TN += 1
                        for res in result[:5]:
                            if res['target'] == 1:
                                five_hit += 1
                                break
                        for res in result[:3]:
                            if res['target'] == 1:
                                three_hit += 1
                                break

        elif args_type == 'all':
            all_type = ['test', 'neg']
            all_feat_folder = './out/features/trainonknown/known/'
            all_data_folder = './data/splits/known/'
            if args_phase == 'known':
                pass
            elif args_phase == 'zero_shot':
                all_feat_folder = './out/features/trainonknown/zero_shot/'
                all_data_folder = './data/splits/zero_shot/'

            for type in all_type:
                if type == 'test':
                    with open(os.path.join(all_data_folder, '{}_{}.json'.format(relation, type)), 'r') as f:
                        queries = json.load(f)
                        all_queries.extend(queries)
                    if os.path.exists(
                            os.path.join(all_feat_folder, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, type))):
                        with open(
                                os.path.join(all_feat_folder, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, type)),
                                'r') as f:
                            for index, line in enumerate(f):
                                len_sample += 1
                                result = json.loads(line.strip())
                                all_samples.append(result)
                                if len(result) == 1 and result[0]['null_odds'] == 1000:  # => NEGATIVE
                                    if queries[index]['is_impossible'] == True:  # FALSE
                                        FN += 1
                                        FN_N += 1
                                    elif queries[index]['is_impossible'] == False:
                                        TN += 1
                                    negative_sample += 1
                                    continue
                                # get the max/min of span score and the same for null_odds
                                for pred in result:
                                    if (pred['null_odds'] != 1000):
                                        if (pred['null_odds'] > max_null_odd):
                                            max_null_odd = pred['null_odds']
                                        if (pred['null_odds'] < min_null_odd):
                                            min_null_odd = pred['null_odds']
                                        if (pred['span_score'] > max_span_score):
                                            max_span_score = pred['span_score']
                                        if (pred['span_score'] < min_span_score):
                                            min_span_score = pred['span_score']

                                # if the first prediction has the right answer add 1 to len_H@1
                                # Make one or more preds and first target => 1
                                if result[0]['target'] == 1:  # => POSITIVE
                                    if queries[index]['is_impossible']:  # ==> FALSE
                                        FP += 1
                                    else:  # => TRUE
                                        TP += 1
                                    one_hit += 1
                                # Make wrong preds
                                elif result[0]['target'] == 0:  # ==> Negative
                                    if queries[index]['is_impossible'] == True and result[0][
                                        'text'] == "":  # ==> False + Negative
                                        FN += 1
                                        FN_N += 1
                                    elif queries[index]['is_impossible'] == True and result[0]['text'] != "":
                                        FN += 1
                                        FN_P += 1
                                    elif queries[index]['is_impossible'] == False:  # ==> True + Negative
                                        TN += 1
                                for res in result[:5]:
                                    if res['target'] == 1:
                                        five_hit += 1
                                        break
                                for res in result[:3]:
                                    if res['target'] == 1:
                                        three_hit += 1
                                        break
                elif type == 'neg':
                    with open(os.path.join(all_data_folder, '{}_{}.json'.format(relation, type)), 'r') as f:
                        queries = json.load(f)
                        all_queries.extend(queries)
                    if os.path.exists(
                            os.path.join(all_feat_folder, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, type))):
                        with open(
                                os.path.join(all_feat_folder, '{}-{}-kw_sent-feat-batch-0.txt'.format(relation, type)),
                                'r') as f:
                            for index, line in enumerate(f):
                                len_sample += 1
                                result = json.loads(line.strip())
                                all_samples.append(result)

                                if result[0]['text'] == '':
                                    FN += 1
                                    FN_N += 1
                                elif result[0]['text'] != '':
                                    FN_P += 1
                                    FN += 1

        elif args_type == 'neg':
            if os.path.exists(
                    os.path.join(path_relations, '{}-neg-kw_sent-feat-batch-0.txt'.format(relation))):
                with open(os.path.join(path_relations, '{}-neg-kw_sent-feat-batch-0.txt'.format(relation)),
                          'r') as f:
                    for index, line in enumerate(f):
                        len_sample += 1
                        result = json.loads(line.strip())
                        all_samples.append(result)

                        if result[0]['text'] == '':
                            FN += 1
                            FN_N += 1
                        elif result[0]['text'] != '':
                            FN_P += 1
                            FN += 1

    assert len_sample == len(all_samples)
    assert (TP + TN + FP + FN) == len_sample
    assert (FN_N + FN_P) == FN

    stats['max_sc'], stats['min_sc'], stats['max_no'], stats['min_no'], stats['negative_num'] = formulate_params(
        max_span_score), formulate_params(min_span_score), formulate_params(max_null_odd), formulate_params(min_null_odd), TN + FN
    metric_stats['TP'], metric_stats['TN'], metric_stats['FP'], metric_stats['FN'], metric_stats['FN_N'], \
        metric_stats['FN_P'], metric_stats['F1'], metric_stats['LEN'] = TP, TN, FP, FN, FN_N, FN_P, (TP+FN_N)/(TP+TN+FP+FN), TP+TN+FP+FN
    print(stats)
    return all_samples, all_queries, metric_stats


def print_metrics_neg(frequencies, all_samples):
    range_max = [99999999, 1, 10, 99999999]
    range_min = [-1, 1, 2, 100]
    name_of_round = ['FULL', 'ONE-SHOT', 'FEW_SHOT', 'FREQUENT']
    known_neg_stats = dict()
    for i in range(4):
        len_effective = 0
        FN_P = 0
        FN_N = 0


        for idx, sample in enumerate(all_samples):
            if frequencies[idx] < range_min[i]:
                continue
            if frequencies[idx] > range_max[i]:
                continue

            len_effective += 1

            if sample[0]['null_odds'] < args.threshold:
                pass
            else:
                FN_N += 1
                continue

            if sample[0]['text'] == '':
                FN_N += 1
            elif sample[0]['text'] != '':
                FN_P += 1

        assert len_effective == (FN_N + FN_P)
        if args.type != 'both':
            print(f' == INFO({name_of_round[i]}) == length: {len_effective} , FN_N: {FN_N} , FN_P: {FN_P} '
                  f'F1-SCORE: {FN_N / (FN_N + FN_P)}', end='\n')
        else:
            known_neg_stats[name_of_round[i]] = dict()
            known_neg_stats[name_of_round[i]]['FN_P'], known_neg_stats[name_of_round[i]]['FN_N'], \
                known_neg_stats[name_of_round[i]]['F1'] = FN_P, FN_N, FN_N / (FN_P + FN_N)

    if args.type == 'both':
        return known_neg_stats


def print_metrics(frequencies, all_samples, all_queries):
    range_max = [99999999, 99999999, 10, 1]
    range_min = [-1, 100, 2, 1]
    name_of_round = ['FULL', 'FREQUENT', 'FEW_SHOT', 'ONE-SHOT']
    known_pos_stats = dict()
    for i in range(4):
        TP, TN, FP, FN, FN_N, FN_P = 0, 0, 0, 0, 0, 0
        len_effective = 0
        hit_one = 0
        hit_five = 0
        for idx, sample in enumerate(all_samples):
            if frequencies[idx] < range_min[i]:
                continue
            if frequencies[idx] > range_max[i]:
                continue
            len_effective += 1
            # args.threshold
            if sample[0]['null_odds'] < args.threshold:
                pass
            else:
                if all_queries[idx]['is_impossible']:
                    FN += 1
                    FN_N += 1
                    continue
                else:
                    TN += 1
                    continue

            if len(sample) == 1 and sample[0]['null_odds'] == 1000:  # => NEGATIVE
                if all_queries[idx]['is_impossible']:  # FALSE
                    FN += 1
                    FN_N += 1
                elif not all_queries[idx]['is_impossible']:
                    TN += 1
                continue

            if sample[0]['target'] == 1:  # => POSITIVE
                if all_queries[idx]['is_impossible']:  # ==> FALSE
                    FP += 1
                else:  # => TRUE
                    TP += 1
            # Make wrong preds
            elif sample[0]['target'] == 0:  # ==> Negative
                if all_queries[idx]['is_impossible'] and sample[0]['text'] == "":  # ==> False + Negative
                    FN += 1
                    FN_N += 1
                elif all_queries[idx]['is_impossible'] and sample[0]['text'] != "":
                    FN += 1
                    FN_P += 1
                elif not all_queries[idx]['is_impossible']:  # ==> True + Negative
                    TN += 1

            if sample[0]['target'] == 1:  # => POSITIVE
                hit_one += 1
            # elif sample[0]['target'] == 0 and sample[0]['null_odds'] == 1000:
            #    hit_one += 1
            #    hit_five += 1
            #    continue
            for res in sample[:5]:
                if res['target'] == 1:
                    hit_five += 1
                    break

        assert len_effective == (TP + TN + FP + FN)
        assert FN == FN_P + FN_N
        if args.type != 'both':
            print(
                f' == INFO({name_of_round[i]}) == length: {len_effective} , H@1: {hit_one / len_effective} ,'
                f' H@5: {hit_five / len_effective}, TP: {TP}, FN_N: {FN_N}, F1: {(TP+FN_N)/(TP+TN+FP+FN)}',
                end='\n')
        else:
            known_pos_stats[name_of_round[i]] = dict()
            known_pos_stats[name_of_round[i]]['TP'], known_pos_stats[name_of_round[i]]['FN_N'], \
                known_pos_stats[name_of_round[i]]['F1'], known_pos_stats[name_of_round[i]]['LEN'] = TP, FN_N, (TP+FN_N)/(TP+TN+FP+FN), (TP+TN+FP+FN)

    if args.type == 'both':
        return known_pos_stats


def print_metrics_zero_shot(relations, feature_dir):
    name_of_round = ['FULL', 'UNSEEN_HEAD', 'UNSEEN_TAIL', 'UNSEEN_HEAD_TAIL', 'INFREQUENT']
    data_path = os.path.join('data', 'splits', 'zero_shot')
    with open(os.path.join('data', 'train', 'known', 'seen_entities.json'), 'r') as f:
        ent = json.load(f)
        all_heads = ent['heads']
        all_tails = ent['tails']

    all_samples, is_rare, seen_head, seen_tail = utils.load_full_zs(relations, feature_dir, data_path, all_heads,
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
        hit_five = 0
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

            if sample[0]['null_odds'] < args.threshold:
                pass
            else:
                if all_queries[idx]['is_impossible']:
                    FN += 1
                    FN_N += 1
                    continue
                else:
                    TN += 1
                    continue


            if len(sample) == 1 and sample[0]['null_odds'] == 1000:  # => NEGATIVE
                if all_queries[idx]['is_impossible']:  # FALSE
                    FN += 1
                    FN_N += 1
                elif not all_queries[idx]['is_impossible']:
                    TN += 1
                continue

            if sample[0]['target'] == 1:  # => POSITIVE
                if all_queries[idx]['is_impossible']:  # ==> FALSE
                    FP += 1
                else:  # => TRUE
                    TP += 1
            # Make wrong preds
            elif sample[0]['target'] == 0:  # ==> Negative
                if all_queries[idx]['is_impossible'] and sample[0]['text'] == "":  # ==> False + Negative
                    FN += 1
                    FN_N += 1
                elif all_queries[idx]['is_impossible'] and sample[0]['text'] != "":
                    FN += 1
                    FN_P += 1
                elif not all_queries[idx]['is_impossible']:  # ==> True + Negative
                    TN += 1

            if sample[0]['target'] == 1:  # => POSITIVE
                hit_one += 1

            for res in sample[:5]:
                if res['target'] == 1:
                    hit_five += 1
                    break
        assert len_effective == (TP + TN + FP + FN)
        assert FN == FN_P + FN_N
        if args.type != 'both':
            print(f' == INFO({name_of_round[i]}) == LENGTH: {len_effective} , H@1: {hit_one / len_effective} , '
                  f'H@5: {hit_five / len_effective} , TP: {TP}, FN_N: {FN_N}, '
                  f'F1: {(TP+FN_N)/(TP+TN+FP+FN)}', end='\n')
        else:

            zero_shot_pos_stats[name_of_round[i]] = dict()
            zero_shot_pos_stats[name_of_round[i]]['TP'], zero_shot_pos_stats[name_of_round[i]]['FN_N'], \
                zero_shot_pos_stats[name_of_round[i]]['F1'],  zero_shot_pos_stats[name_of_round[i]]['LEN'] = TP, FN_N, (TP+FN_N)/(TP+TN+FP+FN), (TP+TN+FP+FN)

    if args.type == 'both':
        return zero_shot_pos_stats


def print_metrics_zero_shot_neg(relations, feature_dir):
    name_of_round = ['FULL', 'UNSEEN_HEAD', 'UNSEEN_TAIL', 'UNSEEN_HEAD_TAIL', 'INFREQUENT']
    data_path = os.path.join('data', 'splits', 'zero_shot')
    with open(os.path.join('data', 'train', 'known', 'seen_entities.json'), 'r') as f:
        ent = json.load(f)
        all_heads = ent['heads']
        all_tails = ent['tails']

    all_samples, is_rare, seen_head, seen_tail = utils.load_full_zs_neg(relations, feature_dir, data_path, all_heads,
                                                                    all_tails)
    all_queries = []
    for relation in relations:
        with open(os.path.join(data_path, '{}_neg.json'.format(relation)), 'r') as f:
            queries = json.load(f)
            all_queries.extend(queries)
    zero_shot_neg_stats = dict()
    for i in range(5):
        len_effective = 0
        FN_N, FN_P = 0, 0
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

            if sample[0]['null_odds'] < args.threshold:
                pass
            else:
                FN_N += 1
                continue

            if sample[0]['text'] == '':
                FN_N += 1
            elif sample[0]['text'] != '':
                FN_P += 1
        assert len_effective == (FN_N + FN_P)
        if args.type != 'both':
            print(f' == INFO({name_of_round[i]}) == length: {len_effective} , FN_N: {FN_N} , FN_P: {FN_P} '
                  f'F1-SCORE: {FN_N / (FN_N + FN_P)}', end='\n')
        else:
            zero_shot_neg_stats[name_of_round[i]] = dict()
            zero_shot_neg_stats[name_of_round[i]]['FN_P'], zero_shot_neg_stats[name_of_round[i]]['FN_N'], \
                zero_shot_neg_stats[name_of_round[i]]['F1'] = FN_P, FN_N, FN_N / (FN_P + FN_N)

    if args.type == 'both':
        return zero_shot_neg_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='pos', help='evaluation on neg/pos/all')
    parser.add_argument('--phase', type=str, default='known', help='known/zero_shot part')
    parser.add_argument('--threshold', type=float, default=-0.0, help='threshold')

    args = parser.parse_args()
    val_dir = os.path.join('./out/features/trainonknown', args.phase)
    if args.type == 'neg':
        relations = sorted(list(set([f[:-len('_neg_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                 f.endswith('_neg_kw_sent_meta_results.json')])))
    elif args.type == 'pos' or args.type == 'all':
        relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                     f.endswith('_test_kw_sent_meta_results.json')])))
    else:
        relations = []

    # assert 0 != len(relations)
    if args.type != 'both':
        all_samples, all_queries, stats = get_all_samples(relations, val_dir, args.type, args.phase)
        if args.type != 'neg':
            assert len(all_samples) == len(all_queries)

        print(stats)

    elif args.type == 'both' and args.phase == 'known':
        name_of_round = ['FULL', 'FREQUENT', 'FEW_SHOT', 'ONE-SHOT']
        relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                     f.endswith('_test_kw_sent_meta_results.json')])))
        all_samples, all_queries, stats = get_all_samples(relations, val_dir, 'pos', 'known')
        freqs = get_frequencies(relations, val_dir, 'test')
        stats_1 = print_metrics(freqs, all_samples, all_queries)

        relations = sorted(list(set([f[:-len('_neg_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                     f.endswith('_neg_kw_sent_meta_results.json')])))
        all_samples_2, all_queries_2, stats_2 = get_all_samples(relations, val_dir, 'neg', 'known')
        freqs = get_frequencies(relations, val_dir, 'neg')
        stats_2 = print_metrics_neg(freqs, all_samples_2)

        csv_columns = ['FN_N', 'FN_P', 'F1']
        dict_data = [stats_2[key] for key in name_of_round]
        csv_file = 'before_known.csv'
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)

        for i in range(4):
            TP = stats_1[name_of_round[i]]['TP']
            FN_N = stats_2[name_of_round[i]]['FN_N'] + stats_1[name_of_round[i]]['FN_N']
            LENGTH = stats_1[name_of_round[i]]['LEN'] + stats_2[name_of_round[i]]['FN_N'] + stats_2[name_of_round[i]]['FN_P']
            F1_POS = stats_1[name_of_round[i]]['F1']
            F1_NEG = stats_2[name_of_round[i]]['F1']
            print(f' == INFO({name_of_round[i]}) == LENGTH: {LENGTH}({stats_1[name_of_round[i]]["LEN"]} + {stats_2[name_of_round[i]]["FN_N"] + stats_2[name_of_round[i]]["FN_P"]}) F1_POS: {F1_POS}, F1_NEG: {F1_NEG} F1_ALL: {(TP+FN_N)/(LENGTH)}')

    elif args.type == 'both' and args.phase == 'zero_shot':
        relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                     f.endswith('_test_kw_sent_meta_results.json')])))
        stats_2 = print_metrics_zero_shot_neg(relations, val_dir)
        stats_1 = print_metrics_zero_shot(relations, val_dir)

        csv_columns = ['FN_N', 'FN_P', 'F1']
        name_of_round = ['FULL', 'UNSEEN_HEAD', 'UNSEEN_TAIL', 'UNSEEN_HEAD_TAIL', 'INFREQUENT']
        dict_data = [stats_2[key] for key in name_of_round]
        csv_file = 'after_known.csv'
        with open(csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)

        for i in range(5):
            TP = stats_1[name_of_round[i]]['TP']
            FN_N = stats_2[name_of_round[i]]['FN_N'] + stats_1[name_of_round[i]]['FN_N']
            LENGTH = stats_1[name_of_round[i]]['LEN'] + stats_2[name_of_round[i]]['FN_N'] + stats_2[name_of_round[i]]['FN_P']
            F1_POS = stats_1[name_of_round[i]]['F1']
            F1_NEG = stats_2[name_of_round[i]]['F1']
            print(f' == INFO({name_of_round[i]}) == LENGTH: {LENGTH}({stats_1[name_of_round[i]]["LEN"]} + {stats_2[name_of_round[i]]["FN_N"] + stats_2[name_of_round[i]]["FN_P"]}) F1_POS: {F1_POS}, F1_NEG: {F1_NEG} F1_ALL: {(TP+FN_N)/(LENGTH)}')



    if args.type == 'pos' and args.phase == 'known':
        print('[EVALUATION MATRIX] FOR KNOWN PART')
        freqs = get_frequencies(relations, val_dir, 'test')
        print_metrics(freqs, all_samples, all_queries)
    elif args.type == 'pos' and args.phase == 'zero_shot':
        print('[EVALUATION MATRIX] FOR ZERO_SHOT PART')
        print_metrics_zero_shot(relations, val_dir)
    elif args.type == 'neg' and args.phase == 'known':
        print('[EVALUATION MATRIX] FOR KNOWN & NEGATIVE PART')
        freqs = get_frequencies(relations, val_dir, 'neg')
        print_metrics_neg(freqs, all_samples)
    elif args.type == 'neg' and args.phase == 'zero_shot':
        print('[EVALUATION MATRIX] FOR ZERO_SHOT & NEGATIVE PART')
        print_metrics_zero_shot_neg(relations, val_dir)

    """
        Get the .csv file of result
    """
    data = []
    # for i in tqdm(np.arange(stats['min_sc'], stats['max_sc'] + 0.1, 0.1)):
    #     for j in np.arange(stats['max_no'], stats['min_no'] - 0.1, -0.1):
    #         res = get_onehit(all_samples, i, j)
    #         data.append([round(i, 2), round(j, 2), round(res[1], 4), res[0], round(res[1] * res[0], 4), res[2], res[3],
    #                      get_f1(res[2], res[3])])
    #
    # df = pd.DataFrame(data=data, columns=['span_score', 'null_odd', 'H@1', 'docs_filtered', 'doc_left', 'P', 'R', 'F1'])
    # df.to_csv('./before_rankerNet.csv')
