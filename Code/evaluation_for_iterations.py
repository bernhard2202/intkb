import os
import json
from utils import bcolors
from src.retriever.utils import get_filename_for_article_id
import termplotlib as tpl
import argparse
from tqdm import tqdm
import pandas as pd


def formulate_params(params):
    return float("%.4f" % round(params, 4))


def visualize_result(x, y):
    fig = tpl.figure()
    fig.plot(x,y, xlabel='ITER_X', width=80, height=20)
    fig.show()


def label_ds(relation_dir, wiki_dir):
    print('[INFO] label_ds()')
    is_impossible = 0
    relations = [f.split('_test.json')[0] for f in os.listdir(relation_dir) if
                 f.endswith('_test.json') and f.startswith('P')]
    for relation in relations:
        tmp_num = 0
        print('Processing the relation {} [LABELING]'.format(relation))
        with open(os.path.join(relation_dir, '{}_test.json'.format(relation)), 'r') as f:
            test_queries = json.load(f)
        for query in test_queries:
            query = label_one_sample(query, wiki_dir)
            if query['is_impossible']:
                tmp_num += 1

        is_impossible += tmp_num
        print(f"[{relation}]Number of false case: {tmp_num}")
        with open (os.path.join(relation_dir, '{}_test.json'.format(relation)), 'w') as f:
            json.dump(test_queries, f)
    print(f"Number of false case(in total): {is_impossible}")


def label_one_sample(original_query, wiki_dir):
    answers = original_query['answer']
    fname = get_filename_for_article_id(original_query['wikipedia_link'].split('wiki/')[-1])
    if not os.path.exists(os.path.join(wiki_dir, fname)):
        print('DOC NOT FOUND [LABELING SAMPLE]')
    with open(os.path.join(wiki_dir, fname), 'r') as f:
        doc_string = f.read()
    for answer in answers:
        if doc_string.lower().find(answer.lower()) != -1:
            original_query['is_impossible'] = False
            return original_query

    original_query['is_impossible'] = True
    return original_query


def get_all_samples_per_iteration(feat_path, relations, data_path):
    all_samples = []
    len_sample = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FN_N = 0
    FN_P = 0
    one_hit = 0
    three_hit = 0
    five_hit = 0
    max_span_score = -10000
    min_span_score = 10000
    max_null_odd = -10000
    min_null_odd = 10000
    stats = {}

    for relation in relations:
        with open(os.path.join(data_path, '{}_test.json'.format(relation)), 'r') as f:
            queries = json.load(f)

        if os.path.exists(os.path.join(feat_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation))):
            with open(os.path.join(feat_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation)), 'r') as f:
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
                        if queries[index]['is_impossible'] == True and result[0]['text'] == "":  # ==> False + Negative
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

    stats['max_sc'], stats['min_sc'], stats['max_no'], stats['min_no'] = formulate_params(max_span_score), \
                                                                         formulate_params(min_span_score), \
                                                                         formulate_params(max_null_odd), \
                                                                         formulate_params(min_null_odd)

    # print("=== INFO === FP:{} TP:{} FN:{} TN:{} one_hit[P]:{}".format(FP, TP, FN, TN, one_hit))
    # print("== INFO (Negative) == FN_N:{} FN_P:{} ".format(FN_N, FN_P), end='')
    # print(stats)
    # if FP + TP + FN + TN != 0:
    #     print('== H@1 before cutoff: {}, H@3: {} , H@5: {} docs_len: {} =='.format((TP + FN_N) / (FP + TP + FN + TN),
    #             (three_hit + FN_N) / (FP + TP + FN + TN), (five_hit + FN_N) / (FP + TP + FN + TN), len_sample))
    # else:
    #     print('FP=TP=FN=FN=0')
    assert len_sample == (TP+FP+FN+TN)
    if len_sample != 0:
        return [formulate_params(TP/len_sample), formulate_params(three_hit/len_sample), formulate_params(five_hit/len_sample), formulate_params((TP+FN_N)/len_sample), TP, FN_N]    
    else:
        return [0, 0, 0, 0, 0, 0]


def neg_get_all_samples_per_iteration(feat_path, relations, data_path):
    len_sample = 0
    FN_N = 0
    FN_P = 0
    all_samples = []
    for relation in relations:
        with open(os.path.join(data_path, '{}_neg.json'.format(relation)), 'r') as f:
            queries = json.load(f)
        if os.path.exists(os.path.join(feat_path, '{}-neg-kw_sent-feat-batch-0.txt'.format(relation))):
            with open(os.path.join(feat_path, '{}-neg-kw_sent-feat-batch-0.txt'.format(relation)), 'r') as f:
                for index, line in enumerate(f):
                    len_sample += 1
                    result = json.loads(line.strip())
                    all_samples.append(result)

                    if result[0]['target'] == 0 and result[0]['text'] == "":
                        FN_N += 1
                    elif result[0]['target'] == 0 and result[0]['text'] != "":
                        FN_P += 1
    assert (FN_N + FN_P) == len_sample == len(all_samples)
    return [FN_N/len_sample, FN_N, FN_P]


def main():
    eval_dir = args.dir_eval
    iterations = [folder_name for folder_name in os.listdir(eval_dir) if folder_name.startswith('iter_')]

    data_path = './data/splits/zero_shot/'
    data = []
    neg_data = []
    columns = ['H@1', 'H@3', 'H@5', 'F1-SCORE', 'TP', 'FN_N', 'PROCESS']
    neg_columns = ['ACC', 'FN_N', 'FN_P', 'PROCESS']
    index = []
    current_log = ""
    neg_current_log = ""
    print(f'\n{bcolors.WARNING} >>> Got result for {len(iterations)} iterations in the {eval_dir} <<<')

    csv_path = os.path.join(eval_dir, 'df_old.csv')
    neg_csv_path = os.path.join(eval_dir, 'neg_df_old.csv')

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, index_col=0)
        neg_df_old = pd.read_csv(neg_csv_path, index_col=0)
        len_old_csv = df_old.shape[0]
        print(f' >>> Found {len_old_csv} iterations cached before, {len(iterations) - len_old_csv} more to update <<<{bcolors.ENDC}')
    else:
        df_old = pd.DataFrame(data=[], columns=['H@1', 'H@3', 'H@5', 'F1-SCORE', 'TP', 'FN_N', 'PROCESS'], index=[])
        neg_df_old = pd.DataFrame(data=[], columns=neg_columns, index=[])
        len_old_csv = 0
        print(f' >>> No cache for any iteration <<<{bcolors.ENDC}')
    print('',end='\n')
    for i in tqdm(range(len_old_csv, len(iterations))):

        #print(f'{bcolors.WARNING} Evaluation for iter_{i} {bcolors.ENDC}')


        folder_name = os.path.join(eval_dir, 'iter_{}'.format(i))
        relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')]
                                     for f in os.listdir(folder_name + '/features') if
                                     f.endswith('_test_kw_sent_meta_results.json')])))
        #print('======== Getting all samples from {} relations for {}========'.format(len(relations), 'iter_{}'.format(i)))
        # get all the preds
        # [TP/len_sample, three_hit/len_sample, five_hit/len_sample, (TP+FN_N)/len_sample, TP, FN_N]
        result_iter = get_all_samples_per_iteration(folder_name + '/features', relations, data_path)
        neg_result_iter = neg_get_all_samples_per_iteration(folder_name + '/features', relations, data_path)
        process = '[{}/50]'.format(len(relations))
        result_iter.append(process)
        neg_result_iter.append(process)

        if i != len(iterations) - 1:
            if i < 10:
                index.append('[ITER__{}]'.format(i))
            else:
                index.append('[ITER_{}]'.format(i))
            data.append(result_iter)
            neg_data.append(neg_result_iter)
        else:
            current_log = current_log + f'[ITER_{i}]  H@1:{formulate_params(result_iter[0])}  ' \
                                        f'F1-SCORE:{formulate_params(result_iter[3])}  ' \
                                        f'TP:{result_iter[4]}  FN_N:{result_iter[5]} PROCESS{process}'
            neg_current_log = neg_current_log + f'[ITER_{i} ACC: {formulate_params(neg_result_iter[0])} ' \
                                                f'FN_N: {neg_result_iter[1]} FN_P: {neg_result_iter[2]} PROCESS: {process}'

    df = pd.DataFrame(data=data, columns=columns, index=index)
    neg_df = pd.DataFrame(data=neg_data, columns=neg_columns, index=index)
    df_final = df_old.append(df)
    neg_df_final = neg_df_old.append(neg_df)
    df_final.to_csv(csv_path, columns=df.columns)
    neg_df_final.to_csv(neg_csv_path, columns=neg_df.columns)
    # print all the iterations that are finished
    print('', end='\n')
    print(df_final)

    # print latest iteration's status.
    print(f'{bcolors.OKBLUE}')
    print(current_log)
    print(f'\n[_BEST_X]  H@1:{formulate_params(df_final["H@1"].max())}  F1-SCORE:{formulate_params(df_final["F1-SCORE"].max())}  TP:{df_final["TP"].max()}')
    print(f'{bcolors.ENDC}')

    # Visualize the metrics.
    # columns = ['H@1', 'H@3', 'H@5', 'F1-SCORE', 'TP', 'FN_N', 'PROCESS']
    if args.curve_type == 'F1':
        y = df_final['F1-SCORE'].values.tolist()
    elif args.curve_type == 'H1':
        y = df_final['H@1'].values.tolist()
    elif args.curve_type == 'H3':
        y = df_final['H@3'].values.tolist()
    elif args.curve_type == 'H5':
        y = df_final['H@5'].values.tolist()
    elif args.curve_type == 'TP':
        y = df_final['TP'].values.tolist()
    else:
        raise ValueError('Wrong Input about curve_type! Should enter things among F1/H1/H3/H5/TP [F1 by default]')

    x = [i for i in range(df_final.shape[0])]

    visualize_result(x, y)
    print('', end='\n'*2)

    # For the negative part
    if args.neg == 1:
        print(neg_df_final)

        # print latest iteration's status.
        print(f'{bcolors.OKBLUE}')
        print(neg_current_log)
        print(f'\n[_BEST_X]  ACC:{formulate_params(neg_df_final["ACC"].max())}  FN_N:{neg_df_final["FN_N"].max()} ')
        print(f'{bcolors.ENDC}')

        # Visualize the metrics.
        neg_x = [i for i in range(neg_df_final.shape[0])]
        y = neg_df_final['ACC'].values.tolist()
        visualize_result(x, y)
        print('', end='\n' * 2)

    elif args.neg == 0:
        pass
    else:
        raise ValueError('Wrong --neg argument input  (0/1)')


if __name__ == '__main__':
    #label_ds('./data/splits/zero_shot', './data/wiki')
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_eval', type=str, default='./iterations', help='directory to evaluate on')
    parser.add_argument('--curve_type', type=str, default='F1', help='Curve type: H1/H3/H5/TP/F1')
    parser.add_argument('--neg', type=int, default=0, help='1 <=> both, default => only pos')


    args = parser.parse_args()


    main()
