import os
import json
from src.retriever.utils import get_filename_for_article_id
from src.retriever.sentence_retriever import retrieve_sentences_multi_kw_query
import nltk
from tqdm import tqdm
import argparse


def construct_question(relation_num):
    with open(os.path.join('data/labels', '{}_labels.json'.format(relation_num)), 'r') as f:
        query_list = json.load(f)
        query_list.sort(key=lambda s: -len(s))
    query_list = query_list[0:min(len(query_list), 5)]
    query_list = [q + ' ?' for q in query_list]
    return query_list


def retriever_for_noAns(query, doc):
    results, _ = retrieve_sentences_multi_kw_query([query], [doc], 1)
    if len(results) == 0:
        return ""
    return results[0]


def get_all_relations():
    return [f.split('_train')[0] for f in os.listdir('out/features/trainonknown/known') if f.endswith('_train_kw_sent_meta_results.json')]


def get_all_relations_zero_shot():
    return [f.split('_train')[0] for f in os.listdir('out/features/trainonknown/zero_shot') if f.endswith('_train_kw_sent_meta_results.json')]


def label_all_queries(relation_dir='data/splits/known'):
    is_impossible = 0
    total = 0
    relations = get_all_relations()

    for relation in tqdm(relations):
        tmp_num = 0
        with open(os.path.join(relation_dir, '{}_train.json'.format(relation)), 'r') as f:
            train_queries = json.load(f)
        for query in train_queries:
            total += 1
            query = label_one_query(query)
            if query['is_impossible']:
                tmp_num += 1

        is_impossible += tmp_num
        # print(f"[{relation}]Number of false case: {tmp_num}")
        with open(os.path.join(relation_dir, '{}_train.json'.format(relation)), 'w') as f:
            json.dump(train_queries, f)
    print(f"    >>>> NUMBER OF [NO_ANS] CASES: {is_impossible} out of {total} <<<<")


def label_one_query(original_query=None, wiki_dir='data/wiki'):
    answers = original_query['answer']

    file_name = get_filename_for_article_id(original_query['wikipedia_link'].split('wiki/')[-1])
    if not os.path.exists(os.path.join(wiki_dir, file_name)):
        print('DOC NOT FOUND [LABELING SAMPLE]')
    with open(os.path.join(wiki_dir, file_name), 'r') as f:
        doc_string = f.read()
    for answer in answers:
        if doc_string.lower().find(answer.lower()) != -1:
            original_query['is_impossible'] = False
            return original_query

    original_query['is_impossible'] = True
    return original_query


def get_query_for_relation(relation):
    query_file_name = 'data/splits/known/' + '{}_train.json'.format(relation)
    with open(query_file_name, 'r') as f:
        queries = json.load(f)

    return queries


def get_sample_for_relation(relation):
    samples_for_relation = []
    for i in range(2):
        feat_path = os.path.join('out/features/trainonknown/known/{}-train-kw_sent-feat-batch-{}.txt'.format(relation, i))
        if os.path.exists(feat_path):
            with open(feat_path, 'r') as f:
                for idx, line in enumerate(f):
                    result = json.loads(line.strip())
                    samples_for_relation.append(result)

    return samples_for_relation


def iterate_all_data(threshold=0.0):
    relations = get_all_relations()
    all_queries = []
    all_samples = []
    TP, TN, FP, FN = 0, 0, 0, 0
    FN_P, FN_N = 0, 0
    LEN_SAMPLES = 0
    ONE_HIT, THREE_HIT, FIVE_HIT = 0, 0, 0
    MAX_SPAN_SCORE, MIN_SPAN_SCORE, MAX_NULL_ODD, MIN_NULL_ODD = -10000, 10000, -10000, 10000
    FILTER_OUT_LEN = 0
    NEG = 0

    for relation in relations:
        queries_relation = get_query_for_relation(relation)
        samples_relation = get_sample_for_relation(relation)

        assert len(queries_relation) == len(samples_relation)

        LEN_SAMPLES += len(samples_relation)

        for index in range(len(queries_relation)):

            # threshold filter
            """
                null_odds < threshold => PASS TO NEXT OPS
                
                null_odds > threshold => no_Ans for prediction and
                            1) If indeed has_Ans => TN += 1
                            1) If indeed No_Ans => FN += 1 and FN_N += 1 
            """
            if samples_relation[index][0]['null_odds'] < threshold:
                pass
            else:
                FILTER_OUT_LEN += 1
                if queries_relation[index]['is_impossible']:
                    FN += 1
                    FN_N += 1
                    continue
                else:
                    TN += 1
                    continue

            # Prediction: Give no ans
            if len(samples_relation[index]) == 0 and samples_relation[index][0]['null_odds'] == 1000:
                NEG += 1
                # Indeed: No ans
                if queries_relation[index]['is_impossible']:
                    FN += 1
                    FN_N += 1
                # Indeed: Has ans
                elif not queries_relation[index]['is_impossible']:
                    TN += 1
                continue

            # Get statistics for the data about span_score and null_odd
            for pred in samples_relation[index]:
                if pred['null_odds'] != 1000:

                    if pred['null_odds'] > MAX_NULL_ODD:
                        MAX_NULL_ODD = pred['null_odds']
                    if pred['null_odds'] < MIN_NULL_ODD:
                        MIN_NULL_ODD = pred['null_odds']

                    if pred['span_score'] > MAX_SPAN_SCORE:
                        MAX_SPAN_SCORE = pred['span_score']
                    if pred['span_score'] < MIN_SPAN_SCORE:
                        MIN_SPAN_SCORE = pred['span_score']

            if samples_relation[index][0]['target'] == 1:
                if queries_relation[index]['is_impossible']:
                    FP += 1
                else:
                    TP += 1
                    ONE_HIT += 1

            elif samples_relation[index][0]['target'] == 0:
                if queries_relation[index]['is_impossible']:
                    FN += 1
                    if samples_relation[index][0]['text'] != '':
                        FN_P += 1
                    else:
                        FN_N += 1
                elif not queries_relation[index]['is_impossible']:
                    TN += 1

            for pred in samples_relation[index][:3]:
                if pred['target'] == 1:
                    THREE_HIT += 1
                    break

            for pred in samples_relation[index][:5]:
                if pred['target'] == 1:
                    FIVE_HIT += 1
                    break

        all_queries.extend(queries_relation)
        all_samples.extend(samples_relation)

    # print(f'[==INFO==] TP: {TP} TN: {TN} FP: {FP} FN: {FN}(FN_P: {FN_P} | FN_N: {FN_N})')
    # print(f'[==INFO==] H@1: {ONE_HIT}(TP) H@3: {THREE_HIT} H@5: {FIVE_HIT}')
    # print(f'[==INFO==] F1-SCORE (TP+FN_N)/(ALL):{(TP+FN_N)/LEN_SAMPLES}')
    # print(f'[==INFO==] LEN_SAMPLES: {LEN_SAMPLES}')
    print(f'LEN_FILTER_OUT: {FILTER_OUT_LEN}')

    stats = dict()
    stats['THRESHOLD'] = threshold
    stats['F1-SCORE'] = (TP + FN_N) / LEN_SAMPLES
    stats['TP'], stats['TN'], stats['FP'], stats['FN'] = TP, TN, FP, FN
    stats['FN_N'], stats['FN_P'] = FN_N, FN_P

    return all_queries, all_samples, stats


def generate_negative_query(relation_dir='./data/splits/known', output_dir='./neg/data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if relation_dir.split('/')[-1] == 'zero_shot':
        relations = get_all_relations_zero_shot()
    elif relation_dir.split('/')[-1] == 'known':
        relations = get_all_relations()

    for relation in tqdm(relations):
        with open(os.path.join(relation_dir, '{}_{}.json'.format(relation, output_dir.split('/')[-1])), 'r') as f:
            train_queries = json.load(f)

        for query in train_queries:
            generate_articles_for_neg_query(query)
            del query['answer']
            del query['answer_entity']
            query['is_impossible'] = True
            query['answer'] = []
            query['answer_entity'] = []

        with open(os.path.join(output_dir, '{}_neg.json'.format(relation)), 'w') as f:
            json.dump(train_queries, f)


def generate_articles_for_neg_query(query, wiki_dir='./data/wiki'):
    file_name = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
    answers = query['answer']

    if not os.path.exists(os.path.join(wiki_dir, file_name)):
        return
    with open(os.path.join(wiki_dir, file_name), 'r') as f:
        doc_string = f.read()
    for para in doc_string.split('\n'):
        for sentence in nltk.sent_tokenize(para):
            for answer in answers:
                if answer.lower() in sentence.lower():
                    if sentence.lower().find(answer.lower()) != -1:
                        start_idx = doc_string.find(sentence)
                        doc_string = doc_string[0:start_idx] + doc_string[start_idx + len(sentence):]
                        break
    new_fname = file_name.split('.txt')[0] + '_negative.txt'
    with open(os.path.join(wiki_dir, new_fname), 'w') as f:
        f.write(doc_string)


def iterate_all_neg(stats):
    threshold = stats['THRESHOLD']
    TP, TN, FP, FN = stats['TP'], stats['TN'], stats['FP'], stats['FN']
    FN_N, FN_P = stats['FN_N'], stats['FN_P']

    TOTAL = TP + TN + FP + FN
    NEG = 0

    feat_dir = './neg/features'
    relations = get_all_relations()

    for relation in relations:
        for i in range(2):
            feat_path = os.path.join(feat_dir, '{}-neg-kw_sent-feat-batch-{}.txt'.format(relation, i))
            if os.path.exists(feat_path):
                with open(feat_path, 'r') as f:
                    for idx, line in enumerate(f):
                        NEG += 1
                        result = json.loads(line.strip())

                        if result[0]['null_odds'] < threshold:
                            pass
                        else:
                            FN += 1
                            FN_N += 1
                            continue

                        if result[0]['text'] == '':
                            FN += 1
                            FN_N += 1
                        elif result[0]['text'] != '':
                            FN_P += 1
                            FN += 1

    stats['FN_N'], stats['FN'], stats['FN_P'] = FN_N, FN, FN_P
    assert (TOTAL + NEG) == (TP + TN + FP + FN)
    stats['F1-SCORE'] = (TP + FN_N) / (TOTAL + NEG)
    stats['LEN_POS'] = TOTAL
    stats['LEN_NEG'] = NEG

    return stats


def get_f1_score():
    all_queries, all_samples, stats = iterate_all_data()
    # print(f'LENGTH QUERIES: {len(all_queries)}')
    # print(f'LENGTH SAMPLES: {len(all_samples)}')
    assert len(all_queries) == len(all_samples)
    stats = iterate_all_neg(stats)

    return stats


def run_re():
    feat_path = './neg/features'
    if not os.path.exists(feat_path):
        os.makedirs(feat_path)
    data_path = './neg/data'

    command = 'CUDA_VISIBLE_DEVICES=0 python src/relation_extraction.py ' \
              '--feat_path={} ' \
              '--split=custom_neg ' \
              '--wiki_data=./data/wiki ' \
              '--vocab_file={} ' \
              '--bert_config_file={} ' \
              '--init_checkpoint={} ' \
              '--output_dir=/tmp/tmp_neg ' \
              '--do_predict=True ' \
              '--do_train=False ' \
              '--predict_file=./ ' \
              '--k_sentences=20 ' \
              '--predict_batch_size=32 ' \
              '--num-kw-queries=5 ' \
              '--out_name=kw_sent ' \
              '--version_2_with_negative=True ' \
              '--null_score_diff_threshold=0.0 ' \
              '--data_path={}'
    command = command.format(feat_path, args.vocab_file, args.config_file, args.model_path, data_path)
    # print(command)

    os.system(command)


def print_hints():
    print('0 - Run relation_extraction.py for negative queries.')
    print('1 - Label all queries. ')
    print('2 - Generate negative queries. ')
    print('3 - Get f1 score. ')
    print('[Enter multi-options separated by COMMA e.g. 1,2,3]')
    print('[OPTION-0 need pass paths: --model_path && --vocab_file && --config_file]')
    print('RUN: ', end='')


def get_f1_neg(type='known'):
    val_dir = './out/features/trainonknown/' + type
    relations = sorted(list(set([f[:-len('_neg_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                                 f.endswith('_neg_kw_sent_meta_results.json')])))
    FN, FN_P, FN_N = 0, 0, 0
    for relation in relations:
        for i in range(2):
            if os.path.exists(
                    os.path.join(val_dir, '{}-neg-kw_sent-feat-batch-{}.txt'.format(relation, i))):
                with open(os.path.join(val_dir, '{}-neg-kw_sent-feat-batch-{}.txt'.format(relation, i)),
                          'r') as f:
                    for index, line in enumerate(f):
                        result = json.loads(line.strip())

                        if result[0]['text'] == '':
                            FN += 1
                            FN_N += 1
                        else:
                            FN += 1
                            FN_P += 1

    stats = dict()
    stats['FN'], stats['FN_P'], stats['FN_N'] = FN, FN_P, FN_N
    print(stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='the path to the model checkpoint')
    parser.add_argument('--vocab_file', type=str, default='', help='the path to the vocabulary .txt file')
    parser.add_argument('--config_file', type=str, default='', help='the path to the model_config file')
    args = parser.parse_args()

    folder_list = ['train', 'test']
    for f in folder_list:
        folder = os.path.join('data', 'neg', 'zero_shot', f)
        if not os.path.exists(folder):
            os.makedirs(folder)
        generate_negative_query('./data/splits/zero_shot', folder)

    # print_hints()
    # options = str(input())

    # for option in options.split(','):
    #     option = int(option.strip())
    #     if option == 1:
    #         print('==== [INFO] START LABELING THE QUERIES ====')
    #         label_all_queries()
    #         print('==== [INFO] FINISHED LABELING THE QUERIES ====')
    #     elif option == 2:
    #         print('==== [INFO] START GENERATING THE NEG_QUERIES ====')
    #         generate_negative_query()
    #         print('==== [INFO] FINISHED GENERATING THE NEG_QUERIES ====')
    #     elif option == 3:
    #         print('==== [INFO] START GETTING THE F1-SCORE ====')
    #         statistics = get_f1_score()
    #     elif option == 0:
    #         run_re()
    #     else:
    #         pass




