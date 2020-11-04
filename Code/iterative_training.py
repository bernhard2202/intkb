import argparse
import os
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from utils import bcolors
import nltk
import time
import random
from src.retriever.utils import get_filename_for_article_id
from src.retriever.sentence_retriever import retrieve_sentences_multi_kw_query
from src.utils.answer_normalization import normalize_answer
from distutils.dir_util import copy_tree

ID_IE = 10000000


def construct_question(relation_num):
    with open(os.path.join('./data/labels', '{}_labels.json'.format(relation_num)), 'r') as f:
        query_list = json.load(f)
        query_list.sort(key=lambda s: -len(s))
    query_list = query_list[0:min(len(query_list), 5)]
    query_list = [q + ' ?' for q in query_list]
    return query_list


def random_pick_query(filename):
    with open(filename, 'r') as f:
        queries = json.load(f)

    if len(queries) >= args.top_N:
        random_n_indexes = random.sample(range(0, len(queries)), args.top_N)
    else:
        random_n_indexes = random.sample(range(0, len(queries)), len(queries))

    random_queries = [queries[index] for index in random_n_indexes]

    with open(filename, 'w') as f:
        json.dump(random_queries, f)


def retriever_for_noAns(query, doc):
    results, _ = retrieve_sentences_multi_kw_query([query], [doc], 1)
    return results[0]


def copy_all_test(from_dir, to_dir):
    copy_tree(from_dir, to_dir)


def make_predictions(model_path, out_feat_path, data_path):
    # make predictions => zero_shot train part
    output_dir = args.output_iter
    vocal_file = os.path.join(output_dir, 'model_start_ckt/vocab.txt')
    bert_config_file = os.path.join(output_dir, 'model_start_ckt/bert_config.json')
    threshold = args.threshold
    # 0- feat_path 1- vocal_file 2- bert_config_file 3-init_checkpoint
    # 4- threshold

    #command = 'python src/relation_extraction.py ' \
    command = 'CUDA_VISIBLE_DEVICES=0,2 python src/relation_extraction.py ' \
              '--feat_path={} ' \
              '--split=custom ' \
              '--wiki_data=./data/wiki ' \
              '--vocab_file={} ' \
              '--bert_config_file={} ' \
              '--init_checkpoint={} ' \
              '--output_dir=/tmp/{} ' \
              '--do_predict=True ' \
              '--do_train=False ' \
              '--predict_file=./ ' \
              '--k_sentences=20 ' \
              '--predict_batch_size=32 ' \
              '--num-kw-queries=5 ' \
              '--out_name=kw_sent ' \
              '--version_2_with_negative=True ' \
              '--null_score_diff_threshold={} ' \
              '--data_path={}'
    # print(command.format(out_feat_path, vocal_file, bert_config_file, model_path, str(time.time()).split('.')[0] ,threshold, data_path))
    os.system(command.format(out_feat_path, vocal_file, bert_config_file, model_path, str(time.time()).split('.')[0], threshold, data_path))


def generate_training_data(save_path):
    output_dir = args.output_iter
    len_duplicates = 0

    contexts = []
    old_questions = []

    with open(os.path.join(output_dir, 'DS_train.json'), 'r') as f:
        queries = json.load(f)
    wiki_path = './data/wiki'
    all_heads = set()
    all_tails = set()
    heads = []
    tails = []
    data = []
    data_2_dump = []
    relation_to_count = {}

    filename = ''
    global ID_IE
    score_terms = []
    keywords = set()

    for query in queries:
        relation_num = query['relation']
        with open(os.path.join('data', 'labels', '{}_labels.json'.format(relation_num)), 'r') as f:
            tmp_terms = json.load(f)
        score_terms.extend(tmp_terms)

    score_terms.sort(key=lambda s: -len(s))
    for score_term in score_terms:
        for word in nltk.tokenize.word_tokenize(score_term):
            if word.lower() not in stopwords.words('english') and (word != '(' and word != ')'):
                keywords.add(word.lower())

    for query in tqdm(queries):
        questions = construct_question(query['relation'])
        fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
        answers = query['answer']
        if not os.path.exists(os.path.join(wiki_path, fname)):
            print(f'SKIPPING {wiki_path+"/"+fname}')
            continue
        if len(answers) == 0:
            fname = fname.split('.txt')[0] + '_negative.txt'
        with open(os.path.join(wiki_path, fname), 'r') as f:
            doc_string = f.read()
        cands = []
        if len(answers) == 0:
            top_1_sent = retriever_for_noAns(random.choice(questions)[:-2], os.path.join('data', 'wiki', fname))
            cands.append({
                'title': fname,
                'paragraphs': [
                    {
                        'context': top_1_sent,
                        'qas': [{
                            'id': ID_IE,
                            'question': random.choice(questions),
                            'answers': [

                            ],
                            'is_impossible': True
                        }]
                    }
                ]
            })
        else:
            for para in doc_string.split('\n'):
                for sentence in nltk.sent_tokenize(para):
                    for answer in answers:
                        if answer.lower() in sentence.lower():
                            if sentence.lower().find(answer.lower()) != -1:
                                cands.append({'title': fname,
                                              'paragraphs': [
                                                  {
                                                    "context": sentence,
                                                    "qas": [
                                                        {"id": ID_IE,
                                                         "question": random.choice(questions),
                                                         "answers": [
                                                             {
                                                                "answer_start": sentence.lower().find(answer.lower()),
                                                                "text": answer
                                                             }
                                                         ],
                                                         'is_impossible': False
                                                        }
                                                    ]
                                                  }
                                              ]
                                            })
                        elif normalize_answer(answer) in sentence.lower():
                            if sentence.lower().find(answer.lower()) != -1:
                                cands.append({'title': fname,
                                              'paragraphs': [
                                                  {"context": sentence,
                                                      "qas": [
                                                          {
                                                            "id": ID_IE,
                                                            "question": random.choice(questions),
                                                            "answers": [
                                                               {
                                                                   "answer_start": sentence.lower().find(answer.lower()),
                                                                   "text": answer
                                                               }
                                                           ],
                                                            'is_impossible': False
                                                          }
                                                      ]}]})
            if len(cands) == 0:
                question_rand = random.choice(questions)
                # question_rand[:-2] to ignore the question mark
                top_1_sent = retriever_for_noAns(question_rand[:-2], os.path.join('data', 'wiki', fname))
                cands.append(
                    {
                        'title': fname,
                        'paragraphs': [
                            {
                                'context': top_1_sent,
                                'qas': [
                                    {
                                        'id': ID_IE,
                                        'question': question_rand,
                                        'answers': [

                                        ],
                                        'is_impossible': True
                                    }
                                ]
                            }
                        ]
                    }
                )

        if len(cands) > 1:  # in the case that we have multiple candidates for a single query we take the one with the
            # highest score, if two have the same score we use the one with the shorter answer
            best_score = 0.5  # setting best_score to 0.5 discards any answer candidate that has a score of 0
            best_cand = None
            for cand in cands:
                score = 0
                sent = cand['paragraphs'][0]['context']
                for word in nltk.tokenize.word_tokenize(sent):
                    if word.lower() in keywords:
                        score += 1
                if score > best_score:  # prefer answers with higher score
                    best_cand = cand
                    best_score = score  # prefer shorter answers
                # elift score == best_score and len(cand['paragraphs'][0]['qas'][0]['answers'][0]['text']) \
                #        < len(best_cand['paragraphs'][0]['qas'][0]['answers'][0]['text']) and \
                #        len(cand['paragraphs'][0]['qas'][0]['answers'][0]['text']) > 3:
                #    best_cand = cand
                #    best_score = score
            if best_cand is not None:
                if best_cand['paragraphs'][0]['context'] in contexts and best_cand['paragraphs'][0]['qas'][0]['question'] in old_questions:
                    len_duplicates += 1
                else:
                    data.append(best_cand)
                    heads.append(query['entity_label'])
                    tails.append(query['answer'])
                    contexts.append(best_cand['paragraphs'][0]['context'])
                    old_questions.append(best_cand['paragraphs'][0]['qas'][0]['question'])
                    ID_IE += 1


            else:
                if cands[0]['paragraphs'][0]['context'] in contexts and cands[0]['paragraphs'][0]['qas'][0]['question'] in old_questions:
                    len_duplicates += 1

                else:
                    data.append(cands[0])
                    heads.append(query['entity_label'])
                    tails.append(query['answer'])
                    contexts.append(cands[0]['paragraphs'][0]['context'])
                    old_questions.append(cands[0]['paragraphs'][0]['qas'][0]['question'])
                    ID_IE += 1

        elif len(cands) == 1:

            if cands[0]['paragraphs'][0]['context'] in contexts and cands[0]['paragraphs'][0]['qas'][0]['question'] in old_questions:
                len_duplicates += 1

            else:
                data.append(cands[0])
                heads.append(query['entity_label'])
                tails.append(query['answer'])
                contexts.append(cands[0]['paragraphs'][0]['context'])
                old_questions.append(cands[0]['paragraphs'][0]['qas'][0]['question'])
                ID_IE += 1

    # data(d), heads, tails
    relation_to_count['DS_train'] = len(data)
    data_2_dump.extend(data)
    for h in heads:
        all_heads.add(h)
    for t in tails:
        for t_ in t:
            all_tails.add(t_)

    with open(os.path.join(save_path, 'known_relations_train.json'), 'w') as f:
        json.dump({"data": data_2_dump}, f)

    with open(os.path.join(save_path, 'seen_entities.json'), 'w') as f:
        json.dump({"heads": list(all_heads), "tails": list(all_tails)}, f)
    with open(os.path.join(save_path, 'train_entities_stats.json'), 'w') as f:
        json.dump(relation_to_count, f)

    print('{} samples generated for training.'.format(len(data_2_dump)))

    print(f'{bcolors.WARNING}=== COPYING THE DS_train.json to {save_path} ==={bcolors.ENDC}')
    with open(os.path.join(save_path, 'DS_train_this_iter.json'), 'w') as f:
        json.dump(queries, f)


def retrain_model(model_dir, model_output_dir, train_file):
    # retrain the model with DS_train
    output_dir = args.output_iter
    vocal_file = os.path.join(output_dir, 'model_start_ckt/vocab.txt')
    bert_config_file = os.path.join(output_dir, 'model_start_ckt/bert_config.json')

    init_checkpoint = model_dir
    threshold = args.threshold

    # params 0 - NUM_GPUs, 1- vocal_file 2- bert_config_file
    # 3- init_checkpoint 4- train_file 5-output_dir 6- null_score_diff_threshold
    # command = 'python src/train_bert/run_squad.py ' \
    command = 'CUDA_VISIBLE_DEVICES=0,2 python src/train_bert/run_squad.py ' \
              '--vocab_file={} ' \
              '--bert_config_file={} ' \
              '--init_checkpoint={} ' \
              '--do_train=True ' \
              '--train_file={} ' \
              '--do_predict=False ' \
              '--train_batch_size=6 ' \
              '--learning_rate=3e-5 ' \
              '--num_train_epochs=2.0 ' \
              '--max_seq_length=128 ' \
              '--doc_stride=128 ' \
              '--output_dir={} ' \
              '--do_lower_case=False ' \
              '--version_2_with_negative=True ' \
              '--null_score_diff_threshold={}'
    # print(command.format(NUM_GPUs, vocal_file, bert_config_file, init_checkpoint, train_file
    #                           , model_output_dir, threshold))
    os.system(command.format(vocal_file, bert_config_file, init_checkpoint, train_file
                              , model_output_dir, threshold))


def update_DS_train_dataset(top_n_samples, top_n_queries, DS_path, log_file):
    DS_train = []
    output_dir = args.output_iter
    if os.path.exists(os.path.join(output_dir, 'DS_train.json')):
        with open(os.path.join(output_dir, 'DS_train.json'), 'r') as f:
            queries = json.load(f)
            DS_train.extend(queries)
    one_target = 0
    zero_target_noAns = 0
    doubled_for_pos = 0
    for i, sample in enumerate(top_n_samples):
        if sample[0]['target'] == 1 and not top_n_queries[i]['is_impossible']:
            one_target += 1
            DS_train.extend([top_n_queries[i]])
            # construct the negative query for this query => top_n_queries[i]
            neg_doubled = dict()
            neg_doubled['entity_label'], neg_doubled['question'], neg_doubled['wikipedia_link'] = \
                top_n_queries[i]['entity_label'], top_n_queries[i]['question'], top_n_queries[i]['wikipedia_link']
            neg_doubled['answer'], neg_doubled['answer_entity'] = [], []
            neg_doubled['is_impossible'] = True
            neg_doubled['relation'] = top_n_queries[i]['relation']
            DS_train.extend([neg_doubled])
            doubled_for_pos += 1
        if sample[0]['target'] == 0 and top_n_queries[i]['is_impossible'] and sample[0]['text'] == '':
            zero_target_noAns += 1
            # DS_train.extend([top_n_queries[i]])

    print(f'{bcolors.WARNING}=== Among N:{args.top_N * 50} predictions got from initial_model '
          f'{one_target + zero_target_noAns}(P:{one_target} + N:{zero_target_noAns}) are right ==={bcolors.ENDC}')

    print(f'{bcolors.WARNING}=== Adding {one_target + zero_target_noAns} samples to DS_train ==={bcolors.ENDC}')

    log = f'N: {len(top_n_queries)} predictions (P:{one_target}(doubled query for: {doubled_for_pos}) + N:{zero_target_noAns} = {one_target + zero_target_noAns + doubled_for_pos})' \
          f' are right, {one_target + doubled_for_pos} are added to the DS_train.json'

    with open(log_file, 'w') as f:
        f.write(log)

    if os.path.exists(output_dir):
        with open(DS_path, 'w') as f:
            json.dump(DS_train, f)


def generate_random_data(data_path):

    output_dir = args.output_iter
    # pick random 100 samples in the ./data/splits/zero_shot

    relations = sorted(list(set(
        [f[:-len('_train.json')] for f in os.listdir('./data/splits/zero_shot') if f.endswith('_train.json')])))
    all_queries = []
    all_neg_queries = []

    for relation in relations:
        # load positive part's queries
        with open('./data/splits/zero_shot/{}_train.json'.format(relation), 'r') as f:
            relation_queries = json.load(f)
            len_queries = len(relation_queries)

        # ./data/neg/zero_shot/train
        # load negative part's queries
        with open('./data/neg/zero_shot/train/{}_neg.json'.format(relation), 'r') as f:
            neg_queries = json.load(f)
            len_neg_queries = len(neg_queries)

            len_queries += len_neg_queries
            relation_queries.extend(neg_queries)
            all_neg_queries.extend(neg_queries)
            random.shuffle(relation_queries)

        if args.random == 'False':
            if len_queries >= args.number:
                random_indexes = random.sample(range(0, len_queries), args.number)
            else:
                random_indexes = random.sample(range(0, len_queries), len_queries)
        elif args.random == 'True':
            limit = 10
            if args.option == '1':
                pass
            elif args.option == '2':
                limit = 30
            if len_queries >= limit:
                random_indexes = random.sample(range(0, len_queries), limit)
            else:
                random_indexes = random.sample(range(0, len_queries), len_queries)

        relation_queries = [relation_queries[random_index] for random_index in random_indexes]

        all_queries.extend(relation_queries)
        with open(os.path.join(data_path, '{}_train.json'.format(relation)), 'w') as f:
            json.dump(relation_queries, f)
    copy_all_test(os.path.join(output_dir, 'data'), data_path)
    print(f'[TOTAL QUERY] LENGTH: {len(all_queries)}')


def get_top_predictions(relations, initial_feat_path, initial_data_path):
    num_N = args.top_N
    all_samples = []
    all_queries = []


    for i, relation in enumerate(relations):
        relation_samples = []
        relation_queries = []
        data_path = os.path.join(initial_data_path, '{}_train.json'.format(relation))
        for index in range(2):
            feat_path = os.path.join(initial_feat_path, '{}-train-kw_sent-feat-batch-{}.txt'.format(relation, index))
            if os.path.exists(feat_path):
                with open(feat_path, 'r') as f:
                    for idx, line in enumerate(f):
                        result = json.loads(line.strip())
                        relation_samples.append(result)

        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                queries = json.load(f)
                relation_queries.extend(queries)

        len_samples, len_queries = len(relation_samples), len(relation_queries)
        if len_samples == len_queries:
            pass
        else:
            print(f'[LENGTH] samples:{len_samples} != queries:{len_queries}')

        assert args.random == 'True' or args.random == 'False'
        assert args.option == '1' or args.option == '2'

        if args.option == '1':
            if args.random == 'False':
                # pick the top_N ranked by span_score
                top_n_indexes = sorted(range(len(relation_samples)), key=lambda i:relation_samples[i][0]['span_score'])[-num_N:]
            if args.random == 'True':
                # randomly pick the samples for training
                if len(relation_samples) >= num_N:
                    top_n_indexes = random.sample(range(len(relation_samples)), num_N)
                else:
                    top_n_indexes = random.sample(range(len(relation_samples)), len(relation_samples))
        elif args.option == '2':
            if args.random == 'True':
                list_above_threshold = [sample for sample in relation_samples if sample[0]['null_odds'] < args.option_threshold]
                if len(list_above_threshold) >= 10:
                    top_n_indexes = random.sample(range(len(list_above_threshold)), 10)
                else:
                    top_n_indexes = random.sample(range(len(list_above_threshold)), len(list_above_threshold))
            elif args.random == 'False':
                raise ValueError(" 'random=False && option=2' => Not an option")

        top_n_samples = [relation_samples[top_N_index] for top_N_index in top_n_indexes]
        top_n_queries = [relation_queries[top_N_index] for top_N_index in top_n_indexes]
        all_samples.extend(top_n_samples)
        all_queries.extend(top_n_queries)
        # print(f'[INFO]: SAMPLE_LEN {len(all_samples)}, QUERY_LEN{len(all_queries)}')
    print(f'[INFO]: SAMPLE_LEN {len(all_samples)}, QUERY_LEN{len(all_queries)}')
    return all_samples, all_queries


def label_ds(relations):
    print('[INFO] label_ds()')
    is_impossible = 0

    for relation in relations:
        tmp_num = 0
        print('Processing the relation {} [LABELING]'.format(relation))
        with open(os.path.join('data', 'splits', 'zero_shot', '{}_train.json'.format(relation)), 'r') as f:
            test_queries = json.load(f)
        for query in tqdm(test_queries):
            query = label_one_sample(query, relation)
            if query['is_impossible']:
                tmp_num += 1

        is_impossible += tmp_num
        print(f"[{relation}]Number of false case: {tmp_num}")
        with open (os.path.join('data', 'splits', 'zero_shot', '{}_train.json'.format(relation)), 'w') as f:
            json.dump(test_queries, f)
    print(f"Number of false case(in total): {is_impossible}")


def label_one_sample(original_query, relation):
    answers = original_query['answer']
    original_query['relation'] = relation
    fname = get_filename_for_article_id(original_query['wikipedia_link'].split('wiki/')[-1])
    if not os.path.exists(os.path.join('data', 'wiki', fname)):
        print('DOC NOT FOUND [LABELING SAMPLE]')
    with open(os.path.join('data', 'wiki', fname), 'r') as f:
        doc_string = f.read()
    for answer in answers:
        if doc_string.lower().find(answer.lower()) != -1:
            original_query['is_impossible'] = False
            return original_query

    original_query['is_impossible'] = True
    return original_query


def main():
    output_dir = args.output_iter
    DS_path = os.path.join(output_dir, 'DS_train.json')

    # *-train-kw_sent-feat-batch-0.txt
    relations = sorted(list(set(
        [f[:-len('-train-kw_sent-feat-batch-0.txt')] for f in os.listdir('./out/features/trainonknown/zero_shot') if
         f.endswith('-train-kw_sent-feat-batch-0.txt')])))

    if args.label_ds == 'True':
        label_ds(relations)
    else:
        print(f'{bcolors.WARNING}=== NO LABELING ==={bcolors.ENDC}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'model_start_ckt')):
        os.makedirs(os.path.join(output_dir, 'model_start_ckt'))

    num_iteration = args.num_iter

    print(f'{bcolors.WARNING}=== GET the TOP-{args.top_N*50} predictions from initial_model ==={bcolors.ENDC}')
    # Get the samples, and training queries for the top_N predictions for Z_train

    for i in range(num_iteration):
        print(f'{bcolors.WARNING}[ITERATION({i})]{bcolors.ENDC}')
        iter_dir = os.path.join(os.path.join(output_dir, 'iter_{}'.format(i)))
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)
        if not os.path.exists(iter_dir + '/train'):
            os.makedirs(iter_dir + '/train')
        if not os.path.exists(iter_dir + '/data'):
            os.makedirs(iter_dir + '/data')
        if not os.path.exists(iter_dir + '/model'):
            os.makedirs(iter_dir + '/model')
        if not os.path.exists(iter_dir + '/features'):
            os.makedirs(iter_dir + '/features')

        # Generate data => iter_X/data
        generate_random_data(os.path.join(iter_dir, 'data'))

        # Make predictions
        # generate features/preds => iter_X/features
        if i == 0:
            make_predictions(model_path=os.path.join(output_dir, 'model_start_ckt'),
                             out_feat_path=os.path.join(iter_dir, 'features'),
                             data_path=os.path.join(iter_dir, 'data'))
        else:
            make_predictions(model_path=os.path.join(output_dir, 'iter_{}'.format(i-1), 'model'),
                             out_feat_path=os.path.join(iter_dir, 'features'),
                             data_path=os.path.join(iter_dir, 'data'))

        # Get top-10 for each relation.
        top_n_samples, top_n_queries = get_top_predictions(relations, os.path.join(iter_dir, 'features'),
                                                           os.path.join(iter_dir, 'data'))

        update_DS_train_dataset(top_n_samples, top_n_queries, DS_path,
                                os.path.join(os.path.join(iter_dir, 'log.txt')))

        generate_training_data(os.path.join(iter_dir, 'train'))

        retrain_model(os.path.join(output_dir, 'model_start_ckt'), os.path.join(iter_dir, 'model'),
                      os.path.join(iter_dir, 'train/known_relations_train.json'))


if __name__ == '__main__':
    # create argument parser
    parser = argparse.ArgumentParser(description='Iterative experiment script.')
    parser.add_argument('--num_iter', type=int, default=5, help='number of training iterations')
    parser.add_argument('--initial_model', type=str, default='', help='the path to the model checkpoint')
    parser.add_argument('--vocab_file', type=str, default='', help='the path to the vocabulary .txt file')
    parser.add_argument('--config_file', type=str, default='', help='the path to the model_config file')
    parser.add_argument('--top_N', type=int, default=50, help='the value of N to pick the top_N predictions')
    parser.add_argument('--num_GPUs', type=int, default=1, help='the value of GPUs in your machine')
    parser.add_argument('--threshold', type=float, default=-1.89, help='the initial threshold for '
                                                                     'relation_extraction.py')
    parser.add_argument('--label_ds', type=str, default='False', help='label the DS_train (is_impossible, relation...')

    parser.add_argument('--number', type=int, default=100, help='Number to pick among the samples')

    parser.add_argument('--random', type=str, default='False', help='flag of randomly pick the samples or not.')
    parser.add_argument('--output_iter', type=str, default='./iterations', help='output dir for iterative training')

    parser.add_argument('--option', type=str, default='1', help='1 - random 10 | 2 - up to 10 above threshold')

    parser.add_argument('--option_threshold', type=float, default=0.0, help='the threshold for option 2')

    # get all the arguments.
    args = parser.parse_args()

    main()

