import json
import os
import argparse
from utils import bcolors
from utils import get_relations_by_entityID, extracts_prediction_samples, all_label_property, print_stats_validation_get, \
                  extracts_training_samples
from tqdm import tqdm
import random
from iterative_training import construct_question, retriever_for_noAns
from src.retriever.utils import get_filename_for_article_id
from src.utils.answer_normalization import normalize_answer
import nltk
from nltk.corpus import stopwords
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from api import get_most_related_categories

ID_IE = 10000000

endpoint_url = "https://query.wikidata.org/sparql"

basic_folders = ['data', 'features', 'model', 'data/train', 'data/relations']

# P31 - 'instance of'
relation_excluded = ['P31']

relation_included = ['P178']

# Category List to extract
#  >>>>> Add more entity ids here [START]<<<<<
category_2_extract = ['model', 'Q783794', 'Q571', 'Q15416', 'Q5398426', 'Q7889', 'Q349', 'Q41298', 'Q39201', 'Q211236', 'Q901']
#  >>>>> Add more entity ids here [END]<<<<<


def create_folders(categories_):
    categories = []
    if args.option == 1:
        categories = category_2_extract
    elif args.option == 2:
        categories = categories_
    for categoryId in categories:
        if not os.path.exists(os.path.join(args.output_dir, categoryId)):
            os.makedirs(os.path.join(args.output_dir, categoryId))

        for folder in basic_folders:
            if categoryId == 'model':
                continue
            if not os.path.exists(os.path.join(args.output_dir, categoryId, folder)):
                os.makedirs(os.path.join(args.output_dir, categoryId, folder))


def construct_query_for_entity(entity_id):
    if entity_id[0] == 'Q':
        query = """select ?p ?c where {
                {
                    select ?p (count(?s) as ?c ) where 
                    {
                      ?s  wdt:P31 <http://www.wikidata.org/entity/""" + entity_id +"""> .
                      ?s ?p ?o .
                    } group by ?p limit 100
                }
                ?s2 wikibase:directClaim ?p .
                ?s2 wikibase:propertyType wikibase:WikibaseItem
                } order by desc(?c)"""
    else:
        raise ValueError('Right EntityId, e.g. Q11032')

    return query


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()["results"]["bindings"]


def get_top_k_relations_4_entity(entityId, k):
    query_for_entity = construct_query_for_entity(entityId)
    results = get_results(endpoint_url=endpoint_url, query=query_for_entity)
    relations = [result['p']['value'].split('/')[-1] for result in results if
                 result['p']['value'].split('/')[-1] not in relation_excluded]
    counts = [result['c']['value'].split('/')[-1] for result in results if
                 result['p']['value'].split('/')[-1] not in relation_excluded]
    relation_count_dict = dict()
    if k > len(relations):
        for i in range(len(relations)):
            relation_count_dict[relations[i]] = int(counts[i])
    else:
        for i in range(k):
            relation_count_dict[relations[i]] = int(counts[i])

    return relation_count_dict, list(relation_count_dict.keys())


def get_query_data(relations, entityId, amount):
    write_to_log(log_file, '[START] get_query_data()')
    for relation in relations:
        property_url = 'http://www.wikidata.org/prop/direct/' + relation
        extracts_prediction_samples(entityId=entityId, property=property_url, amount=amount, output_dir=os.path.join(args.output_dir, entityId))
    write_to_log(log_file, '[END] get_query_data()')


def get_article_data(entity):
    write_to_log(log_file, '[START] get_article_data()')
    command_1 = 'python src/download_wikipedia.py {} ./data/wiki'.format(os.path.join(args.output_dir, entity, 'data'))
    command_2 = 'python src/download_wikipedia.py {} ./data/wiki'.format(os.path.join(args.output_dir, entity, 'data', 'relations'))
    os.system(command_1)
    os.system(command_2)
    write_to_log(log_file, '[END] get_article_data()')


def get_labels_data(relations):
    write_to_log(log_file, '[START] get_labels_data()')
    for relation in relations:
        if os.path.exists('data/labels/{}_labels.json'.format(relation)):
            continue
        property = 'http://www.wikidata.org/prop/direct/' + relation
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        labels =[];
        query = """
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            select ?label
                where {{
                {{
                    ?property wikibase:directClaim <{property}> .
                    ?property rdfs:label ?label .
                    filter(lang(?label)='en').
                }} UNION {{
                    ?property wikibase:directClaim <{property}> .
                    ?property skos:altLabel ?label .
                    filter(lang(?label)='en').
                }}
                }} limit 3000
                    """.format(property=property)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            labels.append(result["label"]["value"])
        with open('data/labels/' + property.replace("http://www.wikidata.org/prop/direct/", "") + '_labels.json', 'w') as outfile:
            json.dump(labels, outfile)
        outfile.close()
    write_to_log(log_file, '[END] get_labels_data()')


def get_training_data(relations, entityId):
    write_to_log(log_file, '[START] get_training_data()')

    for relation in relations:
        property_url = 'http://www.wikidata.org/prop/direct/' + relation
        extracts_training_samples(property=property_url, output_dir=os.path.join(args.output_dir, entityId), entityId=entityId)

    write_to_log(log_file, '[END] get_training_data()')


def generate_training_data(entityId):
    write_to_log(log_file, '[START] generate_training_data()')
    len_duplicates = 0

    contexts = []
    old_questions = []

    relations_list = [f for f in os.listdir(os.path.join(args.output_dir, entityId, 'data/relations')) if f.endswith('.json')]
    queries = []
    for relation in relations_list:
        with open(os.path.join(args.output_dir, entityId, 'data/relations/{}'.format(relation)), 'r') as f:
            query_relation = json.load(f)
            for q in query_relation:
                q['relation'] = relation.split('.json')[0]
            queries.extend(query_relation)

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
            continue
        with open(os.path.join(wiki_path, fname), 'r') as f:
            doc_string = f.read()
        cands = []
        if len(answers) == 0:
            cands.append({
                'title': fname,
                'paragraphs': [
                    {
                        'context': doc_string,
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

    save_path = os.path.join(args.output_dir, entityId, 'data/train')

    with open(os.path.join(save_path, 'known_relations_train.json'), 'w') as f:
        json.dump({"data": data_2_dump}, f)

    with open(os.path.join(save_path, 'seen_entities.json'), 'w') as f:
        json.dump({"heads": list(all_heads), "tails": list(all_tails)}, f)
    with open(os.path.join(save_path, 'train_entities_stats.json'), 'w') as f:
        json.dump(relation_to_count, f)

    print('{} samples generated for training.'.format(len(data_2_dump)))
    write_to_log(log_file, '[END] generate_training_data()')


def generate_negative_data(entityId):
    write_to_log(log_file, '[START] generate_negative_data()')
    global ID_IE
    pos_part_path = os.path.join(args.output_dir, entityId, 'data/train/known_relations_train.json')
    with open(pos_part_path, 'r') as f:
        input_data = json.load(f)['data']
    new_IDD = len(input_data) + ID_IE
    relations_list = [f for f in os.listdir(os.path.join(args.output_dir, entityId, 'data/relations')) if f.endswith('.json')]
    queries = []
    for relation in relations_list:
        with open(os.path.join(args.output_dir, entityId, 'data/relations/{}'.format(relation)), 'r') as f:
            query_relation = json.load(f)
            for q in query_relation:
                q['relation'] = relation.split('.json')[0]
            queries.extend(query_relation)

    for query in queries:
        fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
        if not os.path.exists(os.path.join('data', 'wiki', fname)):
            print(f'DOC {fname} not exist !')
            continue
        with open(os.path.join('data', 'wiki', fname), 'r') as f:
            doc_string = f.read()

        answers = query['answer']
        if len(answers) == 0:
            print(f'Length answers == 0 !')
            continue
        flag = False
        for ans in answers:
            if (ans.lower() in doc_string.lower()) and (doc_string.lower().find(ans.lower()) != -1):
                flag = True
        if not flag:
            print('Answer not exist in the article')
            continue

        questions = construct_question(relation.split('.json')[0])
        rand_question = random.choice(questions)

        for para in doc_string.split('\n'):
            for sentence in nltk.sent_tokenize(para):
                for answer in answers:
                    if answer.lower() in sentence.lower():
                        if sentence.lower().find(answer.lower()) != -1:
                            start_idx = doc_string.find(sentence)
                            doc_string = doc_string[0:start_idx] + doc_string[start_idx + len(sentence):]
                            break

        new_fname = fname.split('.txt')[0] + '_negative.txt'
        path_to_neg_doc = os.path.join('data', 'wiki', new_fname)
        with open(path_to_neg_doc, 'w') as f:
            f.write(doc_string)

        if len(doc_string) != 0:
            top_1_sent = retriever_for_noAns(rand_question[:-2], path_to_neg_doc)
            input_data.append(
                {
                    'title': new_fname,
                    'paragraphs': [
                        {
                            'context': top_1_sent,
                            'qas': [
                                {
                                    'id': new_IDD,
                                    'question': rand_question,
                                    'answers': [

                                    ],
                                    'is_impossible': True
                                }
                            ]
                        }
                    ]
                }
            )
            new_IDD += 1

    with open(pos_part_path, 'w') as f:
        json.dump({"data": input_data}, f)
    write_to_log(log_file, '[END] generate_negative_data()')


def retrain_model(entityId):
    write_to_log(log_file, '[START] retrain_model()')
    vocal_file = os.path.join(args.output_dir, 'model/vocab.txt')
    bert_config_file = os.path.join(args.output_dir, 'model/bert_config.json')
    init_checkpoint = os.path.join(args.output_dir, 'model')
    train_file = os.path.join(args.output_dir, entityId, 'data/train/known_relations_train.json')
    model_output = os.path.join(args.output_dir, entityId, 'model')

    # params 0 - NUM_GPUs, 1- vocal_file 2- bert_config_file
    # 3- init_checkpoint 4- train_file 5-output_dir 6- null_score_diff_threshold
    # command = 'python src/train_bert/run_squad.py ' \
    command = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train_bert/run_squad.py ' \
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
    os.system(command.format(vocal_file, bert_config_file, init_checkpoint, train_file, model_output,
                             str(args.threshold)))
    write_to_log(log_file, '[END] retrain_model()')


def make_predictions(entity):
    write_to_log(log_file, '[START] make_predictions()')
    feature_path = os.path.join(args.output_dir, entity, 'features')
    model_path = os.path.join(args.output_dir, entity, 'model')
    vocab_path = os.path.join(args.output_dir, 'model', 'vocab.txt')
    config_path = os.path.join(args.output_dir, 'model', 'bert_config.json')
    data_path = os.path.join(args.output_dir, entity, 'data')
    tmp_path = str(time.time()).split('.')[0]
    command = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python src/relation_extraction.py ' \
              '--feat_path={} ' \
              '--split=prod ' \
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
    # print(command.format(feature_path, vocab_path, config_path, model_path, tmp_path, str(args.threshold), data_path))
    os.system(command.format(feature_path, vocab_path, config_path, model_path, tmp_path, str(args.threshold), data_path))
    write_to_log(log_file, '[END] make_predictions()')


def write_to_log(log_file, text):
    with open(log_file, 'a+') as f:
        f.write(text)
        f.write('\n')


def main():

    categories = []
    rand_relation = ''
    if args.option == 1:
        categories = category_2_extract
    elif args.option == 2:
        rand_relation = random.choice(relation_included)
        cate_dict = get_most_related_categories(rand_relation, 5)
        categories = list(cate_dict.keys())
        categories.append('model')
        print(f'Got {len(categories)} categories for {rand_relation}, they are : {categories}')
    # Create basic folders
    create_folders(categories)
    for category in tqdm(categories):

        if category == 'model':
            continue
        write_to_log(log_file, '[START] Category: {}'.format(category))
        # Get top-k relations about this category
        relations_4_category = []
        if args.option == 1:
            _, relations_4_category = get_top_k_relations_4_entity(category, args.num_k)
        elif args.option == 2:
            relations_4_category.append(rand_relation)

        # Get the queries we want to predict
        # It will be saved in args.output_dir/entityId/data/PXXX_test.json
        get_query_data(relations_4_category, category, args.amount)

        # Get labels of relations about this category
        # It will be saved in project_path/data/label
        get_labels_data(relations_4_category)

        # Get wikipedia articles.
        # It will be saved in the project_path/data/wiki
        get_article_data(category)

        # Get all the relations query for this category
        # It will be saved in the args.output_dir/entityId/data/relations/PXXX.json
        get_training_data(relations_4_category, category)

        # Generate the train dataset base on the queries
        generate_training_data(category)

        # Generate the negative samples for training.
        generate_negative_data(category)

        # Retrain the model with train dataset.
        retrain_model(category)

        # Make predictions with the model of this category.
        make_predictions(category)

        write_to_log(log_file, '[END] Category: {}'.format(category))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative experiment script.')

    parser.add_argument('--num_k', type=int, default=7, help='top-k relations')
    parser.add_argument('--amount', type=int, default=1000, help='amount of predictions for each relation')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for prediction')
    parser.add_argument('--output_dir', type=str, default='prod_new_facts', help='output folder path')
    parser.add_argument('--option', type=int, default='1', help='')

    args = parser.parse_args()
    log_file = os.path.join(args.output_dir, '{}-log.txt'.format(str(time.time()).split('.')[0]))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, 'model/vocab.txt')):
        raise FileNotFoundError(str(os.path.join(args.output_dir, 'model/vocab.txt')) + ' not exist!! ')

    if not os.path.exists(os.path.join(args.output_dir, 'model/bert_config.json')):
        raise FileNotFoundError(str(os.path.join(args.output_dir, 'model/bert_config.json')) + ' not exist!! ')

    main()
