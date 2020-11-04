""" Create training data from known part of knowledge graph in SQuAD format"""
import logging
import argparse
import json
import os
import nltk
from tqdm import tqdm
import random
from nltk.corpus import stopwords
from retriever.utils import get_filename_for_article_id
from retriever.sentence_retriever import retrieve_sentences_multi_kw_query
from utils.answer_normalization import normalize_answer

# query_map
IDD = 10000000


def construct_question(relation_num):
    with open(os.path.join(args.labels_path, '{}_labels.json'.format(relation_num)), 'r') as f:
        query_list = json.load(f)
        query_list.sort(key=lambda s: -len(s))
    query_list = query_list[0:min(len(query_list), 5)]
    query_list = [q + ' ?' for q in query_list]
    return query_list

def retriever_for_noAns(query, doc):
    results, _ = retrieve_sentences_multi_kw_query([query], [doc], 1)
    return results[0]

def generate_data(relation_num):
    heads = []
    tails = []
    data = []
    filename = ''
    global IDD
    keywords = set()
    # use relation synonyms to score individual answer candidates
    with open(os.path.join(args.labels_path, '{}_labels.json'.format(relation_num)), 'r') as f:
        score_terms = json.load(f)
        score_terms.sort(key=lambda s: -len(s))
    for score_term in score_terms:
        for word in nltk.tokenize.word_tokenize(score_term):
            if word.lower() not in stopwords.words('english') and (word!='(' and word!=')'):
                keywords.add(word.lower())

    if(args.data_type  == 'train'):
        filename  = '{}_train.json'
    elif(args.data_type == 'test'):
        filename = '{}_test.json'
    else:
        raise ValueError(
                "data_type should either be train or test")

    with open(os.path.join(args.input_path, filename.format(relation_num)), 'r') as f:
        train_queries = json.load(f)
    for query in tqdm(train_queries):
        questions = construct_question(relation_num)
        fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
        answers = query['answer']
        if not os.path.exists(os.path.join(args.wiki_path, fname)):
            continue
        with open(os.path.join(args.wiki_path, fname), 'r') as f:
            doc_string = f.read()
        cands = []
        if len(answers) == 0:
            cands.append({
                'title': fname,
                'paragraphs': [
                    {
                        'context': doc_string,
                        'qas': [{
                            'id': IDD,
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
                                                        {"id": IDD,
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
                                                            "id": IDD,
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
                                        'id': IDD,
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
                data.append(best_cand)
                heads.append(query['entity_label'])
                tails.append(query['answer'])
                IDD += 1
        elif len(cands) == 1:
            data.append(cands[0])
            heads.append(query['entity_label'])
            tails.append(query['answer'])
            IDD += 1
    return data, heads, tails


def run():
    data = []
    all_heads = set()
    all_tails = set()
    relations = []
    if args.data_type == 'train':
        relations = [f[:-11] for f in os.listdir(args.input_path) if f.endswith('_train.json') and f.startswith('P')]
    elif args.data_type == 'test':
        relations = [f[:-10] for f in os.listdir(args.input_path) if f.endswith('_test.json') and f.startswith('P')]

    relation_to_count = {}

    for relation in relations:
        d, heads, tails = generate_data(relation)
        relation_to_count[relation] = len(d)
        data.extend(d)
        for h in heads:
            all_heads.add(h)
        for t in tails:
            for t_ in t:
                all_tails.add(t_)

    with open(os.path.join(args.save_path, 'known_relations_train.json'), 'w') as f:
        json.dump({"data": data}, f)

    with open(os.path.join(args.save_path, 'seen_entities.json'), 'w') as f:
        json.dump({"heads": list(all_heads), "tails": list(all_tails)}, f)
    with open(os.path.join(args.save_path, 'train_entities_stats.json'), 'w') as f:
        json.dump(relation_to_count, f)
    if args.data_type == 'train':
        logger.info('{} samples generated for training.'.format(len(data)))
    elif args.data_type == 'test':
        logger.info('{} samples generated for testing.'.format(len(data)))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_path', type=str, help='/path/to/input/labels')
    parser.add_argument('--input_path', type=str, help='/path/to/save/data')
    parser.add_argument('--wiki_path', type=str, help='/path/to/save/data')
    parser.add_argument('--save_path', type=str, help='/path/to/save/data')
    parser.add_argument('--data_type', type=str, help='train/test')

    args = parser.parse_args()

    run()
