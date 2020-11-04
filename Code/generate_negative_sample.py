import argparse
import os
import json
import tensorflow as tf
import random
from tqdm import tqdm
from src.retriever.utils import get_filename_for_article_id
from src.retriever.sentence_retriever import retrieve_sentences_multi_kw_query
import nltk
IDD = 10000000

def construct_question(relation_num):
    with open(os.path.join(args.label_dir, '{}_labels.json'.format(relation_num)), 'r') as f:
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


def generate_validation_set_4negative():
    print('[INFO] generate_validation_set_4negative()')

    relations = [f.split('_test.json')[0] for f in os.listdir(args.relation_dir) if
                 f.endswith('_test.json') and f.startswith('P')]

    for relation in relations:
        print('Processing the relation {} [validation]'.format(relation))
        with open(os.path.join(args.relation_dir, '{}_test.json'.format(relation)), 'r') as f:
            test_queries = json.load(f)
        for query in tqdm(test_queries):
            del query['answer']
            del query['answer_entity']
            query['is_impossible'] = True
        with open(os.path.join(args.relation_dir, '{}_neg.json'.format(relation)), 'w') as f:
            json.dump(test_queries, f)


def label_ds():
    print('[INFO] label_ds()')
    is_impossible = 0
    relations = [f.split('_test.json')[0] for f in os.listdir(args.relation_dir) if
                 f.endswith('_test.json') and f.startswith('P')]
    for relation in relations:
        tmp_num = 0
        print('Processing the relation {} [LABELING]'.format(relation))
        with open(os.path.join(args.relation_dir, '{}_test.json'.format(relation)), 'r') as f:
            test_queries = json.load(f)
        for query in tqdm(test_queries):
            query = label_one_sample(query)
            if query['is_impossible']:
                tmp_num += 1

        is_impossible += tmp_num
        print(f"[{relation}]Number of false case: {tmp_num}")
        with open (os.path.join(args.relation_dir, '{}_test.json'.format(relation)), 'w') as f:
            json.dump(test_queries, f)
    print(f"Number of false case(in total): {is_impossible}")


def label_one_sample(original_query):
    answers = original_query['answer']
    fname = get_filename_for_article_id(original_query['wikipedia_link'].split('wiki/')[-1])
    if not os.path.exists(os.path.join(args.wiki_dir, fname)):
        print('DOC NOT FOUND [LABELING SAMPLE]')
    with open(os.path.join(args.wiki_dir, fname), 'r') as f:
        doc_string = f.read()
    for answer in answers:
        if doc_string.lower().find(answer.lower()) != -1: # find the ans
            original_query['is_impossible'] = False
            return original_query

    original_query['is_impossible'] = True
    return original_query


# relation_dir => set to ./data/splits/zero_shot !Important
def zero_shot_generate_articles_4negative():
    relation_dir = './data/splits/zero_shot'
    print('[INFO] Running zero_shot_generate_articles_4negative()')
    print('[INFO] Preparing the negative articles')
    """
        PART I - Prepare the negative articles
    """
    # get the PID of properties
    relations = [f.split('_test.json')[0] for f in os.listdir(relation_dir) if
                 f.endswith('_test.json') and f.startswith('P')]

    for relation in relations:
        print('Processing the relation {}'.format(relation))
        with open(os.path.join(relation_dir, '{}_test.json'.format(relation)), 'r') as f:
            train_queries = json.load(f)
        for query in tqdm(train_queries):
            fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
            answers = query['answer']
            if not os.path.exists(os.path.join(args.wiki_dir, fname)):
                continue
            with open(os.path.join(args.wiki_dir, fname), 'r') as f:
                doc_string = f.read()
            for para in doc_string.split('\n'):
                for sentence in nltk.sent_tokenize(para):
                    for answer in answers:
                        if answer.lower() in sentence.lower():
                            if sentence.lower().find(answer.lower()) != -1:
                                start_idx = doc_string.find(sentence)
                                doc_string = doc_string[0:start_idx] + doc_string[start_idx + len(sentence):]
                                break
            new_fname = fname.split('.txt')[0] + '_negative.txt'
            with open(os.path.join(args.wiki_dir, new_fname), 'w') as f:
                f.write(doc_string)

    print('[INFO] Generating _neg.json files')
    """
            PART II - generate _neg.json files
    """
    relations = [f.split('_test.json')[0] for f in os.listdir(relation_dir) if
                 f.endswith('_test.json') and f.startswith('P')]

    for relation in relations:
        print('Processing the relation {} [validation]'.format(relation))
        with open(os.path.join(relation_dir, '{}_test.json'.format(relation)), 'r') as f:
            test_queries = json.load(f)
        for query in tqdm(test_queries):
            del query['answer']
            del query['answer_entity']
            query['is_impossible'] = True
        with open(os.path.join(relation_dir, '{}_neg.json'.format(relation)), 'w') as f:
            json.dump(test_queries, f)

def main():
    print('[INFO] main()')
    with tf.io.gfile.GFile(args.old_krt, "r") as reader:
        input_data = json.load(reader)["data"]
    new_IDD = len(input_data) + IDD
    # get all the relations in the relation_dir folder
    relations = [f.split('_train.json')[0] for f in os.listdir(args.relation_dir) if f.endswith('_train.json') and f.startswith('P')]
    for relation in relations:
        print('Processing the relation {}'.format(relation))
        with open(os.path.join(args.relation_dir, '{}_train.json'.format(relation)), 'r') as f:
            train_queries = json.load(f)
        for query in tqdm(train_queries):
            questions = construct_question(relation)
            rand_question = random.choice(questions)
            fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
            answers = query['answer']
            if not os.path.exists(os.path.join(args.wiki_dir, fname)):
                continue
            with open(os.path.join(args.wiki_dir, fname), 'r') as f:
                doc_string = f.read()
            for para in doc_string.split('\n'):
                for sentence in nltk.sent_tokenize(para):
                    for answer in answers:
                        if answer.lower() in sentence.lower():
                            if sentence.lower().find(answer.lower()) != -1:
                                start_idx = doc_string.find(sentence)
                                doc_string = doc_string[0:start_idx] + doc_string[start_idx + len(sentence):]
                                break
            new_fname = fname.split('.txt')[0]+'_negative.txt'
            with open(os.path.join(args.wiki_dir, new_fname), 'w') as f:
                f.write(doc_string)

            if len(doc_string) != 0:
                top_1_sent = retriever_for_noAns(rand_question[:-2], os.path.join('data', 'wiki', new_fname))
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
    with open(os.path.join(args.save_dir, 'known_relations_train.json'), 'w') as f:
        json.dump({"data": input_data}, f)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, default='./data/labels', help='path to the labels folder')
    parser.add_argument('--relation_dir', type=str, default='./data/splits/known', help='path to the relations folder')
    parser.add_argument('--save_dir', type=str, default='./data/train/known',
                        help='path to save the negative samples generated')
    parser.add_argument('--wiki_dir', type=str, default='./data/wiki', help='path to the stored articles')
    parser.add_argument('--old_krt', type=str, default='./data/train/known/known_relations_train.json',
                        help='original known_relations_train.json path')

    args = parser.parse_args()


    # label_ds()
    main()

    generate_validation_set_4negative()

    zero_shot_generate_articles_4negative()


