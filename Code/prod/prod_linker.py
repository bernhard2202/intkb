import os
import json
import random
import argparse
from api import upload_generate_train
from tqdm import tqdm

relations_chosen = ['P6']


def get_all_relations():
    pass


def collect_train_pairs(path_data, relations):
    all_train_pairs = dict()
    for relation in relations:
        pairs_relation = []
        with open(os.path.join(path_data, '{}.json'.format(relation)), 'r') as f:
            queries = json.load(f)
        for query in queries:
            if not query['answer']:
                continue
            texts = query['answer']
            uris = query['answer_entity']
            if len(texts) >= 2:
                rand_texts = random.sample(texts, 2)
                for rand_text in rand_texts:
                    pairs_relation.append({'text': rand_text, 'uris': uris})
            else:
                pairs_relation.append({'text': texts[0], 'uris': uris})
        all_train_pairs[relation] = pairs_relation
    return all_train_pairs


def do_chosen_relations():
    train_pairs = collect_train_pairs(args.path_data, relations_chosen)
    for dataset in tqdm(train_pairs.keys()):
        print('PROCESSING RELATION [{}]'.format(dataset))
        upload_generate_train(pairs=train_pairs[dataset], dataset=dataset)


def do_all_relations():
    pass


def main():
    if args.index == 1:
        do_chosen_relations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_data', type=str, default='path/to/data')
    parser.add_argument('--index', type=int, default=2)

    args = parser.parse_args()
    main()
