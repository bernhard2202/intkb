import logging
import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import random

from retriever.utils import get_filename_for_article_id
from utils.answer_normalization import normalize_answer

random.seed(21)
np.random.seed(21)


def split_data(relation):
    """
    load a relation and filter out samples that are not answerable in general (answer string not
    found in the supporting document) split remaining samples into train/test split according
    to args.ratio. very small datasets with < 10 samples are skipped and those with < 20 samples are
    split evenly across train/test (ratio is set to 0.5).
    """
    with open(os.path.join(args.input_path, '{}.json'.format(relation)), 'r') as f:
        json_data = json.load(f)

    answerable_samples = []
    for sample in tqdm(json_data):
        fname = get_filename_for_article_id(sample['wikipedia_link'].split('wiki/')[-1])
        answers = sample['answer']
        for answer in answers:
            if os.path.exists(os.path.join(args.wiki_path, fname)):
                with open(os.path.join(args.wiki_path, fname), 'r') as f:
                    doc_string = f.read()
                if normalize_answer(answer) in normalize_answer(doc_string):
                    answerable_samples.append(sample)
                    break

    ratio = args.ratio

    if len(answerable_samples) < 10:
        return [], []

    if len(answerable_samples) < 20:
        ratio = 0.5

    random.shuffle(answerable_samples)
    splitpoint = int(len(answerable_samples)*ratio)

    return answerable_samples[:splitpoint], answerable_samples[splitpoint:]


def run_for_all():
    """ load datasets randomly and assign them to the known part and the zero-shot part """

    known_length = [[], []]  # just for statistics
    zeroshot_length = [[], []]  # just for statistics

    relations = [f[:-5] for f in os.listdir(args.input_path) if f.endswith('.json') and f.startswith('P')]
    random.shuffle(relations)

    known_ds = 0
    zero_shot = 0

    for relation in relations:
        logger.info('Parsing relation {}'.format(relation))
        train_split, test_split = split_data(relation)

        # in case train_split contains more than args.cutoff_train samples trim it, same with test
        train_split = train_split[:min(len(train_split), args.cutoff_train)]
        test_split = test_split[:min(len(test_split), args.cutoff_test)]

        # fore the rare case that one list is empty skip this relation
        if len(train_split) == 0 or len(test_split) == 0:
            continue

        # store to the known knowledge graph or the zero-shot part
        folder = 'zero_shot'
        if known_ds < args.num_known:
            known_ds += 1
            folder = 'known'
            known_length[0].append(len(train_split))
            known_length[1].append(len(test_split))
        elif zero_shot < args.num_zeroshot:
            assert known_ds == args.num_known, "first fill the known parts"
            zero_shot += 1
            zeroshot_length[0].append(len(train_split))
            zeroshot_length[1].append(len(test_split))
        else:
            break  # we are done

        with open(os.path.join(args.save_path, folder, '{}_train.json'.format(relation)), 'w') as f:
            json.dump(train_split, f)

        with open(os.path.join(args.save_path, folder, '{}_test.json'.format(relation)), 'w') as f:
            json.dump(test_split, f)
        print(len(train_split), len(test_split))
    assert zero_shot == args.num_zeroshot and known_ds == args.num_known, "less relations extracted than expected"

    def _print_stats(name, lengths):
        lengths = np.array(lengths)
        print('{}: total: {} min: {}, mean {}, std {}, max {}'.format(name, np.sum(lengths), min(lengths),
                                                                      np.mean(lengths), np.std(lengths),
                                                                      max(lengths)))
    _print_stats('known train', known_length[0])
    _print_stats('known test', known_length[1])
    if args.num_zeroshot != 0:
        _print_stats('zeroshot_train', zeroshot_length[0])
        _print_stats('zeroshot_test', zeroshot_length[1])


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='/path/to/input/data')
    parser.add_argument('--wiki_path', type=str, help='/path/to/wikipedia/data')
    parser.add_argument('--save_path', type=str, help='/path/to/save/split/data')
    parser.add_argument('--ratio', type=float, help='ratio to split training and test')
    parser.add_argument('--num_known', type=int, default=250, help='number of known relations for the initial kg')
    parser.add_argument('--num_zeroshot', type=int, default=50, help='number of unknown relations for zero-shot')
    parser.add_argument('--cutoff_train', type=int, default=1000, help='maxium number of samples in training data')
    parser.add_argument('--cutoff_test', type=int, default=500, help='maximum number of samples for the test data')

    args = parser.parse_args()

    run_for_all()
