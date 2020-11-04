#!/usr/bin/env python3
""" Data loading from files, processing, vectorize and batchify samples """

import copy
import json
import logging
import os
from retriever.utils import get_filename_for_article_id
import random

import numpy as np
import torch

from .features import get_pos_features, get_ner_features, get_question_type_features, get_jaccard_sim

random.seed(42)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Data loading normalization and subsampling
# ---------------------------------------------------------

def write_results(val_relations, predictions, output_dir, featues_path, data_path):
    j = 0
    res = 0
    print(len(predictions))
    for i, relation in enumerate(val_relations):
        final_pred = []
        final_query = []
        files = [os.path.join(featues_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation)), os.path.join(featues_path, '{}-test-kw_sent-feat-batch-1.txt'.format(relation))]
        for file in files:
            if not os.path.exists(file):
                logger.warning("skipping non existing file {}".format(file))
                continue
            with open(file, 'r') as f:
                for line in f:
                    samples = json.loads(line)
                    final_pred.append(samples[predictions[j]])
                    res += final_pred[-1]['target']
                    j += 1
        with open(os.path.join(data_path, '{}_test.json'.format(relation)), 'r') as f:
            original_query = json.load(f)
            for query in original_query:
                fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
                if not os.path.exists(os.path.join('..', 'data', 'wiki', fname)):
                    continue
                final_query.append(query)
        with open(os.path.join(output_dir, '{}-predictions.json'.format(relation)), 'w') as f:
            for i in range(len(final_pred)):
                f.write(json.dumps({'query': final_query[i], 'prediction': final_pred[i]})+'\n')


def load_full_zs(val_relations, featues_path, data_path, heads, tails):
    """
    Load all the given data samples from disk, extend the dictionary and fit the normalizers to it
    """
    complete_data = []
    is_rare = []
    seen_head = []
    seen_tail = []

    for i, relation in enumerate(val_relations):
        files = [os.path.join(featues_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation)), os.path.join(featues_path, '{}-test-kw_sent-feat-batch-1.txt'.format(relation))]
        for file in files:
            if not os.path.exists(file):
                logger.warning("skipping non existing file {}".format(file))
                continue
            with open(file, 'r') as f:
                i = 0
                for line in f:
                    samples = json.loads(line)
                    complete_data.append(samples)
                    i += 1
                for _ in range(i):
                    is_rare.append(True if i < 10 else False)
        with open(os.path.join(data_path, '{}_test.json'.format(relation)), 'r') as f:
            original_query = json.load(f)
            for query in original_query:
                fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
                if not os.path.exists(os.path.join('..', 'data', 'wiki', fname)):
                    continue
                seen_head.append(True if query['entity_label'] in heads else False)
                tail = False
                for t in query['answer']:
                    if t in tails:
                        tail = True
                seen_tail.append(tail)

    print(len(complete_data))
    print(len(is_rare))
    print(len(seen_tail))

    #assert len(complete_data) == len(is_rare) == len(seen_tail) == len(seen_head)
    return complete_data, is_rare, seen_head, seen_tail


def load_full_known(val_relations, featues_path, data_path, frequencies):
    """
    Load all the given data samples from disk, extend the dictionary and fit the normalizers to it
    """
    complete_data = []
    frequency_of_relation = []

    for i, relation in enumerate(val_relations):
        file = os.path.join(featues_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation))
        if not os.path.exists(file):
            logger.warning("skipping non existing file {}".format(file))
            continue
        with open(file, 'r') as f:
            i = 0
            for line in f:
                samples = json.loads(line)
                complete_data.append(samples)
                i += 1
            for _ in range(i):
                frequency_of_relation.append(frequencies[relation])

    print(len(complete_data))

    assert len(complete_data) == len(frequency_of_relation)

    return complete_data, frequency_of_relation


def load_subsample(data, feature_descriptors, args):
    """
    Load all the given data samples from disk, extend the dictionary and fit the normalizers to it
    """
    subsampled_data = []
    subsampled_data_valid = []
    normalizers = copy.deepcopy(feature_descriptors)
    values = [[] for _ in normalizers]
    for i, file_names in enumerate(data):
        subsampled_data.append([])
        subsampled_data_valid.append([])
        for file in file_names:
            if not os.path.exists(file):
                logger.warning("skipping non existing file {}".format(file))
                continue
            logger.info('loading and processing {}'.format(file))
            with open(file, 'r') as f:
                for qa in f:
                    answers = json.loads(qa)
                    if len(answers) < 1:
                        continue

                    for j, ft in enumerate(normalizers):
                        for answer_candiate in answers:
                            if answer_candiate['text'] == '':
                                continue
                            values[j].append(answer_candiate[ft['feature_name']])

                    subsampled = sample_ranking_pairs(answers, args.max_depth, args.max_per_q)
                    subsampled = copy.deepcopy(subsampled)
                    if len(subsampled) > 0:
                        if random.random() < args.valid_split:
                            subsampled_data[i].extend(subsampled)
                        else:
                            subsampled_data_valid[i].extend(subsampled)
    final_train = []
    final_valid = []
    min_len = min(len(x) for x in subsampled_data_valid)
    for i in range(len(data)):
        final_train.extend(subsampled_data[i])
        if args.stratify_valid:
            final_train.extend(subsampled_data_valid[i][min_len:])
            final_valid.extend(subsampled_data_valid[i][0:min_len])
        else:
            final_valid.extend(subsampled_data_valid[i])

    for j, ft in enumerate(normalizers):
        stats = {'mean': np.mean(values[j]), 'max': np.max(values[j]), 'min': np.min(values[j]),
                 'std': np.std(values[j])}
        for k, v in stats.items():
            ft[k] = v
    return final_train, final_valid, normalizers


def fit_normalizers(data, feature_descriptors):
    """
    Fits a list of normalizers to the given training data
    a feature_descriptors is a dictionary containing the following fields:
        feature_name: the name of the feature as stored in the data
        preprocess: arbitrary function used for preprocessing or None
        normscheme: is either 'normal' to normalize data with 0 mean and 1 std or 'minmax' to normalize within [0,1]
    """
    feature_descriptors = copy.deepcopy(feature_descriptors)
    for normalizer in feature_descriptors:
        vals = []
        for row in data:
            for d in row:
                if normalizer['preprocess'] is not None:
                    vals.append(normalizer['preprocess'](d[normalizer['feature_name']]))
                else:
                    vals.append(d[normalizer['feature_name']])
        for k, v in {'mean': np.mean(vals), 'max': np.max(vals), 'min': np.min(vals), 'std': np.std(vals)}.items():
            normalizer[k] = v
    return feature_descriptors


def apply_normalizer(sample, normalizer):
    """
    apply a normalizer that has been fitted before
    """
    if normalizer['feature_name'] not in sample:
        return 0
    value = sample[normalizer["feature_name"]]
    # preprocessing
    if 'preprocess' not in normalizer or normalizer['preprocess'] is not None:
        value = normalizer['preprocess'](value+0.0001)

    # normalize
    if 'scheme' not in normalizer or normalizer['scheme'] is None:
        return value
    if normalizer['normscheme'] == 'normal':
        return (value - normalizer['mean']) / normalizer['std']
    if normalizer['normscheme'] == 'minmax':
        return (value - normalizer['min']) / (normalizer['max'] - normalizer['min'])
    raise RuntimeError('Unknown normalization scheme in {}'.format(normalizer))


def sample_ranking_pairs(data, max_depth, max_per_q):
    """
    Sample training pairs for ranking, the result is a list of samples where
    the first sample should be ranked before the second.
    :param data: one list containing candidate answers for a single question with annotated targets
    :param max_depth: how deep in terms of candidate answers per question do we go to generate training samples
    :param max_per_q: defines an upper bound on how many pairs we generate per training question
    """
    training_pairs = []
    added = 0
    i = 0
    while i < len(data) - 1 and i < max_depth:
        if not data[i]['target'] == data[i + 1]['target']:
            added += 1
            if data[i]['target'] == 1:
                training_pairs.append((data[i], data[i + 1]))
            else:
                training_pairs.append((data[i + 1], data[i]))
        if added >= max_per_q:
            break
        i += 1
    return training_pairs


def build_validation_dataset(dataset, normalizers):
    X = []
    y = []

    for i, data in enumerate(dataset):
        X.append([])
        y.append([])
        for d in data:
            X[i].append(vectorize(d, normalizers))
            y[i].append(d['target'])
    return X, y


# --------------------------------------------------------------------------------------
# Vectorize and batchify samples to feed them as torch inputs
# --------------------------------------------------------------------------------------


def vectorize(sample, normalizers):
    """
    create torch tensors for a input sample
    """

    # NER POS and Q-TYPE features

    # ner_features = get_ner_features(sample)
    # pos_features = get_pos_features(sample)
    #qtype_features = get_question_type_features(sample)

    # other features

    normalized_features = np.zeros(len(normalizers)) # + 1)
    for i, normalizer in enumerate(normalizers):
        normalized_features[i] = apply_normalizer(sample, normalizer)
    # normalized_features[len(normalizers)] = get_jaccard_sim(sample['context']['tokens'], sample['question_tokens'])

    # aggregate features

    # features = np.concatenate((#ner_features, pos_features, qtype_features, normalized_features), axis=-1)
    features_tensor = torch.from_numpy(normalized_features).float()

    return features_tensor


def batchify_pair(batch):
    xa = torch.stack([ex[0] for ex in batch])
    xb = torch.stack([ex[1] for ex in batch])
    y = torch.stack([ex[2] for ex in batch])
    return xa, xb, y
