import argparse
import logging
import os
import sys
import time
import json
import uuid
import torch.optim as optim
import numpy as np
import torch
from ranker.evaluate import EvaluatorZs, EvaluatorKnown
from ranker import data_utils, data_loader
from ranker import ranker_net
from ranker.data_utils import batchify_pair
from torch.utils import data
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------
# description of features to use during training
# ---------------------------------------------------------------------------------------------


def prepro(value):
    return np.log(value+1000)


feature_names_regular = ['first_occ', 'num_occ', 'span_len', 'question_len', 'context_len']
feature_names_aggregated = ['doc_score', 'doc_pos', 'paragraph_score']
feature_names_aggregated_nolog = ['span_score', 'start_logit', 'end_logit', 'null_odds']
feature_descriptors = [{'feature_name': name, 'normscheme': 'minmax', 'preprocess': np.log} for name in feature_names_regular]
feature_descriptors.extend([{'feature_name': '{}{}'.format(prefix, name), 'normscheme': 'minmax', 'preprocess': np.log} for name in  feature_names_aggregated for prefix in ['',  'min_', 'max_', 'avg_']])
feature_descriptors.extend([{'feature_name': '{}{}'.format(prefix, name), 'normscheme': 'minmax', 'preprocess': prepro} for name in  feature_names_aggregated_nolog for prefix in ['', 'min_', 'max_', 'avg_', 'sum_']])

# -------------------
# Arguments
# -------------------

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# subsampling
parser.add_argument('--max-per-q', type=int, default=2, help='Maximum number of training pairs that are sampled'
                                                             ' per question')
parser.add_argument('--valid-split', type=float, default=0.9, help='amount to use for training vs. model selection')
parser.add_argument('--stratify-valid', type=str2bool, default=True, help='make validation data of equal size for all'
                                                                          ' data sets')
parser.add_argument('--max-depth', type=int, default=2, help='maximally traverse this deep through candidate answers '
                                                             ' when sampling training pairs')

# fixed stuff
parser.add_argument('--cuda', type=str2bool, default=False, help='Enable CUDA support, ie. run on gpu')

# learning
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Training batch size')
parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')

# other
parser.add_argument('--name', type=str, default='regular')

# tuning
parser.add_argument('--linear-dim', type=int, default=512)
parser.add_argument('--reg', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--relation', type=int, default=-1)
parser.add_argument('--querytype', type=str, default='template')
parser.add_argument('--experiment', type=str, default='plain')
parser.add_argument('--val_type', type=str, default='zero_shot')

args = parser.parse_args()


def val(valid_loader_, model_):
    valid_loss = []
    for i, input in enumerate(valid_loader_):
        inl, inr, target = input
        l = model_.eval_pairwise(inl, inr, target)
        valid_loss.append(l)
    return np.mean(valid_loss)

def filter_dataset(origin_dataset, span_score = 8, null_odds = -5.0):
    result = []
    for sample in origin_dataset:
        filtered_sample = []
        for pred in sample:
            if pred['span_score'] > span_score and pred['null_odds'] < null_odds:
                filtered_sample.append(pred)
        if len(filtered_sample) != 0:
            result.append(filtered_sample)
    offset = len(origin_dataset) - len(result)
    return result

def train(train_loader_, valid_loader_, model_, args_, evaluator_, modelfilename_, maxepochs=-1, overwriteoptim=False):
    log, result = evaluator_.evaluate(model_)
    logger.info(log)
    best_val_loss = float('inf')
    best_val_iteration = 0

    if os.path.exists(modelfilename_):
        model_.load(modelfilename_)
        if overwriteoptim:
            model_.optimizer = optim.Adam(model_.network.parameters(), lr=args.lr)

    if maxepochs <= 0:
        maxepochs = args_.epochs

    for b in range(maxepochs):
        loss = []
        logger.info('==================== EPOCH {} ================================'.format(b))
        for i, input in enumerate(train_loader_):
            inl, inr, target = input
            l = model_.update_pairwise(inl, inr, target)
            loss.append(l)
        val_loss = val(valid_loader_, model_)
        logger.info('Epoch finished avg loss = {}'.format(np.mean(loss)))
        logger.info('Validation loss = {}'.format(val_loss))

        # check if
        if best_val_loss > val_loss:
            # save model
            logger.info('BEST EPOCH SO FAR --> safe model')
            model_.safe(modelfilename_)
            best_val_loss = val_loss
            best_val_iteration = 0
        best_val_iteration += 1
        if best_val_iteration > 10:
            # stop training
            logger.info("EARLY STOPPING")
            break
    model_.load(modelfilename_)

    # evaluate
    log, result = evaluator_.evaluate(model_)

    logger.info(log)
    return result

if __name__ == "__main__":
    # Prepare logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    ts = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    logfilename = os.path.join('..', 'out', 'logs', '{}-{}-{}-{}.log'.format(str(args.relation), args.querytype, args.name, ts))
    modelfilename = os.path.join('..', 'out', 'logs', '{}-{}-{}-{}.mdl'.format(str(args.relation), args.querytype, args.name, ts))
    logfile = logging.FileHandler(logfilename, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    logger.info(args)

    train_dir = os.path.join('..', 'out', 'features', args.experiment, 'known')
    train_files = [[os.path.join(train_dir, f) for f in os.listdir(train_dir) if 'train-kw_sent-feat-batch' in f]]

    val_dir = os.path.join('..', 'out', 'features', args.experiment, args.val_type)
    val_relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                              f.endswith('_test_kw_sent_meta_results.json')])))


    # load training data
    logger.info('Loading {} files with training samples..'.format(sum(len(x) for x in train_files)))
    train_data, valid_data, normalizers = data_utils.load_subsample(train_files, feature_descriptors, args)
    logger.info('Done. Number of train pairs loaded: {} (valid = {})'.format(len(train_data), len(valid_data)))

    # load dev data
    if args.val_type == "zero_shot":
        with open(os.path.join('..', 'data', 'train', 'known', 'seen_entities.json'), 'r') as f:
            ent = json.load(f)
            all_heads = ent['heads']
            all_tails = ent['tails']
        test_data, is_rare, seen_head, seen_tail = data_utils.load_full_zs(val_relations, val_dir, os.path.join('..', 'data', 'splits', args.val_type), all_heads, all_tails)
        test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
        evaluator = EvaluatorZs(test_dataset[0], test_dataset[1], is_rare, seen_head, seen_tail)
    elif args.val_type == "known":
        with open(os.path.join('..', 'data', 'train', 'known', 'train_entities_stats.json'), 'r') as f:
            frequencies = json.load(f)
        for k in val_relations:
            assert k in frequencies
        test_data, frequency_of_relation = data_utils.load_full_known(val_relations, val_dir,
                                                                              os.path.join('..', 'data', 'splits',
                                                                                           args.val_type), frequencies)
        test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
        evaluator = EvaluatorKnown(test_dataset[0], test_dataset[1], frequency_of_relation)
    else:
        assert False

    # generate train data loader
    logger.info('Initialize training loader..')
    train_dataset = data_loader.PairwiseRankingDataSet(train_data, normalizers)
    valid_dataset = data_loader.PairwiseRankingDataSet(valid_data, normalizers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(train_dataset),
        pin_memory=args.cuda,
        collate_fn=batchify_pair
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(valid_dataset),
        pin_memory=args.cuda,
        collate_fn=batchify_pair
    )

    # init model
    logger.info('Init model..')
    model = ranker_net.RankerNet(args, train_dataset.num_feat)
    logger.info('Done.')

    # kick off training
    result = train(train_loader, valid_loader, model, args, evaluator, modelfilename)

    # EVALUATE AGAIN ON SELF TRAIN PART:

    val_dir = os.path.join('..', 'out', 'features', 'trainonknown', args.val_type)
    val_relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                              f.endswith('_test_kw_sent_meta_results.json')])))

    if args.val_type == "zero_shot":
        with open(os.path.join('..', 'data', 'train', 'known', 'seen_entities.json'), 'r') as f:
            ent = json.load(f)
            all_heads = ent['heads']
            all_tails = ent['tails']
        test_data, is_rare, seen_head, seen_tail = data_utils.load_full_zs(val_relations,
                                                                           val_dir,
                                                                           os.path.join('..', 'data', 'splits',
                                                                                        args.val_type),
                                                                           all_heads,
                                                                           all_tails)
        test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
        evaluator = EvaluatorZs(test_dataset[0], test_dataset[1], is_rare, seen_head, seen_tail)
    elif args.val_type == "known":
        with open(os.path.join('..', 'data', 'train', 'known', 'train_entities_stats.json'), 'r') as f:
            frequencies = json.load(f)
        for k in val_relations:
            assert k in frequencies
        test_data, frequency_of_relation = data_utils.load_full_known(val_relations, val_dir,
                                                                              os.path.join('..', 'data', 'splits',
                                                                                           args.val_type), frequencies)

        test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
        evaluator = EvaluatorKnown(test_dataset[0], test_dataset[1], frequency_of_relation)
    else:
        assert False

    model.load(modelfilename)
    logger.info('Evaluate self-training')
    log, results = evaluator.evaluate(model)
    logger.info(log)

    logger.info('Write results')
    print("write results")
    data_utils.write_results(val_relations, results, os.path.join('..',  'out', 'predictions', args.val_type),
                             val_dir,  os.path.join('..', 'data', 'splits', args.val_type))

    for handler in logger.handlers:
        handler.flush()



