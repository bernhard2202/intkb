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
import pickle

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
parser.add_argument('--name', type=str, default='kw_sent')

# tuning
parser.add_argument('--linear-dim', type=int, default=512)
parser.add_argument('--reg', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--relation', type=int, default=-1)
parser.add_argument('--querytype', type=str, default='template')
parser.add_argument('--experiment', type=str, default='trainonknown')
parser.add_argument('--val_type', type=str, default='zero_shot')
parser.add_argument('--model_path', type=str, default='/home/guo/rankerNet/ranker.mdl')
parser.add_argument('--data_path', type=str, default='path/to/data_path')
parser.add_argument('--feature_path', type=str, default='path/to/feature_path')
parser.add_argument('--output_path', type=str, default='path/to/output')

args = parser.parse_args()


if __name__ == "__main__":
    # Prepare logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    ts = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    logfilename = os.path.join('out', 'logs', '{}-{}-{}-{}.log'.format(str(args.relation), args.querytype, args.name, ts))
    modelfilename = args.model_path
    logfile = logging.FileHandler(logfilename, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    logger.info(args)

    # TODO Move to args if needed
    path_tmp = '/home/guo/rankerNet'
    path_list_1 = os.path.join(path_tmp, 'list_1.txt')
    path_list_2 = os.path.join(path_tmp, 'list_2.txt')
    if os.path.exists(path_list_1) and os.path.exists(path_list_2):
        with open(path_list_1, "rb") as fp:
            train_data = pickle.load(fp)
        with open(path_list_2, "rb") as fp:
            normalizers = pickle.load(fp)
    else:
        train_dir = os.path.join('out', 'features', 'trainonknown', 'known') # DO NOT CHANGE
        train_files = [[os.path.join(train_dir, f) for f in os.listdir(train_dir) if 'train-kw_sent-feat-batch' in f]]
        train_data, _, normalizers = data_utils.load_subsample(train_files, feature_descriptors, args)
        with open(path_list_1, "wb") as fp:
            pickle.dump(train_data, fp)
        with open(path_list_2, "wb") as fp:
            pickle.dump(normalizers, fp)
    train_dataset = data_loader.PairwiseRankingDataSet(train_data, normalizers)
    # init model
    logger.info('Init model..')
    model = ranker_net.RankerNet(args, train_dataset.num_feat)
    logger.info('Done.')

    val_dir = args.feature_path
    val_relations = sorted(list(set([f[:-len('_test_kw_sent_meta_results.json')] for f in os.listdir(val_dir) if
                              f.endswith('_test_kw_sent_meta_results.json')])))

    with open(os.path.join('data', 'train', 'known', 'seen_entities.json'), 'r') as f:
        ent = json.load(f)
        all_heads = ent['heads']
        all_tails = ent['tails']
    test_data, is_rare, seen_head, seen_tail = data_utils.load_full_zs(val_relations,
                                                                       val_dir,
                                                                       args.data_path,
                                                                       all_heads,
                                                                       all_tails)
    test_dataset = data_utils.build_validation_dataset(test_data, normalizers)
    evaluator = EvaluatorZs(test_dataset[0], test_dataset[1], is_rare, seen_head, seen_tail)

    model.load(modelfilename)
    logger.info('Ranking')
    log, results = evaluator.evaluate(model)
    logger.info(log)

    logger.info('Write results')
    print("write results")
    data_utils.write_results(val_relations, results, args.output_path,
                             val_dir, args.data_path)

    for handler in logger.handlers:
        handler.flush()



