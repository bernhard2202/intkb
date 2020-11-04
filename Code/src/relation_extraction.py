"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from reader.tokenization import FullTokenizer

from retriever.sentence_retriever import retrieve_sentences_multi_kw_query
from retriever.utils import get_filename_for_article_id
from utils.answer_normalization import normalize_answer
import numpy as np

import collections
import json
import os

from multiprocessing import Pool as ProcessPool
from reader.utils import decode_bert_predictions, \
    model_fn_builder, FeatureWriter, input_fn_builder, InputFeatures, check_is_max_context, \
    sort_predictions_to_query, aggregate_and_collect_features, sort_null_odds_to_query, sort_preds_v2_to_query
from tqdm import tqdm


from reader import modeling, tokenization


import six
import tensorflow as tf

PROCESS_TOK = None

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("out_name", '5kw', "prefix for eval results")
flags.DEFINE_string("feat_path", 'out/features', "prefix for eval results")  # TODO

flags.DEFINE_string("wiki_data", None,
                    "SQuAD json for training. E.g., train-v1.1.json")
flags.DEFINE_integer(
    "k_sentences", 5, 'number of sentences to retrieve')
flags.DEFINE_integer(
    "num_kw_queries", 5, 'number of sentences to retrieve')

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_string('split', 'known', 'split to evaluate on either known or one_shot')

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_string("data_path", 'iterations/iter_0/data', "data path")

# Move to flags
PREDS_PER_SPAN = 1
NUM_CANDIDATES = 40


def init(tok_class, tok_opts):
    global PROCESS_TOK
    PROCESS_TOK = tok_class(**tok_opts)


def _convert_to_id(tokens):
    global PROCESS_TOK
    return PROCESS_TOK.convert_tokens_to_ids(tokens)


def _tokenize_question(text, max_query_length=64):
    global PROCESS_TOK
    query_tokens = PROCESS_TOK.tokenize(text)
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0: max_query_length]
    return query_tokens


def _tokenize_document(text):
    global PROCESS_TOK

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    all_doc_tokens = []
    tok_to_orig_index = []
    for (i, token) in enumerate(doc_tokens):
        sub_tokens = PROCESS_TOK.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    return all_doc_tokens, tok_to_orig_index, doc_tokens


def retrieve_sent(query, keywords, top_k, flag):
    fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
    if flag != 'test':
        if flag == 'neg' or len(query['answer']) == 0:
            fname = fname.split('.txt')[0]+'_negative.txt'
    results, scores = retrieve_sentences_multi_kw_query(keywords, [os.path.join(FLAGS.wiki_data, fname)], k=top_k)
    return results, scores


def build_bert_inputs(doc_scores, queries, processes):
    documents = doc_scores[0]
    scores = doc_scores[1]
    flat_docs = [d for doc in documents for d in doc]
    flat_scores = [s for score in scores for s in score]
    doc_start_ids = np.cumsum([len(docs) for docs in documents])

    doc_tokens = processes.map_async(_tokenize_document, flat_docs)
    doc_tokens = doc_tokens.get()

    eval_examples = []
    eval_features = []
    unique_id = 1000000000
    sample_id = 0

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    j = 0
    for qID in range(len(documents)):  # i counts the example
        oldj = j
        for query in queries:
            query = query + ' ?'
            j = oldj
            query_tokens = processes.map(_tokenize_question, [query])[0]
            max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

            while j < doc_start_ids[qID]:
                all_doc_tokens = doc_tokens[j][0]

                # filter headline sentence from wikipedia

                if len(doc_tokens[j][2]) < 4 or (doc_tokens[j][2][0] == "===" and doc_tokens[j][2][-1] == "==="):
                    j += 1
                    continue

                doc_tok_to_ind = doc_tokens[j][1]
                # length = min(len(all_doc_tokens), max_tokens_for_doc)
                doc_spans = []
                start_offset = 0
                _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                    "DocSpan", ["start", "length"])
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append(_DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, FLAGS.doc_stride)

                eval_examples.append({'qid': qID, 'sample_id': sample_id, 'doc_tokens': doc_tokens[j][2],
                                      'query_tokens': query_tokens, 'doc_score': flat_scores[j], 'doc_pos': j - oldj})
                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    tokens = []
                    token_to_orig_map = {}
                    token_is_max_context = {}
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for token in query_tokens:
                        tokens.append(token)
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)

                    for i in range(doc_span.length):
                        split_token_index = doc_span.start + i
                        token_to_orig_map[len(tokens)] = doc_tok_to_ind[split_token_index]

                        is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
                        token_is_max_context[len(tokens)] = is_max_context
                        tokens.append(all_doc_tokens[split_token_index])
                        segment_ids.append(1)

                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    input_ids = processes.map(_convert_to_id, [tokens])[0]

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    while len(input_ids) < FLAGS.max_seq_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)

                    assert len(input_ids) == FLAGS.max_seq_length
                    assert len(input_mask) == FLAGS.max_seq_length
                    assert len(segment_ids) == FLAGS.max_seq_length

                    feature = InputFeatures(
                        unique_id=unique_id,
                        example_index=sample_id,
                        question_id=qID,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids)

                    eval_features.append(feature)
                    eval_writer.process_feature(feature)
                    unique_id += 1
                j += 1
                sample_id += 1
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    return eval_examples, eval_features, eval_writer


# TODO remove answers again or make it optional
def pass_through_bert(doc_scores, queries, answers, processes, estimator, relation):
    eval_examples, eval_features, eval_writer = \
        build_bert_inputs(doc_scores, queries, processes)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
        if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    all_nbest_json , preds, score_diff_json, preds_v2 = decode_bert_predictions(eval_examples, eval_features, all_results,
                                        FLAGS.n_best_size, FLAGS.max_answer_length,
                                        False, FLAGS)
    # sorted_null_odds = sort_null_odds_to_query(score_diff_json, eval_examples)
    # sorted_predictions_v2 = sort_preds_v2_to_query(preds_v2, eval_examples)

    sorted_predictions = sort_predictions_to_query(preds_v2, eval_examples, score_diff_json, PREDS_PER_SPAN)
    result = aggregate_and_collect_features(sorted_predictions, answers, NUM_CANDIDATES)

    # output_preds_v2 = os.path.join(FLAGS.output_dir, "predictions_v2_{}.json".format(relation))
    # preds_v2_path = os.path.join(FLAGS.output_dir, "preds_v2_{}.json".format(relation))
    # out_all_nbest_json = os.path.join(FLAGS.output_dir, "all_nbest_json_{}.json".format(relation))
    # output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds_{}.json".format(relation))
    # score_diff_json_filename = os.path.join(FLAGS.output_dir, "score_diff_{}.json".format(relation))
    # output_examples = os.path.join(FLAGS.output_dir, "examples_{}.json".format(relation))
    # output_preds_v1 = os.path.join(FLAGS.output_dir, "prediction_v1_{}.json".format(relation))
    # out_result = os.path.join(FLAGS.output_dir, "out_result_{}.json".format(relation))

    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(out_result, "w") as writer:
    #         writer.write(json.dumps(result, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(score_diff_json_filename, "w") as writer:
    #         writer.write(json.dumps(score_diff_json, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
    #         writer.write(json.dumps(sorted_null_odds, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(output_preds_v2, "w") as writer:
    #         writer.write(json.dumps(sorted_predictions_v2, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(preds_v2_path, "w") as writer:
    #         writer.write(json.dumps(preds_v2, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(out_all_nbest_json, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(output_examples, "w") as writer:
    #         writer.write(json.dumps(eval_examples, indent=4) + "\n")
    # if FLAGS.version_2_with_negative:
    #     with tf.gfile.GFile(output_preds_v1, "w") as writer:
    #         writer.write(json.dumps(preds, indent=4) + "\n")

    return result


def run_for_relation(relation, processes, estimator, phase):
    tf.logging.info('start evaluation for {}'.format(relation))
    tf.logging.info('load data for relation type..')


    for split in ['train', 'test', 'neg']:
        if (FLAGS.split != 'neg' and FLAGS.split != 'custom_neg' and FLAGS.split != 'custom') and split == 'neg':
            continue
        if FLAGS.split == 'neg' and split != 'neg':
            continue

        if FLAGS.split == 'prod' and split == 'train':
            continue
        if FLAGS.split == 'prod' and split == 'neg':
            continue

        if FLAGS.split == 'custom_neg' and split != 'neg':
            continue


        # load query string
        with open(os.path.join('data', 'labels', '{}_labels.json'.format(relation)), 'r') as f:
            keyword_query = json.load(f)
        if FLAGS.split != 'neg' and FLAGS.split != 'custom' and FLAGS.split != 'custom_neg' and FLAGS.split != 'prod':
            with open(os.path.join('data', 'splits', FLAGS.split, '{}_{}.json'.format(relation, split)), 'r') as f:
                jdata = json.load(f)
        elif FLAGS.split == 'neg':
            with open(os.path.join('data', 'splits', phase, '{}_{}.json'.format(relation, split)), 'r') as f:
                jdata = json.load(f)

        if FLAGS.split == 'custom' or 'prod':
            with open(os.path.join(FLAGS.data_path, '{}_{}.json'.format(relation, split)), 'r') as f:
                jdata = json.load(f)

        if FLAGS.split == 'custom_neg':
            with open(os.path.join(FLAGS.data_path, '{}_{}.json'.format(relation, split)), 'r') as f:
                jdata = json.load(f)


        keyword_query.sort(key=lambda s: -len(s))
        bert_queries = keyword_query[0:min(len(keyword_query), FLAGS.num_kw_queries)]

        ALL_DATA = len(jdata)
        NOT_FOUND = 0

        tf.logging.info('retrieve documents and run intermediate evaluation.. ')
        documents = []
        scores = []
        answers = []
        i = -1
        for query in tqdm(jdata):
            i = i + 1
            docs, score = retrieve_sent(query, keyword_query, FLAGS.k_sentences, split)
            if len(docs) == 0:
                NOT_FOUND += 1
            else:
                documents.append(docs)
                scores.append(score)
                if 'answer' not in query:
                    answers.append([])
                else:
                    answers.append([normalize_answer(a) for a in query['answer']])


        tf.logging.info('build bert_inputs..')
        batch_size = 500
        batches = [(documents[i: i + batch_size], scores[i: i + batch_size])
                   for i in range(0, len(documents), batch_size)]
        answer_batch = [(answers[i: i + batch_size])
                        for i in range(0, len(answers), batch_size)]
        found_in_top_ranked = 0
        found_in_any_rank = 0
        for bn, batch in enumerate(batches):
            tf.logging.info("Running prediction for batch {}".format(bn))
            ranking_features = pass_through_bert(batch, bert_queries, answer_batch[bn], processes,
                                                 estimator, relation)
            assert len(ranking_features) == len(batch[0]), (len(ranking_features), len(batch))
            with open(os.path.join(FLAGS.feat_path,
                                   '{}-{}-{}-feat-batch-{}.txt'.format(relation, split, FLAGS.out_name, bn)), 'w') as f:
                for o in ranking_features:
                    found_in_top_ranked += o[0]['target']
                    if sum([o_['target'] for o_ in o]) > 0:
                        found_in_any_rank += 1
                    f.write(json.dumps(o) + '\n')

        final_results = {
            'found_in_reader': found_in_top_ranked,
            'found_at_any_rank': found_in_any_rank,
            'N': ALL_DATA,
            'no_doc_found': NOT_FOUND,
        }

        with open(os.path.join(FLAGS.feat_path, '{}_{}_{}_meta_results.json'.format(relation, split, FLAGS.out_name)),
                  'w') as f:
            json.dump(final_results, f)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tok_class = FullTokenizer
    tok_options = {'vocab_file': FLAGS.vocab_file, 'do_lower_case': False}
    processes = ProcessPool(
        5,  # TODO move to flags
        initializer=init,
        initargs=(tok_class, tok_options)
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=FLAGS.init_checkpoint)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=1,
        num_train_steps=1,
        num_warmup_steps=1,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=8,
        predict_batch_size=FLAGS.predict_batch_size)
    phase ='known'
    if FLAGS.split != 'custom':
        if FLAGS.split == 'known':
            relations = [f[:-11] for f in os.listdir(os.path.join('data', 'splits', FLAGS.split)) if f.endswith('_train.json')
                        and f.startswith('P')]
        elif FLAGS.split == 'zero_shot' :
            relations = [f[:-10] for f in os.listdir(os.path.join('data', 'splits', FLAGS.split)) if f.endswith('_test.json')
                        and f.startswith('P')]
        elif FLAGS.split == 'neg':
            phase = FLAGS.feat_path.split('/')[-1]
            if phase == 'zero_shot':
                relations = [f[:-9] for f in os.listdir(os.path.join('data', 'splits', 'zero_shot')) if
                             f.endswith('_neg.json')
                             and f.startswith('P')]
            elif phase == 'known':
                relations = [f[:-9] for f in os.listdir(os.path.join('data', 'splits', 'known')) if
                             f.endswith('_neg.json')
                             and f.startswith('P')]
            else:
                print('[ERROR] --feat_path is wrong !!!! must be zero_shot/known')

    if FLAGS.split == 'custom':
        relations = [f[:-10] for f in os.listdir(FLAGS.data_path) if f.endswith('_test.json') and f.startswith('P')]

    if FLAGS.split == 'custom_neg':
        relations = [f[:-9] for f in os.listdir(FLAGS.data_path) if f.endswith('_neg.json') and f.startswith('P')]

    if FLAGS.split == 'prod':
        relations = [f[:-10] for f in os.listdir(FLAGS.data_path) if f.endswith('_test.json') and f.startswith('P')]

    for relation in relations:
        run_for_relation(relation, processes, estimator, phase)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
