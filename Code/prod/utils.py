import os
import numpy as np
import collections
import string


def get_folder_for_id(id):
    first = id[0]
    if not first.isalpha() and not first.isdigit():
        return 'other'
    return first


def get_filename_for_article_id(wiki_id):
    def _get_folder_for_id(_id):
        first = _id[0]
        if not first.isalpha() and not first.isdigit():
            return 'other'
        return first
    wiki_id = wiki_id.replace('/', '_slash_')
    return os.path.join(_get_folder_for_id(wiki_id), '{}.txt'.format(wiki_id))


def get_qa_inputs(question, context, tokenizer):
    # load the example, convert to inputs, get model outputs
    return tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')


def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()


def get_clean_text(tokens, tokenizer):
    text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
        )
    # Clean whitespace
    text = text.strip()
    text = " ".join(text.split())
    return text


def prediction_probabilities(predictions):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    all_scores = [pred.start_logit+pred.end_logit for pred in predictions]
    return softmax(np.array(all_scores))


def tokens_to_sentence(tokens):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def get_evidence(tokens, start, end, tokenizer):
    # 999 1029, 1012 end marks of a sentence.
    left, right = 0, 0
    for i in range(start, 0, -1):
        if tokens[i] in [999, 1029, 1012, 102]:
            left = i+1 # not cover the last end mark and [SEP]
            break
    for i in range(end, len(tokens)):
        if tokens[i] in [999, 1029, 1012]:
            right = i+1 # cover the period/question mark/exclamation mark.
            break
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens[left:right]))
    result.strip()
    result = " ".join(result.split())
    return result


def preliminary_predictions(start_logits, end_logits, input_ids, nbest):
    # convert tensors to lists
    start_logits = tensor_to_list(start_logits)[0]
    end_logits = tensor_to_list(end_logits)[0]
    tokens = tensor_to_list(input_ids)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    start_indexes = [idx for idx, logit in start_idx_and_logit[:nbest]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:nbest]]

    # question tokens are between the CLS token (101, at position 0) and first SEP (102) token
    question_indexes = [i + 1 for i, token in enumerate(tokens[1:tokens.index(102)])]

    # keep track of all preliminary predictions
    PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # throw out invalid predictions
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]
                )
            )
    # sort prelim_preds in descending score order
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    return prelim_preds


def best_predictions(tokens, start_logits, end_logits, prelim_preds, nbest, tokenizer):
    # keep track of all best predictions

    # This will be the pool from which answer probabilities are computed
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit", "evidence"]
    )

    nbest_predictions = []
    seen_predictions = []
    for pred in prelim_preds:
        if len(nbest_predictions) >= nbest:
            break
        if pred.start_index > 0:  # non-null answers have start_index > 0

            toks = tokens[pred.start_index: pred.end_index + 1]
            text = get_clean_text(toks, tokenizer)

            # if this text has been seen already - skip it
            if text in seen_predictions:
                continue

            # flag text as being seen
            seen_predictions.append(text)

            # add this text to a pruned list of the top nbest predictions
            nbest_predictions.append(
                BestPrediction(
                    text=text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    evidence="" if not text else get_evidence(tokens, pred.start_index, pred.end_index, tokenizer)
                )
            )

    # Add the null prediction
    nbest_predictions.append(
        BestPrediction(
            text="",
            start_logit=start_logits[0],
            end_logit=end_logits[0],
            evidence=""
        )
    )
    return nbest_predictions


def compute_score_difference(predictions):
    """ Assumes that the null answer is always the last prediction """
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null


def read_article_doc(wiki_path, name):
    with open(os.path.join(wiki_path, get_filename_for_article_id(name)), 'r') as f:
        doc = f.read()
    if len(doc) > 2:
        return doc
    else:
        return None