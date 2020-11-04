import torch
from apis import get_article, get_label, construct_query
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
from collections import OrderedDict
from utils import tensor_to_list, get_qa_inputs, preliminary_predictions, best_predictions, \
    prediction_probabilities, compute_score_difference, read_article_doc
import threading


class PredictThread(threading.Thread):

    def __init__(self, func, args=()):
        super(PredictThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class KBCompleter:
    def __init__(self, pretrained_model_name_or_path="deepset/bert-large-uncased-whole-word-masking-squad2"):
        self.READER_PATH = pretrained_model_name_or_path
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.wiki_path = os.path.join(os.getcwd(), 'data', 'wiki')
        self.label_path = os.path.join(os.getcwd(), 'data', 'label')
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def predict(self, question, context):
        self.tokenize(question, context)
        return self.get_answer(question, context)

    def construct_query(self, qid, pid):
        return construct_query(qid, pid)

    def prod_predict(self, query):
        name = self.get_article_label(query)
        context = read_article_doc(self.wiki_path, name)
        if context == None:
            raise ValueError('DOC TOO SHORT !')
        question = query['question']
        results = self.predict(question, context)
        return results

    def tokenize(self, question, context):
        self.inputs = get_qa_inputs(question, context, self.tokenizer)
        self.input_ids = tensor_to_list(self.inputs["input_ids"])[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self, question, context):
        if self.chunked:
            answers = []
            threads = []
            for k, chunk in self.inputs.items():
                thread = PredictThread(func=self.get_robust_prediction, args=(None, None, 5, 0.0, chunk))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
                ans, span_score, probability, null_odds, evidence = thread.get_result()
                answers.append({'text': ans, 'span_score': span_score, 'probability': probability, 'null_odds': null_odds, 'evidence': evidence})
            return answers
        else:
            ans, span_score, probability, null_odds, evidence = self.get_robust_prediction(question, context)
            return [{'text': ans, 'span_score': span_score, 'probability': probability, 'null_odds': null_odds, 'evidence': evidence}]

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))

    # Get answer from the context
    """
    return text_span, span_score, probability, null_odds
    """
    def get_robust_prediction(self, question, context, nbest=5, null_threshold=0.0, inputs=None):
        if question != None and context != None:
            inputs = get_qa_inputs(question, context, self.tokenizer)
        start_logits, end_logits = self.model(**inputs)

        # get sensible preliminary predictions, sorted by score
        prelim_preds = preliminary_predictions(start_logits,
                                               end_logits,
                                               inputs['input_ids'],
                                               nbest)

        # narrow that down to the top nbest predictions
        tokens = tensor_to_list(inputs['input_ids'])[0]
        s_logits = tensor_to_list(start_logits)[0]
        e_logits = tensor_to_list(end_logits)[0]
        nbest_preds = best_predictions(tokens, s_logits, e_logits, prelim_preds, nbest, self.tokenizer)

        # compute the probability of each prediction - nice but not necessary
        probabilities = prediction_probabilities(nbest_preds)

        # compute score difference
        score_difference = compute_score_difference(nbest_preds)

        span_score = nbest_preds[0].start_logit + nbest_preds[0].end_logit
        # if score difference > threshold, return the null answer
        if score_difference > null_threshold:
            return "", span_score, probabilities[-1], 1000, ""
        else:
            text_span = nbest_preds[0].text
            return text_span, span_score, probabilities[0], score_difference, nbest_preds[0].evidence
        return inputs

    # Download the label and the article
    def get_article_label(self, query):
        name = query['wikipedia_link'].split('wiki/')[-1]
        res_code_1 = get_article(name, self.wiki_path)
        get_label(query["property"], self.label_path)
        if res_code_1 < 0:
            raise LookupError('Doc not found !')
        return name