import json
import os
from flask import Flask, request, jsonify
from utils import construct_query, make_predictions, call_enhanced_linker, call_ranker_net, get_article, get_filename_for_article_id, get_labels_data
from qa_apis import get_ans_from_context
import time
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Success",
        "time": datetime.now()
    })


@app.route('/fact', methods=['GET'])
def get_fact():
    overall_start = time.time()
    qid = request.args.get('subject_id')
    pid = request.args.get('property_id')
    query = [construct_query(qid, pid)]
    if not os.path.exists('/tmp/data'):
        os.makedirs('/tmp/data')
    if not os.path.exists('/tmp/features'):
        os.makedirs('/tmp/features')
    if not os.path.exists('/tmp/predictions'):
        os.makedirs('/tmp/predictions')

    with open('/tmp/data/{}_test.json'.format(pid), 'w') as f:
        json.dump(query, f)

    # prepare article and label.
    fname = get_filename_for_article_id(query[0]['wikipedia_link'].split('wiki/')[-1])
    if not os.path.exists(os.path.join('data', 'wiki', fname)):
        print('Article not exists start downloading')
        get_article()
    if not os.path.exists(os.path.join('data', 'labels', '{}_labels.json'.format(pid))):
        print('Label not exists start downloading')
        get_labels_data(relations=[pid])

    # make prediction
    start_time = time.time()
    print('[INFO] Prediction')
    make_predictions(model_path='/home/guo/model', data_path='/tmp/data', feature_path='/tmp/features')
    time_1 = (time.time() - start_time)

    # pass to rankerNet
    start_time = time.time()
    print('[INFO] rankerNet')
    call_ranker_net()
    time_2 = (time.time() - start_time)

    # pass to enhanced_linker
    start_time = time.time()
    print('[INFO] Linker')
    result = call_enhanced_linker(pid)
    time_3 = time.time() - start_time

    # return fact
    result["runtime"] = {
        "time_predict": time_1,
        "time_rankerNet": time_2,
        "time_linker": time_3,
        "overall_":  time.time() - overall_start
    }
    result["time"] = str(datetime.now())
    return jsonify(result)


@app.route('/ans4question', methods=['POST'])
def get_answer_from_context():
    start_time = time.time()
    # context = request.args.get("context")
    # question = request.args.get("question")
    if request.method == 'POST':
        from_post = request.get_data()
        json_post = json.loads(from_post)
        answer, span_score = get_ans_from_context(model, tokenizer, json_post['context'], json_post['question'])
    else:
        return 'POST REQUEST ONLY !'

    return jsonify({
        'context': json_post['context'],
        'question': json_post['question'],
        'answer': answer,
        'span_score': float(span_score),
        'process_time': time.time() - start_time
    })


if __name__ == '__main__':
    app.run(debug=True)