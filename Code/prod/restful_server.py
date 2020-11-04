import json
import os
from flask import Flask, request, jsonify
import time
from datetime import datetime
from kbcompleter import KBCompleter
from apis import enhanced_linker
import time

app = Flask(__name__)
kbcompleter = KBCompleter()


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Success",
        "time": datetime.now()
    })


@app.route('/id2ans', methods=['GET'])
def get_fact():
    qid = request.args.get('subject_id')
    pid = request.args.get('property_id')
    start_1 = time.perf_counter()
    query = kbcompleter.construct_query(qid, pid)
    end_1 = time.perf_counter()

    start_2 = time.perf_counter()
    predictions = kbcompleter.prod_predict(query)
    end_2 = time.perf_counter()
    top_prediction = sorted(predictions, key=lambda x:x['span_score'], reverse=True)[0]
    start_3 = time.perf_counter()
    objects = [] if not top_prediction['text'] else enhanced_linker(texts=[top_prediction['text']], dataset=query['property'])
    end_3 = time.perf_counter()
    top_prediction['object'] = objects
    result = {"query": query, "prediction": top_prediction, 'time': {
        'construct_query': end_1 - start_1,
        'prediction': end_2 - start_2,
        'linking': end_3 - start_3
    }}
    return jsonify(result)


@app.route('/context2ans', methods=['POST'])
def get_answer_from_context():
    if request.method == 'POST':
        from_post = request.get_data()
        json_post = json.loads(from_post)
        context, question = json_post['context'], json_post['question']
        return jsonify(kbcompleter.predict(question, context))
    else:
        return 'POST REQUEST ONLY !'


if __name__ == '__main__':
    app.run(debug=True)