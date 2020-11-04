import os
import json
import numpy as np
import random

train_dir = os.path.join('out', 'features', 'trainonknown', 'zero_shot')
train_files = [f for f in os.listdir(train_dir) if 'train-kw_sent-feat-batch' in f]

candidate_example = []
train_score = []
train_doc_score = []
train_target = []
train_num_occ = []
for train_file in train_files:
    relation = train_file[:train_file.find('-')]
    with open(os.path.join('data', 'labels', '{}_labels.json'.format(relation)), 'r') as f:
        query_list = json.load(f)
        query_list.sort(key=lambda s: -len(s))
    query_list = query_list[0:min(len(query_list), 5)]
    query_list = [q + ' ?' for q in query_list]

    with open(os.path.join(train_dir, train_file), 'r') as f:
        for line in f:
            data = json.loads(line)
            targets = []
            scores = []
            doc_scores = []
            num_occ = []
            for pred in data:
                if 'doc_score' not in pred:
                    continue
                targets.append(pred['target'])
                doc_scores.append(pred['doc_score'])
                scores.append(pred['sum_span_score'])
                num_occ.append(pred['num_occ'])
            if len(scores) == 0:
                continue
            candidate_example.append((data[0], random.choice(query_list)))
            train_score.append(scores)
            train_target.append(targets)
            train_num_occ.append(num_occ)
            train_doc_score.append(doc_scores)

with open(os.path.join('data', 'train', 'known', 'known_relations_train.json'), 'r') as f:
    old_td = json.load(f)['data']
print(len(old_td))
ID = 500

th = 0.5
gen = 0
for i in range(len(train_score)):
    # 1. normalize
    curr_scores = np.array(train_score[i]) / sum(train_score[i])
    curr_doc_scores = np.array(train_doc_score[i]) / sum(train_doc_score[i])
    if len(curr_scores) <= 1:
        continue
    if curr_scores[0] - curr_scores[1] > th and curr_doc_scores[0] - curr_doc_scores[1] > th and train_num_occ[i][
        0] > 1:
        para = candidate_example[i][0]['doc_tokens']
        old_td.append({'title': ID,
                       'paragraphs': [{"context": " ".join(candidate_example[i][0]['doc_tokens']),
                                       "qas": [{"id": ID, "question": candidate_example[i][1],
                                                "answers": [{"answer_start": " ".join(
                                                    candidate_example[i][0]['doc_tokens']).lower().find(
                                                    candidate_example[i][0]['text'].lower()),
                                                    "text": candidate_example[i][0]['text']}]}]}]})
        ID += 1

print(len(old_td))
with open(os.path.join('data', 'train', 'known', 'known_relations_self_annotate.json'), 'w') as f:
    json.dump({'data': old_td}, f)
print(len(old_td))