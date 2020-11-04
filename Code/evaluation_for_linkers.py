import os
import json
import argparse
from api import naive_linker, improved_linker, enhanced_linker, find_range_property, upload_generate_train
from tqdm import tqdm
import time
import random
PRED_SUFFIX_TOKEN = '-predictions.json'


def get_all_relations(prediction_path):
    relations = [filename.split(PRED_SUFFIX_TOKEN)[0] for filename in os.listdir(prediction_path) if filename.endswith(PRED_SUFFIX_TOKEN)]
    return relations


def collect_all_pairs(prediction_path, relations):
    # relations = get_all_relations(prediction_path)
    all_pairs = dict()
    for relation in relations:
        pairs_relation = []
        with open(os.path.join(prediction_path, relation + PRED_SUFFIX_TOKEN), 'r') as f:
            for line in f:
                line_data = json.loads(line)
                pairs_relation.append(
                    {'text': line_data['prediction']['text'], 'uris': line_data['query']['answer_entity']}
                )
                all_pairs[relation] = pairs_relation

    return all_pairs


def collect_train_pairs(prediction_path, relations):
    all_train_pairs = dict()
    for relation in relations:
        pairs_relation = []
        with open(os.path.join(prediction_path, relation + PRED_SUFFIX_TOKEN), 'r') as f:
            for line in f:
                line_data = json.loads(line)
                texts = line_data['query']['answer']
                uris = line_data['query']['answer_entity']
                if len(texts) >= 2:
                    rand_texts = random.sample(texts, 2)
                    for rand_text in rand_texts:
                        pairs_relation.append({'text': rand_text, 'uris': uris})
                else:
                    pairs_relation.append({'text': texts[0], 'uris': uris})
            all_train_pairs[relation] = pairs_relation
    return all_train_pairs


def formulate_params(params, precision=4):
    return float(round(params, precision))


def check_overlap(l1, l2):
    return bool(set(l1) & set(l2))


def initial_zero_dict(relations):
    dict_zeros = dict()
    for relation in relations:
        dict_zeros[relation] = 0
    dict_zeros['ALL'] = 0
    return dict_zeros


def evaluate_naive_linker(predictions_path):

    one_hit = 0
    three_hit = 0
    five_hit = 0

    len_pairs = 0
    batch_size = 25

    relations = get_all_relations(predictions_path)
    all_pairs_dict = collect_all_pairs(predictions_path, relations)

    record_ = dict()

    for relation in tqdm(relations):
        one_hit_rel, three_hit_rel, five_hit_rel = 0, 0, 0
        len_pairs_rel = 0
        print(f'Evaluating the relation => {relation}')
        pairs_relation = all_pairs_dict[relation]
        all_ans_to_verify = []
        texts_to_pass = []
        answer_list = []
        for pair in pairs_relation:
            len_pairs += 1
            len_pairs_rel += 1
            texts_to_pass.append(pair['text'])
            answer_list.append(pair['uris'])
            if len_pairs % batch_size == 0:
                print(f'Relation[{relation}] batch-{len_pairs//batch_size}')
                ans_to_verify = naive_linker(texts_to_pass)
                all_ans_to_verify.extend(ans_to_verify)

                texts_to_pass.clear()

        if len(answer_list) != len(all_ans_to_verify) and len(texts_to_pass) != 0:
            ans_to_verify = naive_linker(texts_to_pass)
            all_ans_to_verify.extend(ans_to_verify)
            texts_to_pass.clear()

        assert len(answer_list) == len(all_ans_to_verify)

        for i in range(0, len(answer_list)):

            if check_overlap(all_ans_to_verify[i][:1], answer_list[i]):
                one_hit += 1
                one_hit_rel += 1

            if check_overlap(all_ans_to_verify[i][:3], answer_list[i]):
                three_hit += 1
                three_hit_rel += 1

            if check_overlap(all_ans_to_verify[i][:5], answer_list[i]):
                five_hit += 1
                five_hit_rel += 1

        record_[relation] = {
            'H@1': one_hit_rel, 'H@3': three_hit_rel, 'H@5': five_hit_rel, 'LEN': len_pairs_rel
        }
        res_ = dict()
        res_['H@1'], res_['H@3'], res_['H@5'], res_['LEN'] = formulate_params(one_hit / len_pairs), formulate_params(three_hit / len_pairs), formulate_params(five_hit/len_pairs), len_pairs
        with open('result_naive_linker.json', 'w') as f:
            json.dump(res_, f)

    with open('record_naive_linker.json', 'w') as f:
        json.dump(record_, f)

    return res_


def get_range_dict(relations):
    time.sleep(2)
    range_dict = dict()
    for relation in relations:
        range_dict[relation] = find_range_property(relation)

    return range_dict


def evaluate_improved_link(predictions_path):
    relations = get_all_relations(predictions_path)
    all_pairs_dict = collect_all_pairs(predictions_path, relations)
    print(f'Collected all the data, relation[{len(relations)}], all_pairs[{len(all_pairs_dict)}]')
    with open('range_dict.json', 'r') as f:
        range_dict = json.load(f)
    batch_size = 5
    one_hit, three_hit, five_hit = 0, 0, 0
    len_pairs = 0
    record_ = dict()
    for relation in tqdm(relations):
        one_hit_rel, three_hit_rel, five_hit_rel = 0, 0, 0
        len_pairs_rel = 0
        range_p = range_dict[relation]
        all_ans_to_verify = []
        texts_to_pass = []
        answer_list = []
        print(f'Processing relation--{relation}')
        pairs_relation = all_pairs_dict[relation]
        for pair in pairs_relation:
            len_pairs += 1
            len_pairs_rel += 1
            text_, uris_ = pair['text'], pair['uris']
            answer_list.append(uris_)
            texts_to_pass.append(text_)
            if len_pairs % batch_size == 0:
                print(f'Relation[{relation}] batch-{len_pairs//batch_size}')
                ans_2_verify = improved_linker(texts=texts_to_pass, restrictions=range_p)
                all_ans_to_verify.extend(ans_2_verify)
                texts_to_pass.clear()

        if len(answer_list) != len(all_ans_to_verify) and len(texts_to_pass) != 0:
            ans_2_verify = improved_linker(texts=texts_to_pass, restrictions=range_p)
            all_ans_to_verify.extend(ans_2_verify)
            texts_to_pass.clear()

        assert len(answer_list) == len(all_ans_to_verify)

        for i in range(0, len(answer_list)):
            if check_overlap(all_ans_to_verify[i][:1], answer_list[i]):
                one_hit += 1
                one_hit_rel += 1
            if check_overlap(all_ans_to_verify[i][:3], answer_list[i]):
                three_hit += 1
                three_hit_rel += 1
            if check_overlap(all_ans_to_verify[i][:5], answer_list[i]):
                five_hit += 1
                five_hit_rel += 1

        record_[relation] = {
            'H@1': one_hit_rel, 'H@3': three_hit_rel, 'H@5': five_hit_rel, 'LEN': len_pairs_rel
        }

        res_ = dict()
        res_['H@1'], res_['H@3'], res_['H@5'], res_['LEN'] = round(one_hit / len_pairs,4), round(three_hit/len_pairs,4), round(five_hit/len_pairs,4), len_pairs
        with open('result_improved_linker.json', 'w') as f:
            json.dump(res_, f)
    with open('record_improved_linker.json', 'w') as f:
        json.dump(record_, f)
    return res_


def evaluate_enhanced_linker(prediction_path):
    relations = get_all_relations(prediction_path)
    all_train_pairs = collect_train_pairs(prediction_path, relations)
    all_pairs = collect_all_pairs(prediction_path, relations)

    # upload/generate_features/train for all the datasets[relations]
    print(f'[PHASE-1] == UPLOAD + GENERATE_FEATURES + TRAIN ==')
    for dataset in tqdm(relations):
        upload_generate_train(all_train_pairs[dataset], dataset)
    batch_size = 10
    one_hit, three_hit, five_hit = 0, 0, 0
    len_pairs = 0
    record_ = dict()
    print(f'[PHASE-2] == EVALUATE ON DATASETS ==')
    for relation in tqdm(relations):
        one_hit_rel, three_hit_rel, five_hit_rel = 0, 0, 0
        len_pairs_rel = 0
        all_ans_to_verify = []
        texts_to_pass = []
        answer_list = []
        print(f'Processing relation--{relation}')
        pairs_relation = all_pairs[relation]
        for pair in pairs_relation:
            len_pairs += 1
            len_pairs_rel += 1
            text_, uris_ = pair['text'], pair['uris']
            answer_list.append(uris_)
            texts_to_pass.append(text_)

            if len_pairs % batch_size == 0:
                print(f'Relation[{relation}] batch-{len_pairs//batch_size}')
                ans_2_verify = enhanced_linker(texts=texts_to_pass, dataset=relation)
                all_ans_to_verify.extend(ans_2_verify)
                texts_to_pass.clear()
        if len(answer_list) != len(all_ans_to_verify) and len(texts_to_pass) != 0:
            ans_2_verify = enhanced_linker(texts=texts_to_pass, dataset=relation)
            all_ans_to_verify.extend(ans_2_verify)
            texts_to_pass.clear()

        assert len(all_ans_to_verify) == len(answer_list)

        for i in range(0, len(answer_list)):
            if check_overlap(all_ans_to_verify[i][:1], answer_list[i]):
                one_hit += 1
                one_hit_rel += 1
            if check_overlap(all_ans_to_verify[i][:3], answer_list[i]):
                three_hit += 1
                three_hit_rel += 1
            if check_overlap(all_ans_to_verify[i][:5], answer_list[i]):
                five_hit += 1
                five_hit_rel += 1

        record_[relation] = {
            'H@1': one_hit_rel, 'H@3': three_hit_rel, 'H@5': five_hit_rel, 'LEN': len_pairs_rel
        }

        res_ = dict()
        res_['H@1'], res_['H@3'], res_['H@5'], res_['LEN'] = round(one_hit / len_pairs, 4), \
            round(three_hit / len_pairs, 4), round(five_hit / len_pairs, 4), len_pairs
        with open('result_enhanced_linker.json', 'w') as f:
            json.dump(res_, f)
    with open('record_enhanced_linker.json', 'w') as f:
        json.dump(record_, f)
    return record_


def main():
    print(f'Get index={args.index}')
    index = int(args.index)
    pred_path = args.prediction_path
    if index == 1:
        print('Run index 1 => NAIVE_LINKER')
        result = evaluate_naive_linker(pred_path)
    elif index == 2:
        print('Run index 2 => IMPROVED_LINKER')
        result = evaluate_improved_link(pred_path)
    elif index == 3:
        print('Run index 3 => ENHANCED_LINKER')
        result = evaluate_enhanced_linker(pred_path)
    elif index == 4:
        print('Run both index 1&2')
        result_1 = evaluate_naive_linker(pred_path)

        result_2 = evaluate_improved_link(pred_path)

    if result:
        print(result)

    if result_1:
        print(result_1)

    if result_2:
        print(result_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative experiment script.')

    parser.add_argument('--prediction_path', type=str, default='./out/predictions/known', help='path to the predictions folder')
    parser.add_argument('--index', type=int, default=1, help='1 for naive_linker, 2 for improved_linker, 3 for enhanced_linker')

    args = parser.parse_args()

    main()