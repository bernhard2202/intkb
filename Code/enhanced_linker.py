import json
import os
import string
from api import enhanced_linker
from tqdm import tqdm
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import argparse

PRED_SUFFIX = '-predictions.json'

endpoint = "http://qanswer-core1.univ-st-etienne.fr/api/endpoint/open/wikidata/sparql"

user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
sparql = SPARQLWrapper(endpoint, agent=user_agent)


def tokens_to_sentence(tokens):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def get_wikidata_uri(wikipedia_link):
    query = """
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    select * where {
     <"""+ wikipedia_link +"""> schema:about ?o
    }"""
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    resultsAll = sparql.query().convert()
    wikidata_uri = resultsAll['results']['bindings'][0]['o']['value']
    return wikidata_uri


def add_entity_to_predictions(folder_path):
    categories = os.listdir(folder_path)
    for cate_ in categories:
        print('Category: {}'.format(cate_))
        prefix = folder_path
        rels_cate_ = [f for f in os.listdir(os.path.join(prefix, cate_)) if f.endswith(PRED_SUFFIX)]
        for rel in rels_cate_:
            print(f'Relation: {rel}')
            samples_rel = []
            with open(os.path.join(prefix, cate_, rel), 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    wikipedia_link = sample['query']['wikipedia_link']
                    wikidata_link = get_wikidata_uri(wikipedia_link)
                    sample['query']['entity'] = wikidata_link
                    samples_rel.append(sample)

            with open(os.path.join(prefix, cate_, rel), 'w') as f:
                for sample in samples_rel:
                    json.dump(sample, f)
                    f.write('\n')


def clean_up_dataset(folder_path):
    categories = os.listdir(folder_path)
    for cate_ in tqdm(categories):
        print('Category: {}'.format(cate_))
        prefix = folder_path
        rels_cate_ = [f for f in os.listdir(os.path.join(prefix, cate_)) if f.endswith(PRED_SUFFIX)]
        for rel in rels_cate_:
            samples_rel = []
            with open(os.path.join(prefix, cate_, rel), 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    if sample['prediction']['null_odds'] == 1000 or not sample['prediction']['text']:
                        continue
                    question_, wikipedia_link_, entity_ = sample['query']['question'], sample['query'][
                        'wikipedia_link'], sample['query']['entity']
                    sample['query'] = {}
                    sample['query']['category'] = cate_
                    sample['query']['property'] = rel.split(PRED_SUFFIX)[0]
                    sample['query']['question'], sample['query']['wikipedia_link'], sample['query'][
                        'entity'] = question_, wikipedia_link_, entity_

                    text_, score_, null_odds_ = sample['prediction']['text'], sample['prediction']['span_score'], \
                                                sample['prediction']['null_odds']
                    evidence_ = tokens_to_sentence(sample['prediction']['doc_tokens'])
                    sample['prediction'] = {}
                    sample['prediction']['text'], sample['prediction']['span_score'], sample['prediction'][
                        'null_odds'] = text_, score_, null_odds_
                    sample['prediction']['evidence'] = evidence_

                    objects = enhanced_linker(texts=[text_], dataset=rel.split(PRED_SUFFIX)[0])
                    if len(objects) == 1:
                        sample['prediction']['object'] = objects[0]
                    else:
                        print(f'text: {text_}')
                        print(objects)
                    samples_rel.append(sample)

            with open(os.path.join(prefix, cate_, rel), 'w') as f:
                for sample in samples_rel:
                    json.dump(sample, f)
                    f.write('\n')


def print_info(all_preds, span_score, null_odds):
    preds_with_ans = [pred for pred in all_preds if pred['prediction']['text'] != '']
    preds_with_ans_object = [pred for pred in all_preds if
                             pred['prediction']['text'] != '' and pred['prediction']['object']]
    preds_with_ans_not_object = [pred for pred in all_preds if
                                 pred['prediction']['text'] != '' and not pred['prediction']['object']]
    preds_with_threshold = [pred for pred in preds_with_ans_object if
                            pred['prediction']['span_score'] > span_score and pred['prediction']['null_odds'] < null_odds]

    print('{:^63}'.format('INFORMATION'))
    print('*' * 63)
    print('{:<48} {:<1}    {:<4}{:>}'.format('*  Predictions', '|', len(all_preds), '    *'))
    print('{:<48} {:<1}    {:<4}{:>}'.format('*  Prediction with answer', '|', len(preds_with_ans), '    *'))
    print('{:<48} {:<1}    {:<4}{:>}'.format('*  Prediction with answer&object', '|', len(preds_with_ans_object),
                                             '    *'))
    print('{:<48} {:<1}    {:<4}{:>}'.format('*  Prediction with answer without object', '|',
                                             len(preds_with_ans_not_object), '    *'))
    print('{:<48} {:<1}    {:<4}{:>}'.format('*  Prediction with answer and threshold set', '|',
                                             len(preds_with_threshold), '    *'))
    print('*' * 63)


    return preds_with_threshold


def collect_all_preds(folder_path):
    categories = [f for f in os.listdir(folder_path) if f.startswith('Q')]
    all_preds = []
    for category in categories:
        pred_relations = [f for f in os.listdir(os.path.join(folder_path, category)) if f.endswith(PRED_SUFFIX)]
        for pred_relation in pred_relations:
            with open(os.path.join(folder_path, category, pred_relation), 'r') as f:
                for line in f:
                    all_preds.append(json.loads(line))
    res = print_info(all_preds, args.span_score, args.null_odds)
    return res


def main():
    add_entity_to_predictions(args.folder_path)
    clean_up_dataset(args.folder_path)
    all_preds = collect_all_preds(args.folder_path)

    if len(all_preds) > 0 and all_preds:
        with open(os.path.join(args.folder_path, 'new_preds.json'), 'w') as f:
            json.dump(all_preds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str, default='./all_preds', help='directory to process')
    parser.add_argument('--span_score', type=float, default=10.0, help='span_socre')
    parser.add_argument('--null_odds', type=float, default=-9.0, help='null_odds')

    args = parser.parse_args()

    main()
