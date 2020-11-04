import json
from SPARQLWrapper import SPARQLWrapper, JSON
import os
from filter_before_rankerNet import formulate_params
from src.retriever.utils import get_filename_for_article_id
import sys
import time
import string
from api import enhanced_linker

endpoint = "http://qanswer-core1.univ-st-etienne.fr/api/endpoint/open/wikidata/sparql"

user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
sparql = SPARQLWrapper(endpoint, agent=user_agent)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def label_property(property):
    query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        select ?label
            where {{
                ?property wikibase:directClaim <{property}> .
                ?property rdfs:label ?label.
                filter(lang(?label)='en').
            }} limit 3000
                """.format(property=property)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        return result["label"]["value"]
    return None

def label(entity):
    query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        select ?label
            where {{
                <{entity}> rdfs:label ?label.
                filter(lang(?label)='en').
            }} limit 3000
                """.format(entity=entity)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        return result["label"]["value"]
    return None


def extracts_prediction_samples(entityId, property, amount, output_dir):
    samples = []
    question = label_property(property)
    query = """
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            select ?entity ?wikipedia_link
                where {{
                    ?entity <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/"""+ entityId +"""> .
                    FILTER NOT EXISTS {?entity <""" + property + """> ?answer_entity } .
                    ?wikipedia_link schema:about ?entity ;
                    schema:isPartOf <https://en.wikipedia.org/>.
                    }} limit """ + str(amount)
    print(query)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    resultsAll = sparql.query().convert()

    count = 0
    step = int(len(resultsAll["results"]["bindings"])/3000)
    if (step==0):
        step = 1
    next = 0
    for result in resultsAll["results"]["bindings"]:
        if count==next:
            next=count+step
            print(count,"/",len(resultsAll["results"]["bindings"]))
            entity = result["entity"]["value"]
            wikipedia_link = result["wikipedia_link"]["value"]

            query_2 = """
            PREFIX schema: <http://schema.org/>
            select * where {
             <""" + wikipedia_link +"""> schema:about ?o
            }"""
            sparql.setQuery(query_2)
            sparql.setReturnFormat(JSON)
            resultsAll_2 = sparql.query().convert()
            wikidata_uri = resultsAll_2['results']['bindings'][0]['o']['value']

            entity_label = label(entity)
            answer = []

            if (entity_label!= None and answer!=None and question!=None):
                sample = {
                    'entity_label':  entity_label,
                    'question': question + ' '+entity_label,
                    'wikipedia_link': wikipedia_link,
                    'answer': answer,
                    'entity': wikidata_uri
                }
                samples.append(sample)

        count = count + 1
    with open(os.path.join(output_dir, 'data/{}_test.json'.format(property.replace("http://www.wikidata.org/prop/direct/",""))), 'w') as outfile:
        json.dump(samples, outfile)
    outfile.close()


def extracts_training_samples(property, output_dir, entityId):
    samples = []
    question = label_property(property)
    query = """
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            select ?entity ?wikipedia_link
                where {{
                    ?entity <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/{entityId}> .
                    ?entity <{property}> ?answer_entity .
                    ?wikipedia_link schema:about ?entity ;
                    schema:isPartOf <https://en.wikipedia.org/>.
                    }} limit 1000
            """.format(entityId=entityId, property=property)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    resultsAll = sparql.query().convert()

    count = 0
    step = int(len(resultsAll["results"]["bindings"])/3000)
    if (step==0):
        step = 1
    next = 0
    for result in resultsAll["results"]["bindings"]:
        if count==next:
            next=count+step
            print(count,"/",len(resultsAll["results"]["bindings"]))
            entity = result["entity"]["value"]
            wikipedia_link = result["wikipedia_link"]["value"]
            #answer_entity = result["answer_entity"]["value"]
            query = """
                    PREFIX schema: <http://schema.org/>
                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                    select ?answer_entity
                        where {{
                            # s needs to have a wikipedia link
                            <{entity}> <{property}> ?answer_entity .
                            }}
                    """.format(entity=entity, property=property)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            answers_entities = []
            if len(results["results"]["bindings"])>0:
                for result in results["results"]["bindings"]:
                    answer_entity = result["answer_entity"]["value"]
                    answers_entities.append(answer_entity)
            entity_label = label(entity)
            answer = []
            for answer_entity in answers_entities:
                answer = answer + all_labels(answer_entity)
            if (entity_label!= None and answer!=None and question!=None):
                sample = {
                    'entity_label':  entity_label,
                    'answer': answer,
                    'answer_entity': answers_entities,
                    'question': question + ' '+entity_label,
                    'wikipedia_link': wikipedia_link
                }
                samples.append(sample)

        count = count +1
        #print(count,"/",next)
    with open(os.path.join(output_dir, 'data/relations/{}.json').format(property.replace("http://www.wikidata.org/prop/direct/","")), 'w') as outfile:
        json.dump(samples, outfile)
    outfile.close()


def all_labels(entity):
    query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        select ?label
            where {{
            {{
                <{entity}> rdfs:label ?label.
                filter(lang(?label)='en').
            }} UNION {{
                <{entity}> skos:altLabel ?label .
                filter(lang(?label)='en').
            }}
            }} limit 3000
                """.format(entity=entity)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    labels = []
    for result in results["results"]["bindings"]:
        labels.append(result["label"]["value"])
    return labels


def all_label_property(property):
    labels =[];
    query = """
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        select ?label
            where {{
            {{
                ?property wikibase:directClaim <{property}> .
                ?property rdfs:label ?label .
                filter(lang(?label)='en').
            }} UNION {{
                ?property wikibase:directClaim <{property}> .
                ?property skos:altLabel ?label .
                filter(lang(?label)='en').
            }}
            }} limit 3000
                """.format(property=property)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        labels.append(result["label"]["value"])
    with open('./data/labels/' + property.replace("http://www.wikidata.org/prop/direct/", "") + '_labels.json', 'w') as outfile:
        json.dump(labels, outfile)
    outfile.close()
    return labels


def get_relations_by_entityID(entityID, k=10):
    new_sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    if(entityID[0] == 'Q'):
        query = """select ?p ?c where {
                        {
                            select ?p (count(?s) as ?c ) where 
                            {
                              ?s  wdt:P31 <http://www.wikidata.org/entity/""" + entityID +"""> .
                              ?s ?p ?o .
                            } group by ?p limit 100
                        }
                        ?s2 wikibase:directClaim ?p .
                        ?s2 wikibase:propertyType wikibase:WikibaseItem
                        } order by desc(?c)"""
        new_sparql.setQuery(query)
        new_sparql.setReturnFormat(JSON)
        results = new_sparql.query().convert()
        top_k_relations = []
        for result in results["results"]["bindings"][0:k]:
            # print(result['p']['value'], end=' ')
            # print(result['c']['value'])
            top_k_relations.append([result['p']['value'], result['c']['value']])

        return top_k_relations
    else:
        raise ValueError('Error: Wrong entityID')


def print_stats_validation_get():
    num_preds = 0
    num_noDoc = 0
    num_good = 0
    relations = [f.split('_neg')[0] for f in os.listdir('./out/features/trainonknown/zero_shot') if
                 f.endswith('neg_kw_sent_meta_results.json') and f.startswith('P')]
    for relation in relations:
        print('Processing the relation {} [evaluation_neg]'.format(relation))
        with open(os.path.join('./out/features/trainonknown/zero_shot', '{}_neg_kw_sent_meta_results.json'.format(relation)), 'r') as f:
            for line in f:
                result = json.loads(line)
                num_preds += result['N']
                num_noDoc += result['no_doc_found']

        with open(os.path.join('./out/features/trainonknown/zero_shot', '{}-neg-kw_sent-feat-batch-0.txt'.format(relation)), 'r') as f:
            for line in f:
                result = json.loads(line)
                if result[0]['null_odds'] == 1000:
                    num_good += 1
        print(f'{bcolors.OKGREEN}====================== RESULT EVALUATION ======================{bcolors.ENDC}')
        print ({'N': num_preds, 'NO_DOC': num_noDoc, 'NEG_PRECISION': formulate_params(num_good/(num_preds-num_noDoc))})
        print(f'{bcolors.OKGREEN}============================= END ============================={bcolors.ENDC}')
        input('[Enter] to continue')

# reference: src/ranker/data_utils.py
# function load_full_zs()
def load_full_zs(relations, feat_path, data_path, heads, tails):

    complete_data = []
    is_rare = []
    seen_head = []
    seen_tail = []

    for i, relation in enumerate(relations):
        file = os.path.join(feat_path, '{}-test-kw_sent-feat-batch-0.txt'.format(relation))
        if not os.path.exists(file):
            print("skipping non existing file {}".format(file))
            continue
        with open(file, 'r') as f:
            i = 0
            for line in f:
                samples = json.loads(line)
                complete_data.append(samples)
                i += 1
            for _ in range(i):
                is_rare.append(True if i < 10 else False)
        with open(os.path.join(data_path, '{}_test.json'.format(relation)), 'r') as f:
            original_query = json.load(f)
            for query in original_query:
                fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
                if not os.path.exists(os.path.join('data', 'wiki', fname)):
                    continue
                seen_head.append(True if query['entity_label'] in heads else False)
                tail = False
                for t in query['answer']:
                    if t in tails:
                        tail = True
                        break
                seen_tail.append(tail)

    print(len(complete_data))
    print(len(is_rare))
    print(len(seen_tail))

    assert len(complete_data) == len(is_rare) == len(seen_tail) == len(seen_head)
    return complete_data, is_rare, seen_head, seen_tail

def load_full_zs_neg(relations, feat_path, data_path, heads, tails):

    complete_data = []
    is_rare = []
    seen_head = []
    seen_tail = []

    for i, relation in enumerate(relations):
        file = os.path.join(feat_path, '{}-neg-kw_sent-feat-batch-0.txt'.format(relation))
        if not os.path.exists(file):
            print("skipping non existing file {}".format(file))
            continue
        with open(file, 'r') as f:
            i = 0
            for line in f:
                samples = json.loads(line)
                complete_data.append(samples)
                i += 1
            for _ in range(i):
                is_rare.append(True if i < 10 else False)
        with open(os.path.join(data_path, '{}_neg.json'.format(relation)), 'r') as f:
            original_query = json.load(f)
            for query in original_query:
                fname = get_filename_for_article_id(query['wikipedia_link'].split('wiki/')[-1])
                if not os.path.exists(os.path.join('data', 'wiki', fname)):
                    continue
                seen_head.append(True if query['entity_label'] in heads else False)
                tail = False
                # for t in query['answer']:
                #     if t in tails:
                #         tail = True
                #         break
                seen_tail.append(tail)

    print(len(complete_data))
    print(len(is_rare))
    print(len(seen_tail))

    assert len(complete_data) == len(is_rare) == len(seen_tail) == len(seen_head)
    return complete_data, is_rare, seen_head, seen_tail


def construct_query(qid, pid):
    sparql.setReturnFormat(JSON)
    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    select  ?wikipedia_link ?label
    where{
      ?wikipedia_link schema:about wd:""" + qid + """;
      schema:isPartOf <https://en.wikipedia.org/> .
      wd:""" + qid + """ rdfs:label ?label.
      filter(lang(?label)='en').
    } limit 1"""

    sparql.setQuery(query)
    results = sparql.query().convert()
    entity_label = results["results"]["bindings"][0]["label"]["value"]
    wikipedia_link = results["results"]["bindings"][0]['wikipedia_link']['value']
    question = label_property('http://www.wikidata.org/prop/direct/' + pid) + ' ' + entity_label
    single_query = {
        "entity_label": entity_label,
        "question": question,
        "wikipedia_link": wikipedia_link,
        "answer": [],
        "entity": "http://www.wikidata.org/wiki/" + qid
    }

    return single_query


def make_predictions(model_path, data_path, feature_path):

    vocab_path = os.path.join(model_path, 'vocab.txt')
    config_path = os.path.join(model_path, 'bert_config.json')
    tmp_path = str(time.time()).split('.')[0]
    command = 'python src/relation_extraction.py ' \
              '--feat_path={} ' \
              '--split=prod ' \
              '--wiki_data=./data/wiki ' \
              '--vocab_file={} ' \
              '--bert_config_file={} ' \
              '--init_checkpoint={} ' \
              '--output_dir=/tmp/{} ' \
              '--do_predict=True ' \
              '--do_train=False ' \
              '--predict_file=./ ' \
              '--k_sentences=20 ' \
              '--predict_batch_size=32 ' \
              '--num-kw-queries=5 ' \
              '--out_name=kw_sent ' \
              '--version_2_with_negative=True ' \
              '--null_score_diff_threshold={} ' \
              '--data_path={}'
    # print(command.format(feature_path, vocab_path, config_path, model_path, tmp_path, str(args.threshold), data_path))
    os.system(command.format(feature_path, vocab_path, config_path, model_path, tmp_path, 0.0, data_path))


def tokens_to_sentence(tokens):
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def call_ranker_net():
    command_rankerNet = 'python src/rankerNet.py --data_path={} --feature_path={} --output_path={}'
    os.system(command_rankerNet.format('/tmp/data', '/tmp/features/', '/tmp/predictions'))


def call_enhanced_linker(pid):
    if os.path.exists('/tmp/predictions/{}-predictions.json'.format(pid)):
        with open('/tmp/predictions/{}-predictions.json'.format(pid), 'r') as f:
            for line in f:
                sample = json.loads(line)
                break
        if not(sample['prediction']['null_odds'] == 1000 or not sample['prediction']['text']):

            question_, wikipedia_link_, entity_ = sample['query']['question'], sample['query'][
                'wikipedia_link'], sample['query']['entity']
            sample['query'] = {}
            sample['query']['category'] = 'Q_TODO'
            sample['query']['property'] = pid
            sample['query']['question'], sample['query']['wikipedia_link'], sample['query'][
                'entity'] = question_, wikipedia_link_, entity_

            text_, score_, null_odds_ = sample['prediction']['text'], sample['prediction']['span_score'], \
                                        sample['prediction']['null_odds']
            evidence_ = tokens_to_sentence(sample['prediction']['doc_tokens'])
            sample['prediction'] = {}
            sample['prediction']['text'], sample['prediction']['span_score'], sample['prediction'][
                'null_odds'] = text_, score_, null_odds_
            sample['prediction']['evidence'] = evidence_

            objects = enhanced_linker(texts=[text_], dataset=str(pid))
            if len(objects) == 1:
                sample['prediction']['object'] = objects[0]
            else:
                sample['prediction']['object'] = []

        return sample


def get_article():
    command = 'python src/download_wikipedia.py /tmp/data ./data/wiki'
    os.system(command)


def get_labels_data(relations):
    for relation in relations:
        if os.path.exists('data/labels/{}_labels.json'.format(relation)):
            continue
        property = 'http://www.wikidata.org/prop/direct/' + relation
        labels =[];
        query = """
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            select ?label
                where {{
                {{
                    ?property wikibase:directClaim <{property}> .
                    ?property rdfs:label ?label .
                    filter(lang(?label)='en').
                }} UNION {{
                    ?property wikibase:directClaim <{property}> .
                    ?property skos:altLabel ?label .
                    filter(lang(?label)='en').
                }}
                }} limit 3000
                    """.format(property=property)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            labels.append(result["label"]["value"])
        with open('data/labels/' + property.replace("http://www.wikidata.org/prop/direct/", "") + '_labels.json', 'w') as outfile:
            json.dump(labels, outfile)


if __name__ == '__main__':
    # top_k_relations = get_relations_by_entityID('Q11032', 5)
    # print(top_k_relations[0][0]
    print_stats_validation_get()
