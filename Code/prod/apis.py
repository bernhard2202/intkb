from SPARQLWrapper import SPARQLWrapper, JSON
import sys
from utils import get_folder_for_id, get_filename_for_article_id
import os
import json
import urllib.parse as uparse
import requests

endpoint = "http://qanswer-core1.univ-st-etienne.fr/api/endpoint/open/wikidata/sparql"
endpoint_url_link3 = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3'
endpoint_url_link3_upload = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/upload'
endpoint_url_link3_generate_features = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/generate_features'
endpoint_url_link3_train = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/train'

user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
sparql = SPARQLWrapper(endpoint, agent=user_agent)


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
        "entity": "http://www.wikidata.org/wiki/" + qid,
        "property": pid
    }

    return single_query


def fetch_full_text(title):
    url = "https://en.wikipedia.org/w/api.php"
    querystring = {"action": "query",
                   "format": "json",
                   "titles": uparse.unquote(title),
                   "prop": "extracts",
                   "explaintext": "",
                   "exlimit": "max",
                   "redirects": ""}
    try:
        response = requests.request("GET", url, params=querystring, timeout=15)
    except requests.exceptions.ReadTimeout:
        response = requests.request("GET", url, params=querystring, timeout=60)
    json_response = json.loads(response.text).get('query').get('pages')
    key = list(json_response.keys())
    return json.loads(response.text).get('query').get('pages').get(key[0]).get('extract')


def get_article(name, save_path):

    first = get_folder_for_id(name)
    if not os.path.exists(os.path.join(save_path, first)):
        try:
            os.makedirs(os.path.join(save_path, first))
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(save_path, first))
            exit(-1)
        else:
            print("Creating directory %s" % os.path.join(save_path, first))
    if os.path.exists(os.path.join(save_path, get_filename_for_article_id(name))):
        print('Skip cached file {}'.format(name))
        return 0
    text = fetch_full_text(name)
    if text is None or len(text) < 2:
        print('Error with id {}'.format(name))
        return -1
    with open(os.path.join(save_path, get_filename_for_article_id(name)), 'w') as f:
        f.write(text)
    return 1


def get_label(qid, save_path):

    if not os.path.exists(os.path.join(save_path)):
        try:
            os.mkdir(save_path)
        except OSError:
            print("Creation of the directory %s failed" % save_path)
            exit(-1)
        else:
            print("Creating directory %s" % save_path)

    property = 'http://www.wikidata.org/prop/direct/' + qid
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
    with open(os.path.join(save_path, '{}_labels.json'.format(qid)), 'w') as outfile:
        json.dump(labels, outfile)


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


def upload_4_linker_3(pairs, dataset, language='en', user='open', knowledgebase='wikidata'):
    data_2_pass = {'dataset': dataset, 'language': language, 'user': user, 'knowledgebase': knowledgebase, 'pairs': pairs}

    request = requests.post(url=endpoint_url_link3_upload, json=data_2_pass)
    result_text = request.text

    # print(f'UPLOAD LINKER3 API')
    # print(f'JSON_OF_POST_REQUEST:{data_2_pass}')
    # print('STATUS_CODE:[{}] \nMESSAGE:[{}]'.format(request.status_code, result_text))


def generate_features_link_3(dataset):
    data_2_pass = {'dataset': dataset}

    request = requests.post(url=endpoint_url_link3_generate_features, json=data_2_pass)
    result_text = request.text
    # print(f'GENERATE_FEATURES API')
    # print(f'JSON_OF_POST_REQUEST:{data_2_pass}')
    # print('STATUS_CODE:[{}] \nMESSAGE:[{}]'.format(request.status_code, result_text))


def train_linker_3(dataset, language='en', user='open', knowledgebase='wikidata'):
    data_2_pass = {'dataset': dataset, 'language': language, 'user': user, 'knowledgebase': knowledgebase}

    request = requests.post(url=endpoint_url_link3_train, json=data_2_pass)
    result_text = request.text
    if request.status_code == 200:
        print(f'TRAINED DATASET - [{dataset}] SUCCESSFULLY')
    # print(f'TRAIN LINKER3 API')
    # print(f'JSON_OF_POST_REQUEST:{data_2_pass}')
    # print('STATUS_CODE:[{}] \nMESSAGE:[{}]'.format(request.status_code, result_text))


def enhanced_linker(texts, dataset, language='en', user='open', knowledgebase='wikidata', limit=10):
    data_2_pass = {'texts': texts, 'dataset': dataset, 'language': language, 'user': user,
                   'knowledgebase': knowledgebase, 'limit': limit}

    request = requests.post(url=endpoint_url_link3, json=data_2_pass)
    result_text = request.text
    entity_list = json.loads(result_text)
    # print(f'LINKER3 API')
    # print(f'JSON_OF_POST_REQUEST:{data_2_pass}')
    # print('STATUS_CODE:[{}] \nMESSAGE:[{}]'.format(request.status_code, result_text))
    return entity_list