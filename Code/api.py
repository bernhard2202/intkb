import requests
import json
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url_link1 = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link'
endpoint_url_link2 = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link2'
endpoint_url_link3 = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3'
endpoint_url_link3_upload = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/upload'
endpoint_url_link3_generate_features = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/generate_features'
endpoint_url_link3_train = 'http://qanswer-core1.univ-st-etienne.fr/api/linkDev/link3/train'
wikidata_endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"


def find_range_property(property_id):
    sparql = SPARQLWrapper(wikidata_endpoint)
    query = """
                select ?range where {
                    <http://www.wikidata.org/entity/""" + property_id + """> <http://www.wikidata.org/prop/P2302> ?b .
                    ?b <http://www.wikidata.org/prop/statement/P2302> <http://www.wikidata.org/entity/Q21510865> .
                    ?b <http://www.wikidata.org/prop/qualifier/P2308> ?range .
                }
            """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    types = []
    for result in results["results"]["bindings"]:
        types.append(result["range"]["value"])
    return types


def get_most_related_categories(property_id, limit):
    sparql = SPARQLWrapper(wikidata_endpoint)
    query = """SELECT ?type (count(?s) as ?count)  where {
            ?s wdt:""" + property_id + """ ?o .
            ?s wdt:P31 ?type .
            } group by ?type order by desc(?count) limit """ + str(limit)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]

    res_dict = dict()
    for res in results:
        res_dict[res['type']['value'].split('/')[-1]] = res['count']['value']

    return res_dict

def naive_linker(texts, language="en", user="open", knowledgebase="wikidata", limit=10):
    data_2_pass = {'texts': texts, 'language': language, 'user': user, 'knowledgebase': knowledgebase, 'limit': int(limit)}
    request = requests.post(url=endpoint_url_link1, json=data_2_pass)

    result_text = request.text
    entity_list = None
    try:
        entity_list = json.loads(result_text)
    except json.decoder.JSONDecodeError:
        print(result_text)

    return entity_list


def improved_linker(texts, restrictions, language="en", user="open", knowledgebase="wikidata", limit=10):
    data_2_pass = {'texts': texts, 'range': restrictions, 'language': language, 'user': user, 'knowledgebase': knowledgebase, 'limit': limit}

    request = requests.post(url=endpoint_url_link2, json=data_2_pass)
    result_text = request.text
    entity_list = json.loads(result_text)

    return entity_list


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


def upload_generate_train(pairs, dataset):
    upload_4_linker_3(pairs=pairs, dataset=dataset)
    generate_features_link_3(dataset=dataset)
    train_linker_3(dataset=dataset)


def main():
    texts = ['electric needle scaler', 'book burnings', 'melamine', 'melamine', 'medication']
    restrictions = find_range_property("P1479")

    # Naive linker
    # result_array = naive_linker(text, limit=5)
    # print(result_array)
    #
    # Improved linker with restrictions
    # result_array_2 = improved_linker(texts=texts, restrictions=[])
    #
    # print(result_array_2)
    # print(len(result_array_2))
    generate_features_link_3(dataset='P520')
    train_linker_3(dataset='P520')
    enhanced_linker(texts=['ak47', 'hk416'], dataset='P520')


if __name__ == '__main__':
    # main()
    print(get_most_related_categories('P178', 2))




