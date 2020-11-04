import json
import requests
import os
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint = "http://localhost:4567/api/endpoint/wikidata/sparql"
wikidata_endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"




def find_range_property(property):
    # time.sleep(50)
    sparql = SPARQLWrapper(wikidata_endpoint)
    query = """
                select ?range where {
                    <http://www.wikidata.org/entity/""" + property + """> <http://www.wikidata.org/prop/P2302> ?b .
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


def extract_range_properties():
    for directory in ["known", "zero_shot"]:
        for filename in os.listdir("../../out/predictions/" + directory + "zero_shot""/"):
            if not filename.__contains__("_range"):
                found = False
                for filename2 in os.listdir("../../out/predictions/" + directory + "zero_shot""/"):
                    if filename2 == filename + "_range":
                        found = True
                if found == False:
                    print(filename)
                    types = find_range_property(filename.replace("-predictions.json", ""))
                    print(types)
                    f = open("../../out/predictions/" + directory + "/" + filename + '_range', 'a')
                    for t in types:
                        f.write(t + "\n")
                    f.close()


def check_right_type(entity, type):
    sparql = SPARQLWrapper(endpoint)
    query = """
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    ASK where
    {
    <""" + entity + """> wdt:P31 <""" + type + """>
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results['boolean'] == True:
        return results['boolean']
    sparql = SPARQLWrapper(endpoint)
    query = """
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        ASK where
        {
        <""" + entity + """> wdt:P31/wdt:P279 <""" + type + """>
        }
        """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results['boolean']


def evaluate():
    total = 0
    correct = 0
    correct_at_2 = 0
    correct_at_5 = 0
    correct_at_all = 0
    false = 0
    zero = 0
    ones = 0
    for filename in os.listdir("../../out/predictions/zero_shot/"):
        # linking
        if not filename.__contains__("_range"):
            print(filename)
            with open("../../out/predictions/zero_shot/" + filename, "r") as predictionsFile:
                for line in predictionsFile:

                    total += 1
                    results = json.loads(line)
                    url = "http://localhost:4567/api/link"
                    querystring = {"text": results['prediction']["text"],
                                   "language": "en",
                                   "knowledgebase": "wikidata"}

                    response = requests.request("GET", url, params=querystring)
                    json_response = json.loads(response.text)
                    if len(json_response) == 0:
                        zero += 1
                    else:
                        if len(json_response) == 1:
                            ones += 1

                        count = 0
                        for res in json_response:
                            # rigth_type=False
                            # if len(open("../../out/predictions/zero_shot/"+filename+'_range').readlines())==0:
                            #     rigth_type=True
                            # else:
                            #     #print("../../out/predictions/known/"+filename+'_range')
                            #     with open("../../out/predictions/zero_shot/"+filename+'_range', "r") as types:
                            #         for type in types:
                            #             if rigth_type == False:
                            #                 rigth_type = check_right_type(res,type.replace("\n",""))
                            #
                            # if rigth_type == True:
                            found = False
                            for answer in results['query']['answer_entity']:
                                if res == answer:
                                    correct_at_all += 1
                                    if found == false:
                                        if count < 1:
                                            correct += 1

                                        if count < 3:
                                            correct_at_2 += 1

                                        if count < 5:
                                            correct_at_5 += 1
                                    found = True
                                count += 1
                    response.close()
    print("Total ", total)
    print("zero ", zero)
    print("ones", ones)
    print("correct", correct)
    print("correct_at_2", correct_at_2)
    print("correct_at_5", correct_at_5)
    print("correct_at_all", correct_at_all)
