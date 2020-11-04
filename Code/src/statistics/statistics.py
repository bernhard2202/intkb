import json
from http.client import RemoteDisconnected
from random import shuffle
from SPARQLWrapper import SPARQLWrapper, JSON
import time

#endpoint = "http://localhost:4567/api/endpoint/wikidata/sparql"
endpoint = "http://qanswer-core1.univ-st-etienne.fr/api/endpoint/wikidata/sparql"
#endpoint = "https://query.wikidata.org/sparql"

sparql = SPARQLWrapper(endpoint)

def selection_properties():
    with open("../../data/statistics/statistics_wikipedia.json", "r") as file:
            response = json.load(file)
            # results = QueryResult((response,_SPARQL_JSON))
            # print(results.print_results())
            # answers =[]
            p = []
            count = []
            for result in response["results"]["bindings"]:
                if "http://www.wikidata.org/prop/direct/P" in result["p"]["value"]:
                    #restricting to properties of type WikibaseItem
                    query = """
                        PREFIX wikibase: <http://wikiba.se/ontology#>
                        SELECT * where {
                            ?s wikibase:directClaim <"""+result["p"]["value"]+"""> .
                            ?s wikibase:propertyType wikibase:WikibaseItem .
                        }"""
                    #print(query)
                    sparql.setQuery(query)
                    sparql.setReturnFormat(JSON)
                    results = sparql.query().convert()
                    print(len(results["results"]["bindings"]))
                    if (len(results["results"]["bindings"])>=1):
                        p.append(result["p"]["value"])
                        count.append(int(result["sum"]["value"]))
            return list(zip(count,p))


# this function generates the tikzpicture data to draw the distribution of the properties
def print_latex_plot():
    with open("../../data/statistics/plot.data", "w") as plot:
        l = selection_properties()
        c = 0
        average = 0
        min = 10000
        max = 0
        for count, p in sorted(l, reverse=True):
            print(p,count)
            plot.write(" "+str(c)+"  "+str(count)+" \\\\\n")
            c = c+1
            average = average + count
            if count<min:
                min = count
            if count > max:
                max = count
        print("Total number",c)
        print("Average", average/c)
        print("Min", min)
        print("Max", max)
    plot.close()


# this function samples from the property distribution in wikidata
def property_distribution():
    with open("../../data/statistics/sample.txt", "w") as sample:
        l = selection_properties()
        shuffle(l)
        for count, p in l:
            sample.write(p + "\n")
    print("Done")
    sample.close()


# this function searches for the rdfs:label of a property
def label_property(property):
    query = """
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

# this function searches for the rdfs:label of a property
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

# this function searches all the lexicalizations of an entity
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


# this function chooses for a property in wikidata some samples
def extracts_training_samples(property):
    samples = []
    question = label_property(property)
    query = """
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            select ?entity ?wikipedia_link
                where {{
                    ?entity <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q7889> .
                    ?entity <{property}> ?answer_entity .
                    ?wikipedia_link schema:about ?entity ;
                    schema:isPartOf <https://en.wikipedia.org/>.
                    }} limit 10000
            """.format(property=property)
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
    with open('../../data/relations/'+property.replace("http://www.wikidata.org/prop/direct/","")+'.json', 'w') as outfile:
        json.dump(samples, outfile)
    outfile.close()

def extracts_prediction_samples(property):
    samples = []
    question = label_property(property)
    query = """
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            select ?entity ?wikipedia_link
                where {{
                    ?entity <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q7889> .
                    FILTER NOT EXISTS {?entity <""" + property + """> ?answer_entity } .
                    ?wikipedia_link schema:about ?entity ;
                    schema:isPartOf <https://en.wikipedia.org/>.
                    }} limit 10
            """
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

            entity_label = label(entity)
            answer = []

            if (entity_label!= None and answer!=None and question!=None):
                sample = {
                    'entity_label':  entity_label,
                    'question': question + ' '+entity_label,
                    'wikipedia_link': wikipedia_link
                }
                samples.append(sample)

        count = count +1
        #print(count,"/",next)
    with open('../../data/predictions/'+property.replace("http://www.wikidata.org/prop/direct/","")+'.json', 'w') as outfile:
        json.dump(samples, outfile)
    outfile.close()


# this function searches for the rdfs:label of a property
def all_label_property(property):
    #time.sleep(1)
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
    with open('../../data/labels/' + property.replace("http://www.wikidata.org/prop/direct/", "") + '_labels.json', 'w') as outfile:
        json.dump(labels, outfile)
    outfile.close()
    return labels

# with open('../../data/statistics/sample2.txt') as f:
#     content = f.readlines()
#     for x in content:
#         print(x.rstrip())
#         #generate_samples(x.rstrip())
#         all_label_property(x.rstrip())


#print_latex_plot()
#property_distribution()

# extracts_training_samples("http://www.wikidata.org/prop/direct/P178")
properties_selected = ['P123', 'P136', 'P178', 'P495', 'P750']

# Extract all labels of the properties
for property in properties_selected:
    # extracts_training_samples('http://www.wikidata.org/prop/direct/' + property)
    #all_label_property('http://www.wikidata.org/prop/direct/'+property)
    extracts_prediction_samples('http://www.wikidata.org/prop/direct/' + property)