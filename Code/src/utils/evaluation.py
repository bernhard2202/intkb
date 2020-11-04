import json

endpoint = "http://qanswer-core1.univ-st-etienne.fr/api/endpoint/wikidata/sparql"
#endpoint = "http://query.wikidata.org/bigdata/namespace/wdq/sparql"

from SPARQLWrapper import SPARQLWrapper, JSON


def knoweldgebaseAnswers(entity,predicate):
    try:
        sparql = SPARQLWrapper(endpoint)
        query = """
            SELECT ?object ?answer
            WHERE {
                {
                select ?object ?answer where {
                    <"""+entity+"""> <"""+predicate+"""> ?object .
                    ?object <http://www.w3.org/2000/01/rdf-schema#label> ?answer .
                    FILTER (lang(?answer)="en") . }
                }
                union {
                select ?object ?answer where {
                    <"""+entity+"""> <"""+predicate+"""> ?object .
                    ?object <http://www.w3.org/2004/02/skos/core#altLabel> ?answer .
                    FILTER (lang(?answer)="en") . }
                }
            }
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        answers =[]
        for result in results["results"]["bindings"]:
            answers.append(result["answer"]["value"])
        return answers
    except:
        return []


def evaluate(file, predicate):
    with open(file, "r") as predictionsFile:
        predictions = json.load(predictionsFile)
        count = 0
        countNot = 0
        for prediction in predictions:
            entity = prediction["entity"].replace("http://www.wikidata.org/entity/P","http://www.wikidata.org/entity/Q")
            span = prediction["span"]
            answers = knoweldgebaseAnswers(entity,predicate)
            found = False;
            for answer in answers:
                if answer.lower() == span.lower():
                    found = True
            if found:
                count=count+1
            else:
                countNot = countNot+1
        print("STATISTICS for property ", predicate)
        print("RIGHT ",count/(count+countNot), "%, i.e. ", count, "/", count+countNot)

evaluate("./educatedAt_predictions.json", "http://www.wikidata.org/prop/direct/P69")