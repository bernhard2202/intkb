import requests
import json


def wikipediaText(title):
    url = "https://en.wikipedia.org/w/api.php"
    querystring = {"action":"query","format":"json","titles":title,"prop":"extracts","explaintext":"","exlimit":"max"}
    response = requests.request("GET", url, params=querystring)
    jsonResponse = json.loads(response.text).get('query').get('pages')
    key = list(jsonResponse.keys())
    print(key[0])
    print(json.loads(response.text).get('query').get('pages').get(key[0]).get('extract'))


wikipediaText("Joseph_Ramey_de_Sugny")