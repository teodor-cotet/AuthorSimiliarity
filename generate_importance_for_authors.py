import json
import requests
import csv

def search(uri, term):    
    
    """query = json.dumps({
        "query": {
            "match": {
                "content": term
            }
        }
    })"""

    response = requests.get(uri, data="",headers={"Content-Type": "application/json"})
    results = json.loads(response.text)
    return results

if __name__ == "__main__":
    authorsData = search("http://localhost:9200/index-authors/_search?pretty&size=2000","")
    authorsDataAsString = str(json.dumps(authorsData, indent=4))
    authorsDataAsJSON = json.loads(authorsDataAsString)['hits']['hits']

    with open('word_count.csv', mode='w', newline='', encoding='utf-8') as word_count_file:
        word_count_writer = csv.writer(word_count_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        word_count_writer.writerow(['Nume autor', 'Număr cuvinte'])

        for author in authorsDataAsJSON:
            author_name = author['_source']['name'].replace("[", "").replace("]", "").replace("  ", " ")            
            word_count_writer.writerow([author_name, author['_source']['words_count']])            
            
    #print(authorsDataAsJSON)