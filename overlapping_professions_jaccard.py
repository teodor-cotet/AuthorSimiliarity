import json
import requests
import csv
import numpy as np
from sklearn.metrics import jaccard_similarity_score

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

    authorProfessions = dict()
    # An entry in authorProfessions contains the following
    # ['etnolog',
    # 'critic', 
    # 'autor dramatic | autoare | prozator | prozatoare | eseist', 
    # 'folclorist | culegător de folclor', 
    # 'traducător | traducătoare', 
    # 'poet | versificator | autor de versuri', 
    # 'dramaturg', 
    # 'slavist', 
    # 'jurnalist | gazetar | cronicar | publicist | ziarist', 
    # 'românist german', 
    # 'comparatist',
    # 'editor', 
    # 'memorialist | diarist', 
    # 'clasicist',
    # 'stilistician', 
    # 'filolog', 
    # 'cărturar', 
    # 'anglist', 
    # 'imnograf',
    # 'istoric', 
    # 'estetician', 
    # 'teatrolog | critic de teatru', 
    # 'teoretician', 
    # 'povestitor',
    # 'eminescolog']
    # Array of 25 elements with 0/1

    for author in authorsDataAsJSON:
        author_name = author['_source']['name'].replace("[", "").replace("]", "").replace("  ", " ")
        professions = author['_source']['professions']

        authorProfessions[author_name] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        for profession in professions:
            #print(profession)
            if profession == "etnolog":
                authorProfessions[author_name][0] = 1
            if 'critic' in profession:
                authorProfessions[author_name][1] = 1
            if profession == 'autor dramatic' or profession == 'autoare' or profession == 'prozator' or profession == 'prozatoare' or 'eseist' in profession:
                authorProfessions[author_name][2] = 1
            if 'folclorist' in profession or profession == 'culegător de folclor':
                authorProfessions[author_name][3] = 1
            if profession == 'traducător' or profession == 'traducătoare':
                authorProfessions[author_name][4] = 1
            if 'poet' in profession or profession == 'versificator' or profession == 'versificatoare' or profession == 'autor de versuri':
                authorProfessions[author_name][5] = 1
            if 'dramaturg' in profession:
                authorProfessions[author_name][6] = 1
            if 'slavist' in profession:
                authorProfessions[author_name][7] = 1
            if 'jurnalist' in profession or 'gazetar' in profession or 'cronicar' in profession or 'publicist' in profession or 'ziarist' in profession:
                authorProfessions[author_name][8] = 1
            if profession == "românist german":
                authorProfessions[author_name][9] = 1
            if 'comparatist' in profession:
                authorProfessions[author_name][10] = 1
            if profession == "editor":
                authorProfessions[author_name][11] = 1
            if 'memorialist' in profession or 'diarist' in profession:
                authorProfessions[author_name][12] = 1
            if 'clasicist' in profession:
                authorProfessions[author_name][13] = 1
            if 'stilistician' in profession:
                authorProfessions[author_name][14] = 1
            if 'filolog' in profession:
                authorProfessions[author_name][15] = 1
            if 'cărturar' in profession:
                authorProfessions[author_name][16] = 1
            if 'anglist' in profession:
                authorProfessions[author_name][17] = 1
            if 'imnograf' in profession:
                authorProfessions[author_name][18] = 1
            if 'istoric' in profession:
                authorProfessions[author_name][19] = 1
            if 'estetician' in profession:
                authorProfessions[author_name][20] = 1
            if 'teatrolog' in profession or 'critic de teatru' in profession:
                authorProfessions[author_name][21] = 1
            if 'teoretician' in profession:
                authorProfessions[author_name][22] = 1
            if 'povestitor' in profession:
                authorProfessions[author_name][23] = 1
            if 'eminescolog' in profession:
                authorProfessions[author_name][24] = 1

    with open('jaccard_overlapping_professions.csv', mode='w', newline='', encoding='utf-8') as word_count_file:
        word_count_writer = csv.writer(word_count_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)       
        elemNr=0
        for key, value in authorProfessions.items():    
            elemNr += 1
            jaccard_values = [key]
            for key2, value2 in authorProfessions.items():
                jaccard_values.append(jaccard_similarity_score(value, value2))
            word_count_writer.writerow(jaccard_values)
            print(elemNr)
        #print(authorsDataAsJSON)