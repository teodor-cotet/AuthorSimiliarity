from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
from elasticsearch import Elasticsearch
from parse_htmls import HtmlParser
import json
json.encoder.c_make_encoder = None
import nltk
import numpy as np

show_authors = 50

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
from time import time

class ElasticS():

    def __init__(self, clean_instance=False):
        if clean_instance == False:
            self.es = self.get_elastic_instance()
        else:
            self.es = self.get_clean_elastic_instance()

    def get_clean_elastic_instance(self):
        es = Elasticsearch([{'host': Elastic.ELASTIC_HOST.value, 'port': Elastic.ELASTIC_PORT.value}])
        print('[Elasticsearch] Connected to es server on host: {} and port: {}'.format(Elastic.ELASTIC_HOST.value, Elastic.ELASTIC_PORT.value))

        if es.indices.exists(index=Elastic.ELASTIC_INDEX_AUTHORS.value) == True:
            es.indices.delete(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=[400, 404])
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_AUTHORS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            print('[Elasticsearch] index {} created'.format(Elastic.ELASTIC_INDEX_AUTHORS.value))
        
        if es.indices.exists(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value) == True:
            es.indices.delete(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=[400, 404])
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            print('[Elasticsearch] Index {} created'.format(Elastic.ELASTIC_INDEX_PUBLICATIONS.value))
        return es

    def get_elastic_instance(self):
        es = Elasticsearch([{'host': Elastic.ELASTIC_HOST.value, 'port': Elastic.ELASTIC_PORT.value}])
        print('[Elasticsearch] Connected to es server on host: {} and port: {}'.format(Elastic.ELASTIC_HOST.value, Elastic.ELASTIC_PORT.value))

        if es.indices.exists(index=Elastic.ELASTIC_INDEX_AUTHORS.value) == True:
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_AUTHORS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            print('[Elasticsearch] index {} created'.format(Elastic.ELASTIC_INDEX_AUTHORS.value))
        
        if es.indices.exists(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value) == True:
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            print('[Elasticsearch] Index {} created'.format(Elastic.ELASTIC_INDEX_PUBLICATIONS.value))
        return es

    def index_authors(self, authors_info):
        print('[Elasticsearch] Started indexing {} in {}'\
        .format(len(authors_info), Elastic.ELASTIC_INDEX_AUTHORS))

        for author in authors_info:
            self.es.index(index=Elastic.ELASTIC_INDEX_AUTHORS.value,\
                doc_type=Elastic.ELASTIC_DOC_TYPE_AUTHOR.value, body=author)
        
        print('[Elasticsearch] Finished indexing {} authors in {}'\
        .format(len(authors_info), Elastic.ELASTIC_INDEX_AUTHORS.value))
    
    def index_publications(self, pubs_info):
        print('[Elasticsearch] Started indexing {} in {}'\
        .format(len(pubs_info), Elastic.ELASTIC_INDEX_PUBLICATIONS))

        for publication in pubs_info:
            self.es.index(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value,\
                doc_type=Elastic.ELASTIC_DOC_TYPE_PUBLICATION.value, body=publication)
        
        print('[Elasticsearch] Finished indexing {} publications in {}'\
        .format(len(pubs_info), Elastic.ELASTIC_INDEX_PUBLICATIONS.value))
    

    def get_all_docs(self, size=1, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS,\
        verbose=False):
        search_query = {
            "from": 0, 
            "size": size, # max limit is 10 000, default size=10
            'size' : 10000,
            'query': {
                'match_all' : {}
            }
        }

        query_res = self.es.search(index=index_to_search.value, body=search_query)
        print('[Elasticsearch] Getting docs already indexed for {}'.format(index_to_search.value))
        if verbose == True:
            print(json.dumps(query_res, indent=True))
        return query_res

def process_text(description):
    #description_utf = description.decode('utf-8')
    all_tokens = []
    sentences = nltk.sent_tokenize(description)
    for s in sentences:
        all_tokens += nltk.word_tokenize(s)

    avg = np.float32([0] * 300)
    cnt_words = 0

    for token in all_tokens:
        if token in model_embeddings.wv.vocab:
            cnt_words += 1
            avg += np.float32(model_embeddings.wv[token])
            #avg += np.float32([1] * 300)
    return avg/cnt_words

if __name__ == "__main__":
    es = ElasticS(clean_instance=True)
    #es = ElasticS(clean_instance=False)
    parser = HtmlParser()
    authors_info, pubs_info = parser.parse('corpora/htmls')
    
    #json.dumps(authors_info)
    es.index_authors(authors_info)
    es.index_publications(pubs_info)


    # # for x in authors_info[1]:
    # #     print(x, authors_info[1][x])
    # all_docs = es.get_all_docs(size=10000, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS, verbose=False)
    # #print(len(all_docs["hits"]["hits"]))
    # all_descriptions = [ "\n".join(doc["_source"]["description"]) for doc in all_docs["hits"]["hits"]]
    # all_names = [ doc["_source"]["name"] for doc in all_docs["hits"]["hits"] ]
    
    # model_embeddings = FastTextWrapper.load_fasttext_format(fast_text)
    # datapoints = []
    # for i, description in enumerate(all_descriptions):
    #     datapoint = process_text(description)
    #     datapoints.append((datapoint, all_names[i]))
    # pickle.dump(datapoints, open("datapoints_authors", "wb" ))


    # datapoints = pickle.load(open("datapoints_authors", "rb"))

    # for d in datapoints[0:10]:
    #     print(d[0], d[1])
    #@clustering(datapoints)
    #print(len(all_descriptions))
    #print(len(all_names))
    



