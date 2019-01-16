from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
from elasticsearch import Elasticsearch
from parse_htmls import HtmlParser
import json
import nltk
from gensim.models.wrappers import FastText as FastTextWrapper
import numpy as np
import pickle

model_embeddings = None
fast_text = "fastText/wiki.ro"
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

def clustering(datapoints):
    
    data = [d[0] for d in datapoints]
    estimator = KMeans(init='random', n_clusters=10, n_init=10)
    estimator.fit(data)
    #print(estimator)
    # for i in range(len(datapoints)):
    #     print(estimator.labels_[i], datapoints[i][1])

    
    reduced_data = PCA(n_components=2).fit_transform(data)
    # k-means++, 
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(reduced_data)

    print('silhouette: {}'.format(metrics.silhouette_score(data, estimator.labels_,\
                                      metric='euclidean',\
                                      sample_size=200)))
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .002     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.0001, reduced_data[:, 0].max() + 0.0001
    y_min, y_max = reduced_data[:, 1].min() - 0.0001, reduced_data[:, 1].max() + 0.0001
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(25, 25))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=1)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],\
                marker='x', s=169, linewidths=3,\
                color='w', zorder=10)

    # for i in range(30):
    #     plt.scatter(reduced_data[:, 0], reduced_data[:, 1],\
    #                 marker="$datapoints[i][1]$", s=169, linewidths=3,\
    #                 color='w', zorder=10)
    plt.title('K-means clustering on authors')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    fig, ax = plt.subplots()
    # annotate is a method that belongs to axes
    x = reduced_data[:, 0]
    y = reduced_data[:, 1]
    ax.plot(x, y, 'ro',markersize=1)

    ## controls the extent of the plot.
    ax.set_xlim(min(x)-0.001, max(x)+ 0.001)
    ax.set_ylim(min(y)-0.001, max(y)+ 0.001)
    ind = 0
    for i,j in zip(x,y):
        ax.annotate(str(datapoints[ind][1]),  xy=(i, j))
        ind += 1
        if ind == show_authors:
            break

    plt.show()
    return estimator
    #len(estimator.labels_)



if __name__ == "__main__":
    # es = ElasticS(clean_instance=False)
    # # parser = HtmlParser()
    # # authors_info, pubs_info = parser.parse('corpora/htmls')
    # # es.index_authors(authors_info)
    # # es.index_publications(pubs_info)


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


    datapoints = pickle.load(open("datapoints_authors", "rb"))

    # for d in datapoints[0:10]:
    #     print(d[0], d[1])
    clustering(datapoints)
    #print(len(all_descriptions))
    #print(len(all_names))
    



