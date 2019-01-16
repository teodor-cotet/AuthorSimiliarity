#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import pandas as pd
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from pandas.io.json import json_normalize

def run_kmeans(X, y, estimators, fignum, titles):
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:
        ax.text3D(X[y == label, 3].mean(),
                X[y == label, 0].mean(),
                X[y == label, 2].mean() + 2, name,
                horizontalalignment='center',
                bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title('Ground Truth')
    ax.dist = 12

    fig.show()
    fig.savefig("temp.png")

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
    np.random.seed(5)

    # read from Elastic Search Database
    #authorsData = search("http://localhost:9200/index-authors/_search?pretty&size=2000","")
    authorsData = search("http://localhost:9200/index-authors/_search?pretty&size=2","")
    authorsDataAsJSON = json.loads(json.dumps(authorsData, indent=4))
    authorsDataAsJSON = authorsDataAsJSON['hits']['hits']
    
    counter = 1
    authorsDataAsDS = pd.read_json(json.dumps(authorsDataAsJSON))

    updatedAuthors = []
    headers = []    
    headers.append('id')
    headers.append('writings')
    headers.append('description')
    headers.append('quotes')
    headers.append('important_author')
    headers.append('short_description')
    headers.append('references')
    headers.append('name')

    indexes = []

    for index, row in authorsDataAsDS.iterrows():
        #print(authorSimple)
        result = []
        indexes.append('Row ' + str(counter))
        counter+=1
        result.append(row['_id'])
        result.append(row['_source']['writings'])
        result.append(row['_source']['description'])
        result.append(row['_source']['quotes'])
        result.append(row['_source']['important_author'])
        result.append(row['_source']['short_description'])
        result.append(row['_source']['references'])
        result.append(row['_source']['name'])
        updatedAuthors.append(result)

    updatedAuthorsJSON = json.dumps(updatedAuthors, indent=4)

    print(updatedAuthorsJSON)
        
    updatedDf = pd.DataFrame(data=updatedAuthors, index=indexes, columns=headers)
    
    #print(json.dumps(updatedAuthors, indent=4))
    #print(updatedDf['id'])
    #print(json.dumps(authorsDataAsDS['_source'][0], indent=4))

    iris = datasets.load_breast_cancer()
    X = iris.data
    y = iris.target
    
    estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
                ('k_means_iris_3', KMeans(n_clusters=3)),
                ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                                init='random'))]

    fignum = 1
    titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

    run_kmeans(X, y, estimators, fignum, titles)