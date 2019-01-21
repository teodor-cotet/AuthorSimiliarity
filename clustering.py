import numpy as np
import pickle
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
from time import time
model_embeddings = None
fast_text = "fastText/wiki.ro"
show_authors = 50

def clustering(datapoints):
    
    data = [d[0] for d in datapoints]
    estimator = KMeans(init='random', n_clusters=6, n_init=6)
    estimator.fit(data)
    #print(estimator)
    # for i in range(len(datapoints)):
    #     print(estimator.labels_[i], datapoints[i][1])

    
    reduced_data = PCA(n_components=2).fit_transform(data)
    # k-means++, 
    kmeans = KMeans(init='k-means++', n_clusters=6, n_init=6)
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
    datapoints = pickle.load(open("datapoints_authors", "rb"))
    #print(datapoints[0][0])

    with open("authors.model", "w") as f:
        f.write(f"{len(datapoints)} {300}\n")
        for d in datapoints:
            concat_name = ";".join(d[1].split(' '))
            f.write(f"{concat_name} ")
            for i, v in enumerate(d[0]):
                if i == len(d[0]) - 1:
                    f.write(f"{v}")
                else:
                    f.write(f"{v} ")
            f.write("\n")

    
    #for d in datapoints[0:10]:
    #    print(d[0], d[1])