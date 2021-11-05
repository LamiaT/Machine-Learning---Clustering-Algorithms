"""Hierarchical Clustering for Machine Learning."""

# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering

# Importing the dataset
dataset = pd.read_csv("dataset.csv")

X = dataset.iloc[:, [3, 4]].values

# Finding optimal number of clusters using Dendrogram
dendrogram = hc.dendrogram(hc.linkage(X, method = "ward"))

plt.title("Dendrogram")
plt.xlabel("Users")
plt.ylabel("Euclidean Distances")
plt.show()

# Training the Hierarchical Clustering model
hc = AgglomerativeClustering(n_clusters = 3,
                             affinity = "euclidean",
                             linkage = "ward")
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "blue", label = "Cluster-1")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "green", label = "Cluster-2")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "magenta", label = "Cluster-3")

plt.title("Clusters")
plt.xlabel(" ")
plt.ylabel(" ")
plt.legend()
plt.show()
