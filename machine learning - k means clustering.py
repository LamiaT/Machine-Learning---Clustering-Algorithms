"""K-Means Clustering for Machine Learning."""

# Importing the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv("dataset.csv")

X = dataset.iloc[:, [3, 4]].values

# Finding the optimal number of clusters using Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i,
                    init = "k-means++",
                    random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Training the K-Means model on the dataset
k_means = KMeans(n_clusters = 3,
                 init = "k-means++",
                 random_state = 42)

y_kmeans = k_means.fit_predict(X)

# Visualising Clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "blue", label = "Cluster-1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "green", label = "Cluster-2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "magenta", label = "Cluster-3")

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s = 300,
            c = "cyan",
            label = "Centroids")

plt.title("Clusters")
plt.xlabel(" ")
plt.ylabel(" ")
plt.legend()
plt.show()
