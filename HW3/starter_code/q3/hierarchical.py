"""
EECS 445 - Introduction to Machine Learning
Fall 2022 - Homework 3
Hierarchical Clustering
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


def plot_hierarchical_dendrogram(buildings, ETAs):
    '''
    plot the dendrogram based on hierarchical clustering
    '''
    # TODO: Define a scikit-learn AgglomerativeClustering object
    # Set argument n_clusters(number of clusters) as None, affinity as precomputed, linkage as complete, distance_threshold as 0
    agglomerative = AgglomerativeClustering(
        n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=0)

    # Fit data to obtain clusters
    agglomerative.fit(ETAs)

    # Build the linkage matrix that represents the hierarchical cluster structure
    linkage_matrix = np.column_stack(
        [agglomerative.children_, agglomerative.distances_, np.ones_like(agglomerative.distances_)])

    # TODO: Plot dendrogram using a scipy function dendrogram (scipy.cluster.hierarchy.dendrogram imported above)
    # Set argument labels as buildings
    dendrogram(linkage_matrix, labels=buildings)

    plt.savefig('hierarchical_dendrogram.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    buildings = ["Angel\nHall", "Mason\nHall",
                 "GGBL", "BBB", "Chrysler\nCenter", "FXB"]
    ETAs = np.loadtxt('campus.csv')
    print("Visualize Hierarchical clustering through dendrogram")
    plot_hierarchical_dendrogram(buildings, ETAs)
