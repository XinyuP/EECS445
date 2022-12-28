"""
EECS 445 - Introduction to Machine Learning
Fall 2022 - Homework 3
Spectral Clustering
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


def plot_spectral_num_cluster(data, gamma=1):
    '''
    Plot the 10 smallest eigenvalues of the Laplacian matrix
    to determine the number of clusters
    '''
    # TODO: Define a scikit-learn SpectralClustering object
    # - Set argument gamma (rbf parameter) as gamma
    spectral = SpectralClustering(gamma=gamma)

    # Fit data to obtain affinity matrix
    spectral.fit(data)

    # TODO: Compute the Laplacian matrix of the affinity matrix (affinity_matrix_ attribute in SpectralClustering)
    # using a scipy function laplacian (scipy.sparse.csgraph.laplacian imported above)
    # - Set normed=True
    lap_matrix = laplacian(spectral.affinity_matrix_, normed=True)

    # TODO: Compute the eigenvalues of the Laplacian matrix (lap_matrix) using a scipy function eigh (scipy.linalg.eigh imported above)
    eigenvalues, _ = eigh(lap_matrix)

    # Pick the 10 smallest eigenvalues and plot
    ks = np.arange(1, 11)
    eigenvalues = eigenvalues[:10]
    plt.plot(ks, eigenvalues, 'x')
    plt.title("Spectral Clustering Eigenvalues")
    plt.xlabel("k")
    plt.ylabel("k-th smallest eigenvalue")
    plt.savefig('spectral_num_cluster.png', dpi=200, bbox_inches='tight')


def visualize_spectral(data, gammas, n_clusters=3):
    '''
    Visualize the result of spectral clustering
    '''
    # Build figure and set size
    f, axarr = plt.subplots(1, len(gammas))
    f.set_figheight(5)
    f.set_figwidth(15)

    # for each gamma(rbf parameter)
    for i, gamma in enumerate(gammas):
        # TODO: Define a scikit-learn SpectralClustering object
        # - Set argument n_clusters (number of clusters) as n_clusters and gamma (rbf parameter) as gamma
        spectral = SpectralClustering(n_clusters=n_clusters, gamma=gamma)

        # Fit data to obtain clusters
        spectral.fit(data)

        # for each cluster
        for cluster in np.arange(n_clusters):
            # Plot the data points for the cluster
            axarr[i].plot(data[spectral.labels_ == cluster, 0],
                          data[spectral.labels_ == cluster, 1], 'x')
        axarr[i].axis('equal')
        axarr[i].set_title("gamma:{0}".format(gamma))
    f.suptitle("Spectral Clustering Visualization")
    f.savefig('spectral_visualization.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    data = np.loadtxt('mickey.csv')
    print("Plotting spectral clustering performance as the number of clusters varies ==================")
    plot_spectral_num_cluster(data, gamma=1)
    print("Visualizing spectral clustering assignments for different values of gamma ==================")
    visualize_spectral(data, gammas=[0.1, 1, 10])
