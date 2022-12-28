"""
EECS 445 - Introduction to Machine Learning
Fall 2022 - Homework 3
Spectral Clustering
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


def spectral_clustering():
    '''
    As specified in Question 4, you will initialize the laplacian computed in part (a), compute the eigenvectors corresponding
    to the k lowest eigenvalues, and plot the rows of the eigenvectors to determine the cluster assignments.
    '''
    lap_matrix = np.matrix([[17/12, -3/4, -1/3, 0, -1/3],
                            [-3/4, 3/2, -1/4, -1/2, 0],
                            [-1/3, -1/4, 19/12, -4/5, -1/5],
                            [0, -1/2, -4/5, 13/10, 0],
                            [-1/3, 0, -1/5, 0, 8/15]])
    eigenvalues, eigenvectors = eigh(lap_matrix)
    print(eigenvalues[:3])
    print(eigenvectors[:3])
    ax = plt.axes(projection='3d')

    for i in range(len(eigenvectors)):
        # plt.scatter(eigenvectors[0][i], eigenvectors[1][i])
        ax.scatter(eigenvectors[0][i], eigenvectors[1]
                   [i], eigenvectors[2][i])
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.savefig('q4 k=3')


if __name__ == '__main__':
    spectral_clustering()
