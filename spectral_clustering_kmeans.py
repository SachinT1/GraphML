import pandas as pd
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
from numpy.linalg import eig
filename = "karateset.txt"

A = np.zeros((34,34))
D = np.zeros((34,34))
E = np.zeros((78,2))
temp=0
with open(filename,"r") as file:
    for line in file:
        words = line.strip().split()
        u = int(words[0])-1
        v = int(words[1])-1
        A[u][v]=1
        A[v][u]=1
        D[u][u]+=1
        D[v][v]+=1
        E[temp][0]=u
        E[temp][1]=v
        temp+=1

L = D - A
w,v = eig(L)
#w = eigenvalues, v = eigenvectors
li = []*34
for i in range(34):
   # print(i," ith ",w[i]," abs = ",abs(w[i]))
    li.append([i,abs(w[i])])

li.sort(key=lambda l:l[1])

k = 4
##number of clusters. take k smallest eigenvectors

U = np.zeros((34,k))
for i in range(1,k+1): ##skip the smallest eigenvector of all ones.
    for j in range(34):
        U[j][i-1]=v[j][i]


def k_means_clustering(data, max_iterations=100):
    """
    Perform K-means clustering on a dataset of shape (n_samples, n_features).

    Parameters:
        data (np.ndarray): Input array of shape (n, k)
        max_iterations (int): Maximum optimization steps

    Returns:
        centroids (np.ndarray): Final cluster centers (k, k)
        labels (np.ndarray): Cluster assignments (n,)
    """
    n, k = data.shape  # Get dimensions
    
    # Random centroid initialization from data points
    np.random.seed(42)  # For reproducibility
    initial_indices = np.random.choice(n, k, replace=False)
    centroids = data[initial_indices]

    for _ in range(max_iterations):
        # Calculate pairwise distances between points and centroids
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids as cluster means
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids

    return centroids, labels

centroid,labels = k_means_clustering(U)

def map_labels_to_colors(labels, k):
    colors = ["#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)]) for _ in range(k)]
    
    # Map each unique label to a color
    unique_labels = list(set(labels))
    color_map = dict(zip(unique_labels, colors[:len(unique_labels)]))
    
    return color_map

color_map = map_labels_to_colors(labels,k)
cluster_graph = Network()

for i,label in enumerate(labels):
    col = color_map[labels[i]]
    cluster_graph.add_node(i,f"Node {i}",color=col)
 
cluster_graph.add_edges(E)
cluster_graph.save_graph("spectral_clustering.html")






