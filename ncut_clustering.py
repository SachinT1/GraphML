import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
from numpy.linalg import eig
filename = "karateset.txt"

net = Network()

adjmat = np.zeros((34,34))
deg = np.zeros(34)
edgelist = np.zeros((78,2))
temp=0
with open(filename, "r") as file:
    for line in file:
        words = line.strip().split()
        u = int(words[0])-1
        v = int(words[1])-1
        adjmat[u][v]=1
        adjmat[v][u]=1
        deg[u]+=1
        deg[v]+=1
        edgelist[temp][0]=u
        edgelist[temp][1]=v
        temp+=1
# initalized adjajency matrix and degree vector for 34 nodes.
#initalized network object net
diagD = np.zeros((34,34))
for i in range(34):
    diagD[i][i]=deg[i]
laplacian = diagD - adjmat
normalised_laplacian = np.matmul(np.linalg.inv(diagD),laplacian)
w,v = eig(normalised_laplacian)
mag = np.ones(34)
index=0
for item in w:
    mag[index]=abs(item)
    index+=1
smallest=0
second = 0
currsmall = mag[0]
for i in range(1,34):
    if(mag[i]<currsmall):
        smallest = i
if(smallest==0):
    currsmall = mag[1]
else:
    currsmall = mag[0]
for i in range(34):
    if(i!=smallest):
        if(mag[i]<currsmall):
            second=i
            currsmall = mag[i]

#second smallest eigenvalue of laplacian matrix,and the eigenvector
cluster_vector = v[second]
for i in range(34):
    if(cluster_vector[i]>=0):
        net.add_node(i,f"Node {i}",color='red')
    else:
        net.add_node(i,f"Node {i}",color='blue')
net.add_edges(edgelist)
net.save_graph("ncut.html")


            
