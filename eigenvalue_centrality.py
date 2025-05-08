import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
from numpy.linalg import eig
filename = "karateset.txt"

net = Network()
for i in range(34):
    net.add_node(i,"{b}".format(b=i))
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
        net.add_edge(u,v)
        edgelist[temp][0]=u
        edgelist[temp][1]=v
        temp+=1
# initalized adjajency matrix and degree vector for 34 nodes.
#initalized network object net

w,v = eig(adjmat)

#w has list of eigenvalue, consider only the largest magintude. 

mag = np.ones(34)
index=0
for item in w:
    mag[index]=abs(item)
    index+=1

eigen_index = np.argmax(mag)
eigen_val = float(w[eigen_index])

graph_html = "basic.html"
net.save_graph(graph_html)

eigencentre = v[:,eigen_index].real
#list of eigencentrality values
normeigen = mcolors.Normalize(vmin=min(eigencentre),vmax=max(eigencentre),clip=True)
cmap = cm.get_cmap("coolwarm")
colors = [mcolors.to_hex(cmap(normeigen(v))) for v in eigencentre]

eigen_net = Network(select_menu=True,filter_menu=True,)
for i , color in enumerate(colors):
    eigen_net.add_node(i,label=f"Node {i}",color=color)

for i in range(78):
    eigen_net.add_edge(edgelist[i][0],edgelist[i][1])

eigen_net.show_buttons(filter_=['physics'])
eigen_net.save_graph("eigen.html")
       
        