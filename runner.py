import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pyvis.network import Network
from numpy.linalg import eig

from scipy.linalg import sqrtm 
from scipy.special import softmax
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community import label_propagation_communities
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from gcn import *


def draw_kkl(nx_G, s,label_map, node_color, pos=None, **kwargs):
    fig, ax = plt.subplots(figsize=(10,10))
    if pos is None:
        pos = nx.spring_layout(nx_G, k=5/np.sqrt(nx_G.number_of_nodes()))

    nx.draw(
        nx_G, pos, with_labels=True,  
        node_color=node_color, 
        ax=ax, **kwargs)
    pltname = nx_G.graph['name']
    extra_text = ""
    if s==0:
        extra_text="Graph"
    elif s==1:
        extra_text="Pre Training embeddings"
    elif s==2:
        extra_text="Post Training embeddings"

    plt.savefig(f"{pltname+extra_text}.png", bbox_inches='tight', transparent=True)



def func(optn):
    if optn == 1:
        G = nx.karate_club_graph()
        G.graph['name'] = 'Zachary Karate Club Graph'
        
        comm = greedy_modularity_communities(G)
        train_nodes = np.array([0, 1, 8])
        
    elif optn == 2:
        G = nx.barbell_graph(10,4)
        G.graph['name'] = 'Barbell Graph'
        train_nodes = np.array([4,18])
        comm = greedy_modularity_communities(G)
        #comm = label_propagation_communities(G)
        
        
    elif optn==3:
        G = nx.connected_caveman_graph(5,5)
        G.graph['name'] = 'Caveman Graph'
        train_nodes = np.array([0,5,10,15,20])
        comm = greedy_modularity_communities(G)

    elif optn==4:
        G = nx.windmill_graph(4,5)
        G.graph['name'] = 'Windmill Graph'
        train_nodes = np.array([3,5,10,16])
        comm = greedy_modularity_communities(G)

    else:
        G = nx.barbell_graph(10,4)
        G.graph['name'] = 'Barbell Graph'
        train_nodes = np.array([4,18])
        #comm = greedy_modularity_communities(G)
        comm = label_propagation_communities(G)
    print(G.graph['name'])
    nx.draw(G,with_labels=True)

    plt.show()
    
    A = nx.to_numpy_array(G)
    A_mod = A + np.eye(np.shape(A)[0])
    D_mod = np.zeros_like(A_mod)
    np.fill_diagonal(D_mod, np.asarray(A_mod.sum(axis=1)).flatten())
    D_mod_invroot = np.linalg.inv(sqrtm(D_mod))
    A_hat = D_mod_invroot @ A_mod @ D_mod_invroot #normalised adjajency matrix
    X = np.eye(np.shape(A)[0]) ##one hot labels for nodes.

    
    ##node_index = {node: i for i, node in enumerate(G.nodes())}
    colors = np.zeros(G.number_of_nodes())
    for i, com in enumerate(comm):
        colors[list(com)] = i

    n_classes = np.unique(colors).shape[0]
    labels = np.eye(n_classes)[colors.astype(int)]
    gcn_model = GCNNetwork(n_inputs=G.number_of_nodes(), n_outputs=n_classes, n_layers=2,hidden_sizes=[16, 2], activation=np.tanh,seed=100,)
    print(gcn_model)
    y_pred = gcn_model.forward(A_hat, X)
    embed = gcn_model.embedding(A_hat, X)
    print(xent(y_pred, labels).mean())
    
    
   
    _ = draw_kkl(G,0, None, colors, pos=None, cmap='gist_rainbow', edge_color='gray')
    ###graph
    pos = {node: embed[i, :] for i, node in enumerate(G.nodes())}
    #pos = {i: embed[i,:] for i in range(embed.shape[0])}
    
    p = draw_kkl(G,1, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')


    #pre training embeddings

      
    test_nodes = np.array([i for i in range(labels.shape[0]) if i not in train_nodes])
    opt2 = GradDescentOptim(lr=2e-2, wd=2.5e-2)
    embeds = list()
    accs = list()
    train_losses = list()
    test_losses = list()

    loss_min = 1e6
    es_iters = 0
    es_steps = 50
    # lr_rate_ramp = 0 #-0.05
    # lr_ramp_steps = 1000
    for epoch in range(20000):
        
        
        y_pred = gcn_model.forward(A_hat, X)

        opt2(y_pred, labels, train_nodes)
        
    #     if ((epoch+1) % lr_ramp_steps) == 0:
    #         opt2.lr *= 1+lr_rate_ramp
    #         print(f"LR set to {opt2.lr:.4f}")

        for layer in reversed(gcn_model.layers):
            layer.backward(opt2, update=True)
            
        embeds.append(gcn_model.embedding(A_hat, X))
        # Accuracy for non-training nodes
        acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
            [i for i in range(labels.shape[0]) if i not in train_nodes]
        ]
        accs.append(acc.mean())
        
        loss = xent(y_pred, labels)
        loss_train = loss[train_nodes].mean()
        loss_test = loss[test_nodes].mean()
        
        train_losses.append(loss_train)
        test_losses.append(loss_test)
        
        if loss_test < loss_min:
            loss_min = loss_test
            es_iters = 0
        else:
            es_iters += 1
            
        if es_iters > es_steps:
            print("Early stopping!")
            break
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch+1}, Train Loss: {loss_train:.3f}, Test Loss: {loss_test:.3f}")

    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)

    fig, ax = plt.subplots()
    ax.plot(np.log10(train_losses), label='Train')
    ax.plot(np.log10(test_losses), label='Test')
    ax.legend()
    ax.grid()

    pos = {i: embeds[-1][i,:] for i in range(embeds[-1].shape[0])}
    post_train = draw_kkl(G,2, None, colors, pos=pos, cmap='gist_rainbow', edge_color='gray')
    
    


func(2)


