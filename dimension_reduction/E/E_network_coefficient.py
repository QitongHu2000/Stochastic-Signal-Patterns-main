# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.io import loadmat

def average_triangle(G):
    triangles=list()
    for u in G.nodes:
        triangle=0
        for v in G[u]:
            nodes1=G[u]
            nodes2=G[v]
            triangle+=len([u for u in nodes1 if u in nodes2])
        triangles.append(triangle)
    A=nx.to_numpy_matrix(G)
    degrees=np.sum(A,axis=0)
    degrees=[degrees[0,i]*(degrees[0,i]-1) for i in range(len(A))]
    # for i,j in zip(triangles,degrees):
    #     print(i,j)
    density=[i/j for i,j in zip(triangles,degrees)]
    return np.max(density)
    
A_edge=np.mat(loadmat('../Networks/ER1.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
# print('ER',average_triangle(G_edge))
print(nx.average_clustering(G_edge))

A_edge=np.mat(loadmat('../Networks/BA1.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
# print('BA',average_triangle(G_edge))
print(nx.average_clustering(G_edge))

G_edge = nx.karate_club_graph()#nx.parse_gml(gml)
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=nx.to_numpy_matrix(G_edge)
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
# print('karate',average_triangle(G_edge))
print(nx.average_clustering(G_edge))