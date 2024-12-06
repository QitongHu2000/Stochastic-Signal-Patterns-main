# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

A_edge=np.mat(loadmat('../Networks/BA1.mat')['A'])
G_edge=nx.from_numpy_matrix(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=nx.to_numpy_matrix(G_edge)
G_edge=nx.subgraph(G_edge,range(0,25))
G = nx.DiGraph(G_edge)

fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111)
pos = nx.circular_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#526D92')
for edge in G.edges:
    if edge[0] > edge[1]+1:
        if abs(edge[0]- edge[1]) <10:
            nx.draw_networkx_edges(G, pos, edge_color='#526D92', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {0.3}')
        else:
            nx.draw_networkx_edges(G, pos, edge_color='#526D92', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.3}')
    if edge[0] ==   edge[1]+1:
        nx.draw_networkx_edges(G, pos, edge_color='#526D92', width=1, alpha=0.35,edgelist=[(edge[0],edge[1])], arrowstyle = '-',connectionstyle=f'arc3, rad = {-0.1}')

ax.set_aspect('equal')
ax.set_axis_off()
plt.savefig('B_BA.pdf',dpi = 300)
plt.show()