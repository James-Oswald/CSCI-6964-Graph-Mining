

#Identification of subgraphs

import os
import sys
import math
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


import networkx as nx
import matplotlib.pyplot as plt

# read in our graph, as a list of edges
G = nx.read_edgelist("lec01.data")

#G.size()    # number of vertices
#G.order()   # number of edges
#G.nodes     # list of vertices
#G.edges     # list of edges
#G.adj       # adjacencies of each vertex

# draw and display our graph
nx.draw(G, with_labels=True)
plt.show()

# run our connectivity algorithm
counter = 0
S = {}
for v in G.nodes:
    S[v] = counter
    counter += 1

updates = 1
while updates > 0:
    updates = 0
    for v in G.nodes:
        for u in G.neighbors(v):
            if S[u] > S[v]:
                S[v] = S[u]
                updates += 1

colors = []
for v in G.nodes:
    colors.append(math.floor((hash(S[v])%500/500)*(pow(2,24)-1)))

nx.draw(G, with_labels=True, node_color=colors)
plt.show()