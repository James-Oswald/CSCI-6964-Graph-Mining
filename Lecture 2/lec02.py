
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import networkx as nx
import matplotlib.pyplot as plt 
from itertools import chain

# read in and visualize dataset
G = nx.read_edgelist("out.moreno_zebra_zebra.data", comments="%")
nx.draw(G, with_labels=True)
plt.show()

# look at the connectivity structure using NetworkX functions
print(nx.is_connected(G))
print(nx.number_connected_components(G))
C = nx.connected_components(G)
for c in C:
  print(c)

nx.node_connected_component(G, '14')

# take the largest connected component and create a new graph from it
G1 = G.subgraph(nx.node_connected_component(G, '14'))
G1.order()
G1.size()
nx.is_connected(G1)
nx.draw(G1)
plt.show()

# load a directed graph (no multi-edge)
D = nx.read_edgelist("lec02.data", create_using=nx.DiGraph())
D.order()
D.size()
nx.draw(D, with_labels=True)
plt.show()

# load a directed graph w/multi-edges
D = nx.read_edgelist("lec02-multi.data", create_using=nx.MultiDiGraph())
D.order()
D.size()
nx.draw(D, with_labels=True)
plt.show()

for u in D.neighbors('bob'):
  print(u)

for u in D.successors('bob'):
  print(u)

for u in D.predecessors('bob'):
  print(u)

# directed equivalent to 'all neighbors' for undirected graphs
for u in chain(D.predecessors('bob'), D.successors('bob')):
  print(u)

# connectivity algorithm from lecture 1
def connectivity(G):
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
  
  return S

# connectivity algorithm from lecture 1, but for weak connectivity
def weak_connectivity(G):
  counter = 0
  S = {}
  for v in G.nodes:
    S[v] = counter
    counter += 1
    
  updates = 1
  while updates > 0:
    updates = 0
    for v in G.nodes:
      for u in chain(G.successors(v),G.predecessors(v)):
        if S[u] > S[v]:
          S[v] = S[u]
          updates += 1
  
  return S

S = connectivity(G)
S = weak_connectivity(D)