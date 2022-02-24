import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import operator
from itertools import chain

################################################################################
################################################################################
# Part 1: Bow-tie structure connectivity mining
################################################################################
################################################################################
# Read in the networks
D = nx.read_edgelist("out.link-dynamic-simplewiki.data", create_using=nx.DiGraph(), data=(("exists",int),("timestamp",int)))

################################################################################
# (a) Number of weakly connected components.
num_weak_comps = 0

#num_weak_comps = len(list(nx.weakly_connected_components(D))) #303
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

print("Num weak comps:", num_weak_comps)

exit(0) #TEMP

################################################################################
# (b) Number of strongly connected components.
num_strong_comps = 0

# TODO: determine number of strongly connected components

print("Num strong comps:", num_strong_comps)

################################################################################
# (c) Number of trivial strongly connected components.
num_trivial_strong_comps = 0

# TODO: determine number of strongly connected components

print("Num trivial strong comps:", num_trivial_strong_comps)

################################################################################
# (d) Number of vertices in each of SCC, IN, and OUT.
num_in_SCC = 0
num_in_IN = 0
num_in_OUT = 0

# TODO: determine sizes of SCC, IN, and OUT for bowtie structure

print("Num in SCC:", num_in_SCC)
print("Num in IN:", num_in_IN)
print("Num in OUT:", num_in_OUT)

################################################################################
# (e) Number of vertices in each of Tendrils and Tubes.
num_in_tendrils = 0
num_in_tubes = 0

# TODO: determine total number of vertices in tendrils and tubes

print("Num in tendrils:", num_in_tendrils)
print("Num in tubes:", num_in_tubes)

################################################################################
# (f) Number Tendrils and number of Tubes.
num_tendrils = 0
num_tubes = 0

# TODO: determine number of distinct tendrils and number of distinct tubes

print("Num tendrils:", num_tendrils)
print("Num tubes:", num_tubes)

################################################################################
# (g) CSCI-6964 only: Number of trivial tendrils and tubes
num_trivial_tendrils = 0
num_trivial_tubes = 0

# TODO: determine of trivial tendrils and tubes


print("Num trivial tendrils:", num_trivial_tendrils)
print("Num trivial tubes:", num_trivial_tubes)

################################################################################
################################################################################
# Part 2: Real-world graph properties and measurements
################################################################################
################################################################################
# Read in the network
G = nx.read_edgelist("p2p-Gnutella31.data")

################################################################################
# (a) Degree Skew: estimate power-law coefficient
pl_coefficient = 0.0

# TODO: calculate power-law coefficient

print("Power-law coefficient:", pl_coefficient)

################################################################################
# (b) Hubs: find ratio of vertices in tail of distribution
hub_ratio = 0.0

# TODO: calculate ratio of vertices in tails to be defined as hubs

print("Ratio of hubs:", hub_ratio)

################################################################################
# (c) Small-world: Estimate the average shortest paths length
avg_shortest_paths = 0.0

# TODO: estimate the average shortest paths

print("Average shortest path length:", round(avg_shortest_paths, 0))

################################################################################
#(d) CSCI-6964 only: Estimate diameter
diameter = 0

# TODO: estimate the graph diameter

print("Diameter estimate:", diameter)

################################################################################
################################################################################
# Part 3: Link prediction on amazon video dataset
################################################################################
################################################################################
# Read in dataset	and sort edges in temporal order. Use that to create graph on
# the first 25% of edges
G = nx.read_edgelist("amazon_video.data", create_using=nx.Graph(), comments="%", data=(("rating", float),("time",int)))
edges = sorted(G.edges(data=True), key=lambda t: t[2].get('time', 1))

################################################################################
# Create new links using Jaccard index


################################################################################
# Calculate values for common neighbors, adamic/adar, preferential attachment,
# strong triadic closure, and for CSCI-6964 students: personal pagerank


################################################################################
# take top 100 for each as our predictors


################################################################################
# compare the predicted links to the full dataset and output precision

precision_common_neighbors = 0.0
precision_adamic_adar = 0.0
precision_pref_attachment = 0.0
precision_triadic_closure = 0.0
precision_personal_pagerank = 0.0


print("Precision common neighbors:", round(precision_common_neighbors, 2))
print("Precision Adamic/Adar:", round(precision_adamic_adar, 2))
print("Precision preferential attachment:", round(precision_pref_attachment, 2))
print("Precision triadic closure:", round(precision_triadic_closure, 2))
print("Precision personalized pagerank:", round(precision_personal_pagerank, 2))
