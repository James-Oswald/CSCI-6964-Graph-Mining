

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

weakComponents = list(nx.weakly_connected_components(D))
num_weak_comps = len(weakComponents)
print("Num weak comps:", num_weak_comps)

################################################################################
# (b) Number of strongly connected components.

strongComponents = list(nx.strongly_connected_components(D))
num_strong_comps = len(strongComponents)
print("Num strong comps:", num_strong_comps)

################################################################################
# (c) Number of trivial strongly connected components.
num_trivial_strong_comps = len([i for i in strongComponents if len(i) == 1])
print("Num trivial strong comps:", num_trivial_strong_comps)

################################################################################
# (d) Number of vertices in each of SCC, IN, and OUT.

num_in_SCC = 0
num_in_IN = 0
num_in_OUT = 0


scc = max(strongComponents, key=len)
sccElm = list(scc)[0] #An arbitrary element in the scc
inNodes = nx.ancestors(D, sccElm).difference(scc)
outNodes = nx.descendants(D, sccElm).difference(scc)

num_in_SCC = len(scc)
num_in_IN = len(inNodes)
num_in_OUT= len(outNodes)

print("Num in SCC:", num_in_SCC)
print("Num in IN:", num_in_IN)
print("Num in OUT:", num_in_OUT)

################################################################################
# (e) Number of vertices in each of Tendrils and Tubes.
num_in_tendrils = 0
num_in_tubes = 0

print("Generating ttds")
outside = set(D.nodes).difference(scc.union(inNodes).union(outNodes)) #All the nodes not in in, out, or scc

#Set containing components that are in tendrils, tubes, or disconnects, we need to classify each element in this
ttds = list(nx.weakly_connected_components(D.subgraph(outside))) 

print("Generating helper graphs")
nodeSet = set(D.nodes)
undirectedD = D.to_undirected(as_view=True)  
DwithoutIn = undirectedD.subgraph(nodeSet.difference(inNodes))
DwithoutOut = undirectedD.subgraph(nodeSet.difference(outNodes))

print("Computing Tendrils and tubes")
tendrils = []
tubes = []
for ttd in ttds:
    ttdElm = list(ttd)[0] #An arbitrary element in the tendril, tube, or disconnected component we're classifying 
    connectsThroughIn = nx.has_path(DwithoutIn, ttdElm, sccElm)
    connectsThroughOut = nx.has_path(DwithoutOut, ttdElm, sccElm)
    if connectsThroughIn and connectsThroughOut: #if we can get to the scc from both IN and OUT we are a tube
        tubes.append(ttdElm)
    elif connectsThroughIn or connectsThroughOut: #if we can connect via IN or(xor) OUT we are a tendril 
        tendrils.append(ttdElm)
    #if we cant connect to  SCC at all, we are disconnected

num_in_tendrils = sum([len(tendril) for tendril in tendrils])
num_in_tubes = sum([len(tube) for tube in tubes])

print("Num in tendrils:", num_in_tendrils)
print("Num in tubes:", num_in_tubes)

################################################################################
# (f) Number Tendrils and number of Tubes.
num_tendrils = 0
num_tubes = 0

num_tendrils = len(tendrils)
num_tubes = len(tubes)

print("Num tendrils:", num_tendrils)
print("Num tubes:", num_tubes)

exit(0)
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
