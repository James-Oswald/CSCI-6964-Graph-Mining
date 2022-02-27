
import networkx as nx
import random
import math

##########################################################################################################################################

print("Structure for p2p-Gnutella31.data")
#Computing Metrics for p2p-Gnutella31
D = nx.read_edgelist("p2p-Gnutella31.data", create_using=nx.DiGraph())

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

#There was some debate with classmates over what counts as a being in a tendril
#The following was a point of disagreement
#My implementation assumes that if C is in the IN set and we have edges like this:
#   A -> B <- C(IN)
#Then A and B are both in the tendril, not just B. 

num_in_tendrils = 0
num_in_tubes = 0

#print("Generating ttds")
outside = set(D.nodes).difference(scc.union(inNodes).union(outNodes)) #All the nodes not in in, out, or scc

#Set containing components that are in tendrils, tubes, or disconnects, we need to classify each element in this
ttds = list(nx.weakly_connected_components(D.subgraph(outside))) 

#Create helper 
nodeSet = set(D.nodes)
undirectedD = D.to_undirected(as_view=True)  
DwithoutIn = undirectedD.subgraph(nodeSet.difference(inNodes))
DwithoutOut = undirectedD.subgraph(nodeSet.difference(outNodes))

#Classify nodes into Tendrils, Tubes, Or Disconnects
tendrils = []
tubes = []
discon = []
for ttd in ttds:
    ttdElm = list(ttd)[0] #An arbitrary node in the tendril, tube, or disconnected component we're classifying 
    connectsThroughIn = nx.has_path(DwithoutIn, ttdElm, sccElm) #If our current component can connect to SCC through IN
    connectsThroughOut = nx.has_path(DwithoutOut, ttdElm, sccElm) #if our current component can connect to SCC through OUT
    if connectsThroughIn and connectsThroughOut: #if we can get to the scc from both IN and OUT we are a tube
        tubes.append(ttd)
    elif connectsThroughIn or connectsThroughOut: #if we can connect via IN or(xor) OUT we are a tendril 
        tendrils.append(ttd)
    else: #if we cant connect to  SCC at all, we are disconnected
        discon.append(ttd)

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


################################################################################
# (g) CSCI-6964 only: Number of trivial tendrils and tubes
num_trivial_tendrils = 0
num_trivial_tubes = 0

#if our diameter+1 is the same as the number of nodes in the tube or tendril, we are trivial
for tendril in tendrils:
    tendrilGraph = D.subgraph(set(tendril)).to_undirected(as_view=True)  
    if nx.diameter(tendrilGraph)+1 == len(tendril): 
        num_trivial_tendrils +=  1

for tube in tubes:
    tubeGraph = D.subgraph(set(tube)).to_undirected(as_view=True)  
    if nx.diameter(tubeGraph)+1 == len(tube):
        num_trivial_tubes += 1

print("Num trivial tendrils:", num_trivial_tendrils)
print("Num trivial tubes:", num_trivial_tubes)

print("Measures for out.link-dynamic-simplewiki.data")
################################################################################
################################################################################
# Part 2: Real-world graph properties and measurements
################################################################################
################################################################################
# Read in the network

#We already read in the network for part 1, just converting it to an undirected graph
#G = nx.read_edgelist("p2p-Gnutella31.data") #we already read this for part 1
G = undirectedD

################################################################################
# (a) Degree Skew: estimate power-law coefficient
pl_coefficient = 0.0

degSum = 0
for node in list(G.nodes):
    deg = G.degree(node)
    if(deg > 1): #no division by 0 or by -inf
        degSum += math.log(deg)
pl_coefficient = 1 + (len(G.nodes)/degSum)

print("Power-law coefficient:", pl_coefficient)

################################################################################
# (b) Hubs: find ratio of vertices in tail of distribution
hub_ratio = 0.0

numTailNodes = 0
for node in G.nodes:
    if G.degree(node) > math.log(len(G.nodes)):
        numTailNodes += 1
hub_ratio =  numTailNodes / len(G.nodes)

print("Ratio of hubs:", hub_ratio)

################################################################################
# (c) Small-world: Estimate the average shortest paths length

mainComponent = list(max(nx.connected_components(G), key=len))
randomNodes = random.sample(mainComponent, 100) #our sample of 100 random nodes

numPaths = 0
pathLenSum = 0
for u in randomNodes:
    for v in randomNodes:
        if u != v:
            numPaths += 1
            pathLenSum += nx.shortest_path_length(G, u, v)
avg_shortest_paths = pathLenSum / numPaths

print("Average shortest path length:", round(avg_shortest_paths, 0))

################################################################################
#(d) CSCI-6964 only: Estimate diameter

#Precompute Neighbors to speed up the BFS
neighbors = {}
for v in mainComponent:
    neighbors[v] = set(G.neighbors(v))

iterations = 100
pathLenSum = 0
startNode = random.choice(mainComponent)
progress = 0

for _ in range(iterations):
    progress += 1
    if progress % 10 == 0:
        print(str(progress) + "% done computing Diameter")
    #preform a custom BFS that keeps track of search depth
    visited = set()
    curLevel = set([startNode])
    lastLevel = set()
    depth = 0
    while len(curLevel) > 0:
        depth += 1
        #print("depth" + str(depth))
        nextLevel = set()
        for u in curLevel:
            visited.add(u)
            for v in neighbors[u]:
                if v not in visited:
                    nextLevel.add(v)
        lastLevel = curLevel.copy()
        curLevel = nextLevel.copy()
    pathLenSum += depth
    startNode = next(iter(lastLevel))
diameter = pathLenSum / iterations

print("Diameter estimate:", diameter)