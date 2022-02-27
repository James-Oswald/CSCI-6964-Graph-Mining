
import networkx as nx
import random
import math

################################################################################
################################################################################
# Part 1: Bow-tie structure connectivity mining
################################################################################
################################################################################
# Read in the networks

print("Structure for out.link-dynamic-simplewiki.data")
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
    if progress % 5 == 0:
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


################################################################################
################################################################################
# Part 3: Link prediction on amazon video dataset
################################################################################
################################################################################
# Read in dataset	and sort edges in temporal order. Use that to create graph on
# the first 25% of edges

G = nx.read_edgelist("amazon_video.data", create_using=nx.Graph(), comments="%", data=(("rating", float),("time",int)))
edges = sorted(G.edges(data=True), key=lambda t: t[2].get('time', 1))

#compute the graph with just 25% of the links
firstFourthEdges = edges[0:int(0.25*len(edges))]
users = set()
products = set()
firstFourthEdgesWoData = []
for u,v,_ in firstFourthEdges: #slice off the data so we can call edge_subgraph 
    firstFourthEdgesWoData.append((u,v))
    if u[0]=='A':
        users.add(u)
        products.add(v)
    else:
        users.add(v)
        products.add(u)

#subG is a graph that has 25% of the data
#userG is a subgraph of subG consisting exclusivly of user verts
subG = G.edge_subgraph(firstFourthEdgesWoData).copy()
userG = G.subgraph(users).copy()                       
################################################################################
# Create new links using Jaccard index

#Generate the user-user pairs we're interested in finding the Jaccard index for
jaccardPairs = set()
for u1 in users:
    for u2 in users:
        if u1 != u2 and (u2, u1) not in jaccardPairs: #no self edges + no repeat edges since undirected
            jaccardPairs.add((u1, u2))

#precompute neighbors or all users to speed up jaccard index comp
neighbors = {}
for u in users:
    neighbors[u] = set(subG.neighbors(u))
    
print("Adding new edges based on jaccard indices")
for u, v in jaccardPairs:
    jaccardIndex = len(neighbors[u].intersection(neighbors[v]))/len(neighbors[u].union(neighbors[v]))
    if(jaccardIndex > 0.5):
        userG.add_edge(u,v, rating=5*jaccardIndex, time=0)
        subG.add_edge(u, v, rating=5*jaccardIndex, time=0)

################################################################################
# Calculate values for common neighbors, adamic/adar, preferential attachment,
# strong triadic closure, and for CSCI-6964 students: personal pagerank

#precompute user, product, and all neighbors
userNeighbors = {}
prodNeighbors = {} 
for u in users: 
    userNeighbors[u] = set(userG.neighbors(u)) 
    prodNeighbors[u] = set(i for i in subG.neighbors(u) if i in products)
allNeighbors = {}
for n in subG.nodes:
    allNeighbors[n] = set(subG.neighbors(n))

print("Computing Predictions")
commonNeighbors = {}
adamicAdar = {}
prefAttachment = {}
strongTriadicClosure = {}

#for each user, look at all the products of their neighbors
for u in users: 
    for friend in userNeighbors[u]:
        for prod in prodNeighbors[friend]:
            link = (u, prod)
            #we only compute this connection if we haven't already and if its not something we're already connected to
            if link not in commonNeighbors and not subG.has_edge(u, prod): 
                inter = allNeighbors[u].intersection(allNeighbors[prod]) #the set of common Neighbors
                commonNeighbors[link] = len(inter)
                adamicAdar[link] = sum(1/math.log(len(allNeighbors[n])) for n in inter if len(allNeighbors[n]) > 1)
                prefAttachment[link] = len(allNeighbors[u])*len(allNeighbors[prod])
                strongTriadicClosure[link] = subG.get_edge_data(u, friend)["rating"] + subG.get_edge_data(u, friend)["rating"]

#personalized page rank calcuation
print("Computing Personalized Page Rank")
pageRankUsers = random.sample(users, 100) #we only look at 100 users for page rank
ppr = {} 
for u in pageRankUsers:

    #compute personalized page rank for u
    #init all node probabilities
    p = {} #P(v)
    initValue = 1/len(subG.nodes)
    for v in subG.nodes: 
        p[v] = initValue
    
    #run the personalized page rank alg, I use a modified vertex centric model itterative method
    for _ in range(1): #number of update iterations
        for v in subG.nodes:
            updateSum = 0
            for w in allNeighbors[v]:
                updateSum += p[w] / len(allNeighbors[w])
            #need a way of "jumping back to u" for our vertex centric ppr model
            #I implement this by saying there is a 1% chance at each node update that u steals the update
            if(random.random() < 0.01):
                p[u] += updateSum
            else:
                p[v] += updateSum
    #the ppr of the product with respect to u is our "score" for how likely we think this connection is
    for friend in userNeighbors[u]:
        for prod in prodNeighbors[friend]:
            link = (u, prod)
            #we only compute this connection if we haven't already and if its not something we're already connected to
            if link not in ppr and not subG.has_edge(u, prod):
                ppr[link] = p[prod]


################################################################################
# take top 100 for each as our predictors

#return the top 100 links as a list of tuples [(u1,v1), (u2, v2), ...]
def getTop(predictions):
    items = list(predictions.items())
    sortedTopItems = sorted(items, reverse=True, key=lambda i:i[1])[:100] 
    return [link for link,_ in sortedTopItems]

topCommonNeighbors = getTop(commonNeighbors)
topAdamicAdar = getTop(adamicAdar)
topPrefAttachment = getTop(prefAttachment)
topstrongTriadicClosure = getTop(strongTriadicClosure)
topPPR = getTop(ppr)

################################################################################
# compare the predicted links to the full dataset and output precision

def computePrecision(predictions):
    numCorrectPredictions = 0
    for link in predictions:
        if G.has_edge(link[0], link[1]):
            numCorrectPredictions += 1
    return numCorrectPredictions / len(predictions)

precision_common_neighbors = computePrecision(topCommonNeighbors)
precision_adamic_adar = computePrecision(topAdamicAdar)
precision_pref_attachment = computePrecision(topPrefAttachment)
precision_triadic_closure = computePrecision(topstrongTriadicClosure)
precision_personal_pagerank = computePrecision(topPPR)

print("Precision common neighbors:", round(precision_common_neighbors, 2))
print("Precision Adamic/Adar:", round(precision_adamic_adar, 2))
print("Precision preferential attachment:", round(precision_pref_attachment, 2))
print("Precision triadic closure:", round(precision_triadic_closure, 2))
print("Precision personalized pagerank:", round(precision_personal_pagerank, 2))