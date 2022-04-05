
import networkx as nx
import random
import math

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