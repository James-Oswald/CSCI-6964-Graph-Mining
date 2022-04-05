

import random
import networkx as nx
import matplotlib.pyplot as plt

# Part 1 ================================================================================

G: nx.Graph = nx.read_edgelist("facebook_combined.data", create_using=nx.Graph)

#https://networkx.org/documentation/stable/auto_examples/graph/plot_erdos_renyi.html
erdosRenyi: nx.Graph = nx.gnm_random_graph(len(G.nodes), len(G.edges))

#Configuration model
degreeSequence: nx.Graph = sorted([nx.degree(G, n) for n in G.nodes], reverse=True)
configurationModel: nx.Graph = nx.configuration_model(degreeSequence, create_using=nx.Graph)

# Part 2 ================================================================================

#Label propagation algorithm outlined in https://arxiv.org/pdf/0709.2938.pdf page 5
#Label propagation helper function
## Return the list of labels maximally occurring labels on neighbors of node in graph
def labelCounts(graph: nx.Graph, node: int)->list(int):
    labelCounts: "dict(int, int)" = {}
    maxCount: int = 0
    for neighbor in nx.neighbors(graph, node): 
        #Increase the count of the label of the current node
        if neighbor["C"] not in labelCounts:
            labelCounts[neighbor["C"]] = 1
        else: 
            labelCounts[neighbor["C"]] += 1
        #Update the max label count 
        #(the number of occourances of the most frequently occouring label)
        if labelCounts[neighbor["C"]] > maxCount:
            maxCount = labelCounts[neighbor["C"]]
    #The list of labels occouring most frequently
    maxLabels: "list(int)" = [label for label, count in labelCounts.items() if count == maxCount]
    return maxLabels

def display(graph: nx.Graph, figureName:str):
    #Condense contiguous communities
    communityGraph = graph.copy()
    for n in communityGraph:
        n["Size"] = 1
    while True:
        n = next(communityGraph.__iter__, None)
        if n == None:
            break
        for neighbor in nx.neighbors(communityGraph, n):
            if communityGraph[neighbor]["C"] == communityGraph[n]["C"]:
                n["Size"] += neighbor["Size"]
                nx.contracted_edge(communityGraph, (n, neighbor), copy=False)

    #Draw communities:
    largeCommunities = [community for community in communityGraph if community["Size"] >= 5]
    largeCommunitySubgraph = communityGraph.subgraph(largeCommunities)
    nodeSizes = [community["Size"] for community in largeCommunitySubgraph]
    nx.draw(largeCommunitySubgraph, node_size = nodeSizes)
    plt.savefig(figureName)
    

def labelProp(graph: nx.Graph):
    for n in graph:
        n["C"] = n
    t: int = 1
    while True:
        X = random.shuffle(list(graph.nodes))
        for x in X:
            maxLabels = labelCounts(graph, x)
            #Set the new label
            if len(maxLabels) != 1: #If there is a tie, break it with a uniform random sample 
                x["C"] = random.choice(maxLabels)
            else: #Otherwise we select the main one
                x["C"] = maxLabels[0]
        #Check if we have to stop (each node has the label the maxium number of their neighbors have)
        stopFlag = True
        for n in graph:
            if n["C"] in labelCounts(graph, n):
                stopFlag = False
                break
        if stopFlag:
            break
    
    



