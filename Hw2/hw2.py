
import os
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

# Part 1 ================================================================================

G: nx.Graph = nx.read_edgelist("facebook_combined.data", create_using=nx.Graph, nodetype=int)

#https://networkx.org/documentation/stable/auto_examples/graph/plot_erdos_renyi.html
erdosRenyi: nx.Graph = nx.gnm_random_graph(len(G.nodes), len(G.edges))

#Configuration model
degreeSequence: nx.Graph = sorted([nx.degree(G, n) for n in G.nodes], reverse=True)
configurationModel: nx.Graph = nx.configuration_model(degreeSequence, create_using=nx.Graph)

# Part 2 ================================================================================

#Label propagation algorithm outlined in https://arxiv.org/pdf/0709.2938.pdf page 5
#Label propagation helper function
## Return the list of labels maximally occurring labels on neighbors of node in graph
def labelCounts(graph: nx.Graph, node: int)->"set(int)":
    labelCounts: "dict(int, int)" = {}
    maxCount: int = 0
    for neighbor in nx.neighbors(graph, node): 
        #Increase the count of the label of the current node
        if graph.nodes[neighbor]["C"] not in labelCounts:
            labelCounts[graph.nodes[neighbor]["C"]] = 1
        else: 
            labelCounts[graph.nodes[neighbor]["C"]] += 1
        #Update the max label count 
        #(the number of occourances of the most frequently occouring label)
        if labelCounts[graph.nodes[neighbor]["C"]] > maxCount:
            maxCount = labelCounts[graph.nodes[neighbor]["C"]]
    #The list of labels occouring most frequently
    maxLabels: "set(int)" = set(label for label, count in labelCounts.items() if count == maxCount)
    return maxLabels

def display(graph: nx.Graph, figureName:str):
    #Identify contiguous communities   
    contiguousCommunities = []
    visited = set()
    for n in graph.nodes:   #Just doing a BFS to find contiguous communities
        if n in visited:
            continue
        curLevel = [n]
        members = set()
        members.add(n)
        while len(curLevel) > 0:
            nextLevel = []
            for curLevelNode in curLevel:
                for neighbor in nx.neighbors(graph, curLevelNode):
                    if neighbor not in visited and graph.nodes[neighbor]["C"] == graph.nodes[n]["C"]:
                        visited.add(neighbor)
                        nextLevel.append(neighbor)
                        members.add(neighbor)
            curLevel = nextLevel
        contiguousCommunities.append(members)

    #Contract contiguous communities
    contractedGraph = graph.copy()
    for n in contractedGraph:
        contractedGraph.nodes[n]["Size"] = 1
    for cc in contiguousCommunities:
        nodes = sorted(list(cc))
        for node in nodes[1:]:
            nx.contracted_nodes(contractedGraph, nodes[0], node, copy=False, self_loops=False)
        contractedGraph.nodes[nodes[0]]["Size"] = len(nodes)

    #Draw communities:
    largeCommunities = [cc for cc in contractedGraph if contractedGraph.nodes[cc]["Size"] >= 5]
    largeCommunitySubgraph = contractedGraph.subgraph(largeCommunities)
    nodeSizes = [contractedGraph.nodes[cc]["Size"] for cc in largeCommunitySubgraph]
    nodeColors = [int(contractedGraph.nodes[cc]["C"]) for cc in largeCommunitySubgraph]
    nx.draw(largeCommunitySubgraph, node_size=nodeSizes, node_color=nodeColors)
    plt.savefig(figureName)
    

def labelProp(graph: nx.Graph):
    for n in graph:
        graph.nodes[n]["C"] = n
    t: int = 1
    while True:
        X = list(graph.nodes)
        random.shuffle(X)
        maxLabels: "dict(int,set(int))" = {}
        for x in X:
            maxLabels[x] = labelCounts(graph, x)
            #Set the new label
            if len(maxLabels[x]) != 1: #If there is a tie, break it with a uniform random sample 
                graph.nodes[x]["C"] = random.choice(tuple(maxLabels[x]))
            else: #Otherwise we select the main one
                graph.nodes[x]["C"] = tuple(maxLabels[x])[0]

        if t % 10 == 0:
            display(graph, "./Pics/" + str(t) + ".png")
        print("Step " + str(t))
        #Check if we have to stop (each node has the label the maxium number of their neighbors have)
        stopFlag = True
        for n in graph:
            if graph.nodes[n]["C"] in maxLabels[n]:
                stopFlag = False
                break
        if stopFlag:
            break
        t += 1
    
labelProp(G)

