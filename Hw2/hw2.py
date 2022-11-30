
import itertools
import os
import sys
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

# Part 1 ================================================================================

print("Loading Dataset")
G: nx.Graph = nx.read_edgelist("facebook_combined.data", comments="%")

#https://networkx.org/documentation/stable/auto_examples/graph/plot_erdos_renyi.html
erdosRenyi: nx.Graph = nx.gnm_random_graph(len(G.nodes), len(G.edges))

#Configuration model
degreeSequence: nx.Graph = sorted([nx.degree(G, n) for n in G.nodes], reverse=True)
configurationModel: nx.Graph = nx.configuration_model(degreeSequence)

# Part 2 ================================================================================

#My original algorithm from the recomended paper. 
#It was not faster than the one presented in class as I had hoped.
#Label propagation algorithm outlined in https://arxiv.org/pdf/0709.2938.pdf page 5
#Label propagation helper function
## Return the list of labels maximally occurring labels on neighbors of node in graph
# def labelCounts(graph: nx.Graph, node: int)->"set(int)":
#     labelCounts: "dict(int, int)" = {}
#     maxCount: int = 0
#     for neighbor in nx.neighbors(graph, node): 
#         #Increase the count of the label of the current node
#         if graph.nodes[neighbor]["C"] not in labelCounts:
#             labelCounts[graph.nodes[neighbor]["C"]] = 1
#         else: 
#             labelCounts[graph.nodes[neighbor]["C"]] += 1
#         #Update the max label count 
#         #(the number of occourances of the most frequently occouring label)
#         if labelCounts[graph.nodes[neighbor]["C"]] > maxCount:
#             maxCount = labelCounts[graph.nodes[neighbor]["C"]]
#     #The list of labels occouring most frequently
#     maxLabels: "set(int)" = set(label for label, count in labelCounts.items() if count == maxCount)
#     return maxLabels
    
#A spin on the Label Propigation alg used in class, Does not finish in a reasonable amount of time, but produces some 
#Interesting pictures while generating, showing that the Configuration Model and Erdos Renyi Group up faster. 
#I was never able to get this to finish running.


def labelProp(G, name:str, maxIter = float("inf")):
    comms = {}
    for v in G.nodes():
        comms[v] = int(v)
    updates = 1
    t = 0
    while updates > 0 and t < maxIter:
        updates = 0
        for v in sorted(G.nodes(), key=lambda k: random.random()):
            counts = {}
            for u in G.neighbors(v):
                if comms[u] not in dict.keys(counts):
                    counts[comms[u]] = 1
                else:
                    counts[comms[u]] += 1
            c = random.choice([k for k in counts.keys() if counts[k]==max(counts.values())])
            if c != comms[v]:
                comms[v] = c
                updates += 1
        if t % 100 == 0:
            display(G, comms, f"./{name}{t}.png", True)
        print(t)
        t += 1
    return comms
    
#labelProp(G)

def display(graph: nx.Graph, comms, figureName:str, classLblProp:bool = False):
    #Identify contiguous communities if we're running our custom class label prop
    if not classLblProp:
        contiguousCommunities = comms
    else:
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
                        if neighbor not in visited and comms[neighbor] == comms[n]:
                        #if neighbor not in visited and graph.nodes[neighbor]["C"] == graph.nodes[n]["C"]:
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
    largeCommunities = [cc for cc in contractedGraph if contractedGraph.nodes[cc]["Size"] >= 1]
    largeCommunitySubgraph = contractedGraph.subgraph(largeCommunities)
    nodeSizes = [contractedGraph.nodes[cc]["Size"] for cc in largeCommunitySubgraph]
    nodeLabels = {cc: str(contractedGraph.nodes[cc]["Size"]) for cc in largeCommunitySubgraph}
    nodeColors = [contractedGraph.nodes[cc]["Size"] for cc in largeCommunitySubgraph]
    nx.draw(largeCommunitySubgraph, node_size=nodeSizes, node_color=nodeColors, labels=nodeLabels)
    plt.savefig(figureName)
    plt.clf()


#used for disabling pt2 for debuging 
part2 = True

print("Starting Part 2")
#Using the lbl propigation alg from class failes to finish in a timely manor, we use the builtin instead
useCustomLbLProp = False
if part2:
    if useCustomLbLProp:
        labelProp(G, "raw", 100)
        labelProp(erdosRenyi, "er", 100)
        labelProp(configurationModel, "cofig", 100)
    else:
        comms = nx.community.label_propagation_communities(G)
        display(G, comms, "./raw.png")
        comms = nx.community.label_propagation_communities(erdosRenyi)
        display(erdosRenyi, comms, "./er.png")
        comms = nx.community.label_propagation_communities(configurationModel)
        display(configurationModel, comms, "./config.png")

# Part 2 Responses:
# A) In the raw Facebook SNAP graph we see the formation of around 44 communities.
#    For both the the Erdos-Renyi and the configuration model we see the creation of one giant community.
#    These results were obtained via label_propagation_communities but the custom labelProp
#    alg can be seen converging to similar solutions.
# B) For the raw snap graph it appears to be an exponential distribution. We see 3
#    large communities > 500 nodes, 5 mid tier communities > 100 nodes, and many smaller
#    communities < 100 nodes most closer to 5-6 nodes. For both the Erdos-Renyi and
#    the configuration model, we only observe one community containing all nodes.
# C) While I did observe that both random networks converged to a single community,
#    The way in which this convergence happened as displayed by labelProp was different.
#    Erdos-Renyi converged more slowly than the configuration model after the same number
#    of iterations.
# D) The real network actually converges to a set of reasonable communities whereas the
#    random networks do not. Both random networks converged to a single community while
#    The real network converged to a set of communities following a reasonable distribution.
# E) Due to the way in which random graphs are constructed it would make sense to get these
#    community detection results. Since in Erdos-Renyi each edge has a fixed probability of
#    being present or not being present, it is unlikely that real communities would be able to form
#    since the creation of communities requires multiple high concentrations of edges whereas
#    Erdos-Renyi would distribute these edges uniformly over the network.
#    For the configuration model we encounter the same issue, despite having some nodes with more
#    edges the fact that the edge-stubs are being selected to be joined uniformly means it is most likely
#    for the nodes with lots of edges to be connected to nodes with lots of other edges, creating a single
#    large community in which all high degree nodes are likely to be connected, further pulling in the rest


# Part 3 ================================================================================

#Map nodes in graphs to their states

def SIRModel(G, name):
    print(f"Starting SIR Model for {name}")
    maxInfectedSum = 0
    epidemicDurationSum = 0
    totalInfectedSum = 0

    for i in range(100): #we run the model 100 times to average the stats
        if i % 10 == 0:
            print(f"SIR Model epoch {i}")

        #Map nodes to their infection states
        states:"dict(int,str)" = {}
        #Map nodes to how many cycles they've been infected for
        timeInfected:"dict(int,int)" = {}
        
        #Initially, the entirety of the population is susceptible.
        for v in G.nodes:
            states[v] = "susceptible"
            timeInfected[v] = 0

        #Randomly initialize ten vertices as infectious.
        infectedNodes = random.sample(G.nodes, 10)
        for v in infectedNodes:
            states[v] = "infected"

        #Our stats for this iteration
        maxInfected = 10
        epidemicDuration = 0
        totalInfected = 10

        #Main pandemic loop
        pandemicOver:bool = False
        while not pandemicOver:
            epidemicDuration += 1
            numInfected = 0
            for v in G.nodes:
                if states[v] == "infected":
                    #Check if the infection is over
                    if timeInfected[v] >= 14:
                        states[v] == "recovered"
                    else:
                        numInfected += 1
                        timeInfected[v] += 1
                    
                        #Infect our neighbors
                        neighbors = nx.neighbors(G,v)
                        #only infect susceptible neighbors
                        susceptibleNeighbors = filter(lambda x: states[x]=="susceptible", neighbors)
                        for n in susceptibleNeighbors:
                            if random.random() < 0.1:   #10% chance of infection on each iteration
                                totalInfected += 1
                                states[n] = "infected"
            if numInfected > maxInfected:
                maxInfected = numInfected
            if numInfected == 0:
                pandemicOver = True
        
        maxInfectedSum += maxInfected
        epidemicDurationSum += epidemicDuration
        totalInfectedSum += totalInfected
    
    print(f"Average {name} maxInfected: {maxInfectedSum/100}")
    print(f"Average {name} epidemicDuration: {epidemicDurationSum/100}")
    print(f"Average {name} totalInfected: {totalInfectedSum/100}")

#For deubgging later parts
part3 = True
if part3:
    print("Starting Part 3")
    #Run part 3
    SIRModel(G, "Normal")
    SIRModel(erdosRenyi, "Erdos Renyi")
    SIRModel(configurationModel, "config model")

#Results
#Average Normal maxInfected: 3981.76
#Average Normal epidemicDuration: 38.5
#Average Normal totalInfected: 4012.83

#Average Erdos Renyi maxInfected: 4039.0
#Average Erdos Renyi epidemicDuration: 18.63
#Average Erdos Renyi totalInfected: 4039.0

#Average config model maxInfected: 4011.31
#Average config model epidemicDuration: 29.67
#Average config model totalInfected: 4015.07

#Part 3 discussion
# A) The real graph has the highest total duration and the lowest max and total infected.
#    The Erdos Renyi graph has the lowest duration and the highest max and total infected.
#    The configuration model is in the middle on all 3. It makes sense that there would
#    be an inverse correlation between duration and max infected since the lower number of nodes
#    are infected at a time the longer it will take to spread to remaining nodes, extending the
#    duration of the epidemic.
# B) From a community perspective this makes lots of sense, the config model has lots of high degree nodes connected
#    but still takes time to reach some of the outer nodes. The uniformly distributed edges of the Erdos Renyi model
#    makes it so there are essentially no real communities and the infection is able to spread easily. The real graph
#    has the community structure so makes it difficult for the infection to spread since it is initially largely contained
#    within its community and mist travel one of the few connections between communities to start infecting another one.



# Part 4 ================================================================================

#We ignore lines as graphlets, we expect lots of lines
#Therefore the only 3 node graphlet is the triangle
#4 node graphlets include the 4 cycle, the star, 

#    N
#    | 
#    N---N
def count_3lines(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      for w in G.neighbors(u):
        if v < u and u < w and not G.has_edge(v,w):
          count += 1
  return count

#Triangles
#    N
#    | \
#    N---N
def count_3cycles(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      for w in G.neighbors(u):
        if v < u and u < w and G.has_edge(v,w):
          count += 1
  return count

#    N---N
#    |  
#    N---N 
def count_4lines(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u or G.has_edge(v,w):
          continue
        for x in G.neighbors(w):
          if w < x and not G.has_edge(v,x) and not G.has_edge(x,u):
            count += 1
  return count

#    N---N
#    |   |
#    N---N 
def count_4cycles(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u or G.has_edge(v,w):
          continue
        for x in G.neighbors(w):
          if w < x and G.has_edge(v,x) and not G.has_edge(x,u):
            count += 1
  return count

#    N---N
#    | X |
#    N---N 
def count_4cycleWithX(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u or not G.has_edge(v,w):
          continue
        for x in G.neighbors(w):
          if w < x and G.has_edge(v,x) and G.has_edge(x,u):
            count += 1
  return count

#    N---N
#    | / |
#    N---N 
def count_4cycleWithSlash(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u or G.has_edge(v,w):
          continue
        for x in G.neighbors(w):
          if w < x and G.has_edge(v,x) and G.has_edge(x,u):
            count += 1
  return count

#This version of the star algo was given to us but takes too long to run for me to test against.
#Centeral vertex surounded by 3 disconnected branches
#    N   N
#    | / 
#    N---N 
# def count_4stars(G):
#     count = 0
#     for i,v in enumerate(G.nodes()):
#         if i%100==0:
#             print(f"{i}/{len(G.nodes)}")
#         possible_stars = itertools.permutations(list(G.neighbors(v)),3)
#         for s in possible_stars:
#             if not all(int(s[i]) < int(s[i+1]) for i in range(len(s)-1)):
#                 continue
#             is_star = True
#             edges = itertools.permutations(s, 2)
#             for e in edges:
#                 if G.has_edge(e[0], e[1]):
#                     is_star = False
#                     break
#             if is_star:
#                 count += 1
#     return count

#Centeral vertex surounded by 3 disconnected branches
#    N   N
#    | / 
#    N---N 
def count_4stars(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u:
          continue
        for x in G.neighbors(u):
          if w < x and not G.has_edge(x,w) and not G.has_edge(x,v):
            count += 1
  return count

#This version of the star algo was given to us but takes too long to run for me to test against.
#A star with one connected pair of connected branches
#    N---N
#    | / 
#    N---N 
# def count_4shovels(G):
#     count = 0
#     for v in G.nodes():
#         possible_stars = itertools.permutations(list(G.neighbors(v)),3)
#         for s in possible_stars:
#             if not all(int(s[i]) < int(s[i+1]) for i in range(len(s)-1)):
#                 continue
#             numCons = 0
#             edges = itertools.permutations(s, 2)
#             for e in edges:
#                 if G.has_edge(e[0], e[1]):
#                     numCons += 1
#                     if numCons >= 2:
#                         break
#             if numCons != 1:
#                 count += 1
#     return count

#A star with one connected pair of connected branches
#    N---N
#    | / 
#    N---N 
def count_4shovels(G):
  count = 0
  for v in G.nodes():
    for u in G.neighbors(v):
      if u < v:
        continue
      for w in G.neighbors(u):
        if w < u or not G.has_edge(v,w):
          continue
        for x in G.neighbors(w):
          if w < x and not G.has_edge(u,x) and G.has_edge(w,x) and not G.has_edge(x, v):
            count += 1
  return count

# def computeAll(G):
#     print("3 Lines: " + str(count_3lines(G)))
#     print("3 Cycles: " + str(count_3cycles(G)))
#     print("4 Lines: " + str(count_4lines(G)))
#     print("4 Cycles: " + str(count_4cycles(G)))
#     print("4 Cycle With /: " + str(count_4cycleWithSlash(G)))
#     print("4 Cycle With X: " + str(count_4cycleWithX(G)))
#     print("4 Stars: " + str(count_4stars(G)))
#     print("4 Shovels: " + str(count_4shovels(G)))

print("Computing induced counts for Normal Graphlets")
G3lines = count_3lines(G)
print("3 Lines: " + str(G3lines))
G3cycles = count_3lines(G)
print("3 Cycles: " + str(G3cycles))
G4lines = count_4lines(G)
print("4 Lines: " + str(G4lines))
G4cycles = count_4cycles(G)
print("4 Cycles: " + str(G4cycles))
G4cyclesS = count_4cycleWithSlash(G)
print("4 Cycle With /: " + str(G4cyclesS))
G4cyclesX = count_4cycleWithX(G)
print("4 Cycle With X: " + str(G4cyclesX))
G4stars = count_4stars(G)
print("4 Stars: " + str(G4stars))
G4shovels = count_4shovels(G)
print("4 Shovels: " + str(G4shovels))


print("Computing induced counts for ER Graphlets")
ER3lines = count_3lines(erdosRenyi)
print("3 Lines: " + str(ER3lines))
ER3cycles = count_3lines(erdosRenyi)
print("3 Cycles: " + str(ER3cycles))
ER4lines = count_4lines(erdosRenyi)
print("4 Lines: " + str(ER4lines))
ER4cycles = count_4cycles(erdosRenyi)
print("4 Cycles: " + str(ER4cycles))
ER4cyclesS = count_4cycleWithSlash(erdosRenyi)
print("4 Cycle With /: " + str(ER4cyclesS))
ER4cyclesX = count_4cycleWithX(erdosRenyi)
print("4 Cycle With X: " + str(ER4cyclesX))
ER4stars = count_4stars(erdosRenyi)
print("4 Stars: " + str(ER4stars))
ER4shovels = count_4shovels(erdosRenyi)
print("4 Shovels: " + str(ER4shovels))

print("Computing induced counts for Config model Graphlets")
CM3lines = count_3lines(configurationModel)
print("3 Lines: " + str(CM3lines))
CM3cycles = count_3lines(configurationModel)
print("3 Cycles: " + str(CM3cycles))
CM4lines = count_4lines(configurationModel)
print("4 Lines: " + str(CM4lines))
CM4cycles = count_4cycles(configurationModel)
print("4 Cycles: " + str(CM4cycles))
CM4cyclesS = count_4cycleWithSlash(configurationModel)
print("4 Cycle With /: " + str(CM4cyclesS))
CM4cyclesX = count_4cycleWithX(configurationModel)
print("4 Cycle With X: " + str(CM4cyclesX))
CM4stars = count_4stars(configurationModel)
print("4 Stars: " + str(CM4stars))
CM4shovels = count_4shovels(configurationModel)
print("4 Shovels: " + str(CM4shovels))

def printRes(_3lines, _3cycles, _4lines, _4cycles, _4cyclesS, _4cyclesX, _4stars, _4shovels):
    print("3 Lines: " + str(_3lines))
    print("3 Cycles: " + str(_3cycles))
    print("4 Lines: " + str(_4lines))
    print("4 Cycles: " + str(_4cycles))
    print("4 Cycle With /: " + str(_4cyclesS))
    print("4 Cycle With X: " + str(_4cyclesX))
    print("4 Stars: " + str(_4stars))
    print("4 Shovels: " + str(_4shovels))

print("\nFrequencies with Erdos as base:")
printRes(G3lines/ER3lines, G3cycles/ER3cycles, G4lines/ER4lines, G4cycles/ER4lines, G4cyclesS/ER4cyclesS, G4cycles/ER4cyclesX, G4stars/ER4stars, G4shovels/ER4shovels)

print("\nFrequencies with config model as base:")
printRes(G3lines/CM3lines, G3cycles/CM3cycles, G4lines/CM4lines, G4cycles/CM4lines, G4cyclesS/CM4cyclesS, G4cycles/CM4cyclesX, G4stars/CM4stars, G4shovels/CM4shovels)

#    Part 4 discussion:
# A) Using Erdos Renyi, we identify the motifs we've labeled "4 cycles with /",
#    "4 cycles with X", and "shovels" all to be stong motifs. We have identified
#    "4 cyles" to be an anti-motif, "4 stars" are slightly above average. Using
#    the configuration model, we again find "4 cycles with X" and "4 cycles with /"
#    to be strong motifs, while "4 cycles" and "4 lines" are a anti-motifs.
# B) These motifs make a lot of sense on the real graph. All of our strong motifs are
#    motifs that are very well connected, as we would expect from a community dataset. Our
#    anti-motifs are all the most sparse motifs, cycles without connections, and lines.
#    It would be highly unlikely in a social network to get lots of people in the the
#    4 cycle due to triadic closure, it is much more likely that a 4 cycle becomes a
#    "4 cycle with /" or a "4 cycle with X"
# C) The two core takeaways of "4 cycles with /" and "4 cycles with X" as motifs and
#    "4 cycles" and "4 lines" as anti-motifs is consistent for both random graphs. Others
#    not so much "4 stars" and "4 shovels" go from looking like motifs in Erdos Renyi to not looking
#    like motifs for the configuration model.
# D) When considering how the configuration model and Erdos Renyi random graphs are constructed, we
#    are able to derive an explanation as to why we see the differences for the "4 stars" and "4 shovels"
#    between the relative frequencies for each respective random graph. Since Erdos Renyi randomly distributes
#    edges uniformly, it is much less likely to get high degree vertices hence even mid tier high degree vertices
#    are uncommon but would be very common in a real world graph with a power law degree distribution, leading to
#    high degree motifs have very high frequencies. "4 Cycle With X" and " 4 Cycle With /" are very rare in Erdos Renyi
#    due to the high degree of connectivity required and are hence high freq motifs. This trickles down to mid tier
#    highly connected motifs such as 4 Stars and 4 shovels. Using the configuration model instead, since we preserve
#    degree distributions of our nodes, we get a more accurate picture and see that Stars and Shovels aren't actually
#    motifs.
