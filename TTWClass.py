import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from itertools import cycle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

#%%

class topology:
    def  __init__(self, inputNetwork = None, constant = 1):
        self.networkIndex = None
        self.constant = constant
        self.infrastructure = []
        
        if inputNetwork is not None:
            self.network = inputNetwork
            self.setDistanceMatrix(constant)
            
    def getDistanceMatrix(self, constant = 1, useEuclidean = False, weight = 'WEIGHT', network = None):
        if network is None:
            G = self.network
        else:
            G = network
        
        # The constant term represents the time of travel within the same suburb
        if useEuclidean:
            from scipy.spatial import distance
            
            if self.networkIndex is None:
                self.networkIndex = sorted(G.nodes())
            coords = [(G.node[n]['XCOR'], G.node[n]['YCOR'])  for n in self.networkIndex]

            distances = distance.cdist(coords, coords)
        else:
            if self.networkIndex is None:
                distances = pd.DataFrame(dict(nx.shortest_path_length(G, weight = weight))).as_matrix()
            else:
                distances = pd.DataFrame(dict(nx.shortest_path_length(G, weight = weight)),\
                                         columns = self.networkIndex, index = self.networkIndex).as_matrix()
        return distances + constant
    
    def setDistanceMatrix(self, constant = None, useEuclidean = False):
        if constant is None:
            constant = self.constant
        self.distances = self.getDistanceMatrix(constant, useEuclidean)
    
    def getDistanceMatrixWithCongestion(self, TTW, A = 1, B = 1, c_e = 10, network = None):
        if network is None:
            G = self.network
        else:
            G = network
        indexToNode = self.indexToNode
        weights = nx.get_edge_attributes(G, 'WEIGHT')
        nx.set_edge_attributes(G, weights, 'Congested_Weight')

        HHSplit = [0.4, 0.2, 0.2, 0.2]
        remainingTTWArray = TTW
        
        roadUsage = defaultdict(int)
        for i in range(len(HHSplit)):
            TTWPart = np.ceil(remainingTTWArray * HHSplit[i]).astype(int)
            remainingTTWArray = remainingTTWArray - TTWPart
            targets = np.flatnonzero(np.sum(remainingTTWArray, axis = 0))
            for target in targets:
                sources = np.flatnonzero(remainingTTWArray[:,target])
                for source in sources:
                    shortest_paths = nx.all_shortest_paths(G,source=indexToNode[source],target=indexToNode[target], weight = 'Congested_Weight')
                    shortest_paths = cycle(shortest_paths)
                    
                    for HH in range(TTWPart[source][target] + 1):
                        shortest_path = next(shortest_paths)
                        edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path) - 1)]
                        for edge in edges:
                            roadUsage[edge] += 1
            
            for edge, usage in roadUsage.items():
                freeFlowingWeight = G.edges()[edge]['WEIGHT']  #weights[edge]  somehow doesn't work. CHECK
                weightMultiplier = 1 + A * (usage / c_e) ** B
                G.edges()[edge]['Congested_Weight'] = freeFlowingWeight * weightMultiplier
        
        return self.getDistanceMatrix(constant = self.constant, useEuclidean = False, weight = 'Congested_Weight', network = network)
        
    
    def removeRandomInfrastructure(self, numberToRemove = 1, breakEdge = False, weightRatio = 50):
        roads = self.infrastructure
        numberOfRoads = len(roads)
        if numberToRemove > numberOfRoads:
            roadsToRemove = roads
        else:
            IDofRoadsToRemove = np.random.choice(range(numberOfRoads), size = numberToRemove, replace = False)
            roadsToRemove = [roads[ID] for ID in IDofRoadsToRemove]
        
        for road in roadsToRemove:
            if breakEdge:
                self.network.remove_edge(*road)
            else:
                self.network[road[0]][road[1]]['WEIGHT'] *= weightRatio
            self.infrastructure.remove(road)
        
        self.setDistanceMatrix()
        
    def visualise(self, ax = None, cax = None, householdsPerSuburb = None, workDistribution = None,\
                  showInfrastructure = False, logScale = False, saveName = None, complete = True,\
                  maxHouseholds = None, discreteColour = None, fig = None):
        G = self.network
        
        if ax is None:
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
        
        if self.networkIndex is None:
            networkIndex = sorted(G.nodes())
        else:
            networkIndex = self.networkIndex
        
        ypos = nx.get_node_attributes(G, 'YCOR')
        pos = {key: (value, ypos[key])  for (key, value) in nx.get_node_attributes(G, 'XCOR').items()}
        if len(ypos) == 0:  # 'XCOR' and 'YCOR' does not exist. Need to calculate positions using spring algorithm
            iterations = max(500, G.number_of_edges())
            pos = pos = nx.spring_layout(G, iterations = iterations, k = 1)

        
        if householdsPerSuburb is not None:
            nx.draw_networkx_edges(G, pos = pos, ax = ax, alpha = 0.7, width = 0.2)
            
            if workDistribution is not None:
                nodelist = [node for i, node in enumerate(G.nodes()) if workDistribution[i] != 0]
                nx.draw_networkx_nodes(G, pos = pos, ax = ax, nodelist = nodelist,\
                                       node_size = 20, color = 'red', linewidths = 0, alpha = 0.5)
                
            if logScale:
                householdsPerSuburb = np.log(householdsPerSuburb, where = householdsPerSuburb != 0)
            households_df = pd.Series(householdsPerSuburb, index = networkIndex)
            sortedDistribution = households_df.loc[list(G.nodes())]
            
            nx.draw_networkx_nodes(G, pos = pos, ax = ax, node_size = 10, linewidths = 0.2, \
                                   node_color = sortedDistribution, cmap = 'Greens')
            
        else:
            nx.draw(G, pos = pos, ax=ax, node_size = 10, width = 0.5, alpha = 1, linewidths = 0.2)
        
        if complete:
            self.completeVisualisation(ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds, discreteColour)

    def completeVisualisation(self, ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds = None, discreteColour = None, fig = None):
        cmp = plt.get_cmap('Greens')  # Colour map. Also can use 'jet', 'brg', 'rainbow', 'winter', etc
        
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        
#        if logScale:
#            householdsPerSuburb = np.log(householdsPerSuburb)  # Repeated. Simplify
        
        if householdsPerSuburb is not None:
            if maxHouseholds is None:
                maxHouseholds = np.max(householdsPerSuburb)
            if discreteColour is None:
                if maxHouseholds != int(maxHouseholds) or maxHouseholds == 1:  # The BoundaryNorm method invokes an error when maxHouseholds is 1, since it leads to a divide by 0 error
                    discreteColour = False
                else:
                    discreteColour = True
            if fig is None:
                fig = plt  # Just used for the pyqt GUI so far, to prevent another figure from popping up when plt is called
            if not discreteColour:
                sm = plt.cm.ScalarMappable(cmap=cmp,norm=plt.Normalize(0, maxHouseholds))
                sm._A = []
                if logScale:
                    fig.colorbar(sm, cax=cax, label = 'Number of households (Log)')
                else:
                    fig.colorbar(sm, cax=cax, label = 'Number of households')
            else:
                norm = matplotlib.colors.BoundaryNorm(np.arange(0, maxHouseholds + 1), cmp.N)  # For discrete color scale
                sm = plt.cm.ScalarMappable(cmap=cmp,norm=norm)
                sm._A = []  # Setting up an empty array in the scalar mappable
                fig.colorbar(sm, cax=cax, label = 'Number of households')
        
        if saveName is not None:
            plt.savefig(saveName + '.png', dpi = 250, format = 'png', bbox_inches = 'tight')
            plt.close()
        else:
            plt.rcParams['savefig.dpi'] = 90
#            plt.tight_layout()
#            plt.show()
#            plt.close()

    def visualiseWithPatches(self, ax, patchCoords, householdsPerSuburb, workDistribution, showInfrastructure, logScale, maxHouseholds):
        G = self.network
        nodes = sorted(G.nodes())
        
        if householdsPerSuburb is not None:
            cmp = plt.get_cmap('Greens')  # Colour map. Also can use 'jet', 'brg', 'rainbow', 'winter', etc
            if logScale:
                householdsPerSuburb = np.log(householdsPerSuburb)
            
            if maxHouseholds is None:
                maxHouseholds = max(householdsPerSuburb)
            householdsPerSuburb = householdsPerSuburb / maxHouseholds  # Normalise
        
        for i in range(len(nodes)):
            centreX = G.node[nodes[i]]['XCOR']
            centreY = G.node[nodes[i]]['YCOR']
            
            xCoords, yCoords = patchCoords(centreX, centreY)
            
            if householdsPerSuburb is not None:
                track = list(zip(xCoords, yCoords))
                patch = plt.Polygon(track, ec = 'grey', fc = cmp(householdsPerSuburb[i]),\
                                    antialiased = True, linewidth = 0.2)
                ax.add_patch(patch)
            else:
                ax.add_line(matplotlib.lines.Line2D(xCoords + [xCoords[0]], yCoords + [yCoords[0]],\
                                                    color = 'grey', antialiased = True, linewidth = 0.2))
            
            if workDistribution is not None:
                normalisedWorkDistribution = workDistribution / np.max(workDistribution)  # Redundant to repeat
                if workDistribution[i] != 0:
                    ax.add_line(matplotlib.lines.Line2D(xCoords + [xCoords[0]], yCoords + [yCoords[0]],\
                                                        color = 'red', alpha = normalisedWorkDistribution[i]))

        if showInfrastructure:
            edges = self.infrastructure
            edgeWeightDict = nx.get_edge_attributes(G, 'WEIGHT')
            edgeWeights = []
            for node1, node2 in edges:
                try:
                    edgeWeights.append(edgeWeightDict[(node1,node2)])
                except KeyError:
                    edgeWeights.append(edgeWeightDict[(node2,node1)])
            maxWeight = np.max(edgeWeights)
            minWeight = np.min(edgeWeights)
#            edgeWeightsNorm = np.array(edgeWeights) / maxWeight
#            edgeWeightsNorm = np.array(edgeWeights) / self.weight
            edgeWeightsNorm = minWeight / np.array(edgeWeights)
            edgeWeightSTD = np.std(edgeWeightsNorm)
            for i, edge in enumerate(edges):
                startNode = edge[0]
                endNode = edge[1]
                startNodeX = G.node[startNode]['XCOR']
                endNodeX = G.node[endNode]['XCOR']
                startNodeY = G.node[startNode]['YCOR']
                endNodeY = G.node[endNode]['YCOR']
                ax.add_line(matplotlib.lines.Line2D((startNodeX, endNodeX),(startNodeY, endNodeY),\
                                                    color = 'blue', linewidth = 0.7,\
                                                    alpha = 0.4 * (edgeWeightsNorm[i])))  # Can add the edgeWeightSTD to the base of 0.4

    def getInfrastructureNetwork(self):
        G = nx.Graph()
        G.add_edges_from(t.infrastructure)
        return G
        
    
    def createLatticeInfrastructure(self):
        self.infrastructure = list(self.network.edges())
        
    def randomiseWeightOfInfrastructure(self, k = None, theta = None):
        infrastructure = self.infrastructure
        G = self.network
        
        if k is None:
            k = self.weight / theta
        if theta is None:
            theta = self.weight / k
        
        self.theta = theta  # Store for later reference
        self.k = k
        
        weightDict = dict(zip(infrastructure, np.random.gamma(shape = k, scale = theta, size = len(infrastructure))))
        
        nx.set_edge_attributes(G, weightDict, name = 'WEIGHT')
        
        self.setDistanceMatrix()
    
    def getinfrastructureAccessibleNodes(self, seed):
        G = self.network
        
        try:
            accessibleNodes = self.infrastructureAccessibleNodes[seed]
        except AttributeError:
            accessibleNodes = list(G.neighbors(seed))
            self.infrastructureAccessibleNodes = {seed: accessibleNodes}
        except KeyError: 
            accessibleNodes = list(G.neighbors(seed))
            self.infrastructureAccessibleNodes[seed] = accessibleNodes

        accessibleNodesSet = set(accessibleNodes)
        return accessibleNodes, accessibleNodesSet

    def createPreferentialAttachmentInfrastructure(self, seed = '(  0,  0)', linksToCreate = 50, weightRatio = 0.1, shortRangePreference = 3):
        G = self.network
        
        newInfrastructure = []
        linksCreated = 0

        accessibleNodes, accessibleNodesSet = self.getinfrastructureAccessibleNodes(seed)

        while linksCreated < linksToCreate:
            numberOfExistingLinks = len(newInfrastructure)
            
            if numberOfExistingLinks == 0:
                node1 = seed
            else:
                selectExistingLink = newInfrastructure[np.random.randint(numberOfExistingLinks)]
                node1 = selectExistingLink[round(np.random.random())]  # Select one of the two ends
            
            node2 = accessibleNodes[np.random.randint(len(accessibleNodes))]
            
            try:
                if G[node1][node2]['WEIGHT'] == self.weight:
                    G[node1][node2]['WEIGHT'] *= weightRatio
                    newInfrastructure.append((node1, node2))
                    linksCreated += 1
                for newNeighbour in G.neighbors(node2):  # Add neighbours even if weight doesn't equal self.weight so that it's not possible to get trapped as a new seed by existing infrastructure and loop forever
                    if newNeighbour not in accessibleNodesSet:
                        accessibleNodes.append(newNeighbour)
                        accessibleNodesSet.add(newNeighbour)
            except KeyError:
                if node1 != node2:
                    P = 1 / ((G.node[node1]['XCOR'] - G.node[node2]['XCOR']) ** 2 + (G.node[node1]['YCOR'] - G.node[node2]['YCOR']) ** 2) ** (0.5 * shortRangePreference)
                    if np.random.random() < P:
                        G.add_edge(node1, node2, WEIGHT = self.weight * weightRatio)
                        newInfrastructure.append((node1, node2))
                        linksCreated += 1
                        for newNeighbour in G.neighbors(node2):
                            if newNeighbour not in accessibleNodesSet:
                                accessibleNodes.append(newNeighbour)
                                accessibleNodesSet.add(newNeighbour)
        
#        for i in sorted(newInfrastructure):
#            print(i)
        self.infrastructure += newInfrastructure
        self.setDistanceMatrix()
        
    def growOptimalNetwork(self, householdDistribution, globalEnergyFunction, seed = '(  0,  0)', weightRatio = 0.1, congestion = False, TTW = None):
        G = self.network
        
        try:
            accessibleNodes = self.infrastructureAccessibleNodes[seed]
        except AttributeError:
            accessibleNodes = [seed]
            self.infrastructureAccessibleNodes = {seed: accessibleNodes}
        except KeyError: 
            accessibleNodes = [seed]
            self.infrastructureAccessibleNodes[seed] = accessibleNodes
        print(accessibleNodes)
        accessibleNodesSet = set(accessibleNodes)
        
        energies = np.zeros((len(accessibleNodes), 4))  # 6 for hexagonal grid
        energies.fill(np.nan)
        
        checked = set()  # Avoid checking each node-neighbour pair twice
        
        for i, node in enumerate(accessibleNodes):
#            print(i)
            nodeNeighbours = list(G.neighbors(node))
            nodeNeighbours.sort()
            for j, neighbour in enumerate(nodeNeighbours):
#                print(j)
                if G[node][neighbour]['WEIGHT'] == self.weight and neighbour not in checked:
                    G2 = G.copy()
                    G2[node][neighbour]['WEIGHT'] = weightRatio  # Only constructing infrastructure once
                    if congestion:
                        distances = self.getDistanceMatrixWithCongestion(TTW, A = 1, B = 1, c_e = 10, network = G2)
                    else:
#                        distances = pd.DataFrame(dict(nx.shortest_path_length(G2, weight = 'WEIGHT'))).as_matrix() + self.constant
                        distances = self.getDistanceMatrix(constant = self.constant, useEuclidean = False, weight = 'WEIGHT', network = G2)
                    e = globalEnergyFunction(householdDistribution, distances)
                    energies[i][j] = np.sum(e)
            checked.add(node)
        
        index = np.nanargmin(energies)  # Flattens array if no axis is provided
        index_i = index // 4  # 6 for hexagonal grid
        index_j = index % 4
        print(energies)
        print("i", index_i)
        print("j", index_j)
        
        node1 = accessibleNodes[index_i]
        nodeNeighbours = list(G.neighbors(node1))
        nodeNeighbours.sort()
        node2 = nodeNeighbours[index_j]
        
        G[node1][node2]['WEIGHT'] = weightRatio  # Doing *= seems to cause the wrong edge to be updated sometimes? Using = means the infrastructure update can only be applied once
        
        if node2 not in accessibleNodesSet:
            accessibleNodes.append(node2)
            accessibleNodesSet.add(node2)
        
        self.infrastructure.append((node1, node2))
        print((node1,node2))
        self.setDistanceMatrix()
        
    def save(self, saveName):
        with open(saveName + '.pkl', 'wb') as f:
            pickle.dump(self, f)

class sydneyTopology(topology):
    def __init__(self, constant = 1, SA2 = None, fullyConnected = False):
        super().__init__()
        
        if fullyConnected:
            weights = pd.read_csv('../GIS/Processing/SA2DriveTimes-OSRM.csv', header = None)
            weights = (weights + weights.transpose()) / 2 / 60  # Make symmetric and change seconds to minutes
            G = nx.Graph()
            for i, row in weights.iterrows():
                for j, value in enumerate(row):
                    if j < i: continue
                    G.add_edge(i, j, WEIGHT = value)
            
            G = nx.relabel_nodes(G, pd.read_csv('../GIS/Input/SA2Names.txt', squeeze = True).to_dict())
        else:
            G = nx.read_gml('../GIS/Processing/Neighbouring Suburbs.gml')
            
        if SA2 is not None:
            G = G.subgraph(SA2)
        else:
            SA2 = pd.read_csv('../GIS/Input/SA2Names.txt', squeeze = True)
            
        self.network = G
        self.networkIndex = SA2
        
        self.setDistanceMatrix(constant)
        self.indexToNode = dict(zip(range(len(SA2)), SA2))  # Save dictionary of the node names vs their index


class hexagonalTopology(topology):
    def __init__(self, layers, weight = 1, constant = 1):
        super().__init__()
        
        self.network = self.createHexagonalTopology(layers, weight)
        self.setDistanceMatrix(constant)
        self.layers = layers
        self.weight = weight

    def createHexagonalTopology(self, layers, weight):
        # Creates a hexagonally arranged network of a specified number of layers deep
        # Each node is connected to 6 neighbours
        # This network is used to calculate the shortest path between each node
        # The numpy array returned is arranged layer by layer, without labels. The first entry is suburb (0,0), followed by 6 suburbs for layer 1, 12 suburbs for layer 2, etc
        # The formatting {:>3} is to write numbers with up to 2 blank spaces in front, so that they would be arranged in the correct order and the network is connected correctly
        # {:>2} works fine for up to 17 layers
        G = nx.Graph()
        # Connect network one layer at a time
        for layer in range(layers):
            for i in range(6 * layer):
                if i % layer == 0:
                    G.add_edge("({:>3},{:>3})".format(layer,i), "({:>3},{:>3})".format(layer-1,np.floor(i/layer*(layer-1)).astype(int)),\
                                 WEIGHT = weight)  
                else:    
                    G.add_edge("({:>3},{:>3})".format(layer,i), "({:>3},{:>3})".format(layer-1,np.floor(i/layer*(layer-1)).astype(int)),\
                                 WEIGHT = weight)
                    G.add_edge("({:>3},{:>3})".format(layer,i), "({:>3},{:>3})".format(layer-1,np.floor((i+1)%(layer*6)/layer*(layer-1)).astype(int)),\
                                 WEIGHT = weight)
                G.add_edge("({:>3},{:>3})".format(layer,i), "({:>3},{:>3})".format(layer,(i+1)%(layer*6)),
                             WEIGHT = weight)
        
        ## Determine x and y coordinate of the centre (used for the Euclidean distance and visualisation)
        nodes = sorted(G.nodes())
        self.indexToNode = dict(zip(range(len(nodes)), nodes))  # Save dictionary of the node names vs their index
        self.nodeToIndex = dict(zip(nodes, range(len(nodes))))
        layer = 0
        position = 0
        centresX = centresY = [0]
        for i in range(len(nodes)):
            G.node[nodes[i]]['XCOR'] = centresX[position] * weight
            G.node[nodes[i]]['YCOR'] = centresY[position] * weight
            if position >= layer * 6 - 1:
                # For each new layer, determine the (x,y) coodinates of the centres
                layer += 1
                position = 0
                centresX = [i for i in range(layer + 1)]
                centresX = centresX + [centresX[-1] for i in range(layer)]
                centresX = centresX + [centresX[-1] - i for i in range(1,2*layer + 1)]
                centresX = centresX + [centresX[-1] for i in range(layer)]
                centresX = centresX + [centresX[-1] + i for i in range(1,layer)]
                centresY = [layer - 0.5 * i for i in range(layer + 1)]
                centresY = centresY + [centresY[-1] -i for i in range(1,layer+1)]
                centresY = centresY + [centresY[-1] -0.5 * i for i in range(1,layer+1)]
                centresY = centresY + centresY[-2:0:-1]
            else:
                position += 1
        
        return G
    
    def visualise(self, ax = None, cax = None, householdsPerSuburb = None, workDistribution = None,\
                  showInfrastructure = True, logScale = False, saveName = None, complete = True, \
                  maxHouseholds = None, discreteColour = None, fig = None):               
        maxX = (self.layers) * self.weight        
        maxY = (self.layers - 0.5) * self.weight
        minX = -maxX
        minY = -maxY
        
        if ax is None:
#            ax = plt.axes([minX, minY - 0.1, maxX - minX, maxY -minY])
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
        
        def patchCoords(centreX, centreY):
            s = self.weight
            xCoords = [centreX + s * 2/3, centreX + s/3, centreX - s/3,\
                       centreX - s * 2/3, centreX - s/3, centreX + s/3]			
            yCoords = [centreY, centreY + s/2, centreY + s/2, centreY, \
                       centreY - s/2, centreY - s/2]
            
            return xCoords, yCoords
        
        self.visualiseWithPatches(ax, patchCoords, householdsPerSuburb, workDistribution, showInfrastructure, logScale, maxHouseholds)

        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY - 0.1, maxY + 0.1)

        if complete:
            self.completeVisualisation(ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds, discreteColour, fig)
    

class gridTopology(topology):
    def __init__(self, layers = 20, weight = 1, constant = 1, diagonals = True):
        super().__init__()
        
        self.network = self.createGridTopology(layers, weight, diagonals)
        
        self.setDistanceMatrix(constant)
        self.layers = layers
        self.weight = weight
        self.diagonals = diagonals

    def createGridTopology(self, layers, weight, diagonals):
        G = nx.Graph()
        
        for layer in range(layers):
            for i in range(layers):
                if i != layers - 1:  # Exclude the end
                    G.add_edge("({:>3},{:>3})".format(layer, i), "({:>3},{:>3})".format(layer, i + 1),\
                                 WEIGHT = weight)
                    
                    if diagonals:
                        if layer != 0:  # Diagonal connections
                            G.add_edge("({:>3},{:>3})".format(layer, i), "({:>3},{:>3})".format(layer - 1, i + 1),\
                                     WEIGHT = np.sqrt(2 * weight ** 2))
                            
                        if layer != layers -1:
                            G.add_edge("({:>3},{:>3})".format(layer, i), "({:>3},{:>3})".format(layer + 1, i + 1),\
                                     WEIGHT = np.sqrt(2 * weight ** 2))
                        
                if layer != layers - 1:
                    G.add_edge("({:>3},{:>3})".format(layer, i), "({:>3},{:>3})".format(layer + 1, i),\
                                 WEIGHT = weight)
                    
        ## Determine x and y coordinate
        nodes = sorted(G.nodes())  # Sorted nodes are arranged by x's (like scanning up and across column by column)
        self.indexToNode = dict(zip(range(len(nodes)), nodes))  # Save dictionary of the node names vs their index
        self.nodeToIndex = dict(zip(nodes, range(len(nodes))))
        for i in range(len(nodes)):
            G.node[nodes[i]]['XCOR'] = (i // layers + 0.5) * weight
            G.node[nodes[i]]['YCOR'] = (i % layers + 0.5) * weight
        
        return G        

    def visualise(self, ax = None, cax = None, householdsPerSuburb = None, workDistribution = None,\
                  showInfrastructure = True, logScale = False, saveName = None, complete = True,\
                  maxHouseholds = None, discreteColour = None, fig = None):
        
        if ax is None:
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')

        def patchCoords(centreX, centreY):
            halfSide = self.weight / 2
            xCoords = [centreX - halfSide, centreX + halfSide, centreX + halfSide, centreX - halfSide]
            yCoords = [centreY - halfSide, centreY - halfSide, centreY + halfSide, centreY + halfSide]
            
            return xCoords, yCoords

        self.visualiseWithPatches(ax, patchCoords, householdsPerSuburb, workDistribution, showInfrastructure, logScale, maxHouseholds)
        
        ax.set_xlim(-0.1, self.layers * self.weight + 0.1)
        ax.set_ylim(-0.1, self.layers * self.weight + 0.1)
        
        if complete:
            self.completeVisualisation(ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds, discreteColour, fig)
        


class urbanStreet(gridTopology):
    def __init__(self, layers = 20, weight = 1, constant = 1, diagonals = True, numberOfNodes = 50):
        import BuildInfrastructure
        
        cities, segments, segmentsByID = BuildInfrastructure.build(layers * weight, numberOfNodes, normalDist = True)

        segmentGraph = BuildInfrastructure.getSegmentGraph(cities, segmentsByID, layers * weight / 100)
        
        cities['Node'] = cities.apply(lambda c: "({:>3},{:>3})".format(int(c['x'] / weight),int(c['y'] / weight)), axis = 1)
        
        segmentNameSwapDict = cities.set_index('closestConnection')['Node'].to_dict()  
        
        nx.relabel_nodes(segmentGraph, segmentNameSwapDict, copy = False)
        self.segmentGraph = segmentGraph
        self.segments = segments
        self.cities = cities

        super().__init__(layers, weight, constant, diagonals)
#        self.infrastructure = self.segmentGraph.edges()
        
#        self.networkIndex = None
#        self.network = self.createGridTopology(layers, weight, diagonals)
#        
#        self.distances = self.getDistanceMatrix(constant)
#        self.layers = layers
#        self.weight = weight
#        self.diagonals = diagonals
        

    def visualise(self, ax = None, cax = None, householdsPerSuburb = None, workDistribution = None,\
                  showInfrastructure = True, logScale = False, saveName = None, complete = True,\
                  maxHouseholds = None, discreteColour = None, fig = None):
        if ax is None:
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
        
        super().visualise(ax, cax, householdsPerSuburb, workDistribution, showInfrastructure, logScale, saveName, complete = False)
        
        for segment in self.segments:
            ax.plot(*segment, color = 'red')
        ax.plot(self.cities['x'], self.cities['y'], 'ko', markersize = 3)
        
        self.completeVisualisation(ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds, discreteColour, fig)
        
        
    def getDistanceMatrix(self, constant = 1, useEuclidean = False):
        networkIndex = sorted(self.network.nodes())
        
        combNetwork = nx.compose(self.segmentGraph, self.network)

#        distances = pd.DataFrame(dict(nx.shortest_path_length(combNetwork, weight = 'WEIGHT')),\
#                                         columns = networkIndex, index = networkIndex)
        distances = pd.DataFrame.from_dict(dict(nx.shortest_path_length(combNetwork, weight = 'WEIGHT'))).loc[networkIndex][networkIndex]
        distances = distances.as_matrix()
        return distances + constant

class oneDimTopology(gridTopology):
    def __init__(self, layers = 20, weight = 1, constant = 1):
        super().__init__(constant = constant)
        
        self.network = self.create1DTopology(layers, weight)
        
        self.setDistanceMatrix(constant)
        self.layers = layers
        self.weight = weight

    def create1DTopology(self, layers, weight):
        G = nx.Graph()
        
        for layer in range(layers):
            if layer != layers - 1:  # Exclude the end
                G.add_edge("({:>3})".format(layer), "({:>3})".format(layer + 1), WEIGHT = weight)
                    
        ## Determine x and y coordinate
        nodes = sorted(G.nodes())
        self.indexToNode = dict(zip(range(len(nodes)), nodes))  # Save dictionary of the node names vs their index
        self.nodeToIndex = dict(zip(nodes, range(len(nodes))))
        for i in range(len(nodes)):
            G.node[nodes[i]]['XCOR'] = (i + 0.5) * weight
            G.node[nodes[i]]['YCOR'] = (0.5) * weight
        
        return G
    
    def visualise(self, ax = None, cax = None, householdsPerSuburb = None, workDistribution = None,\
                  showInfrastructure = True, logScale = False, saveName = None, complete = True,\
                  maxHouseholds = None, discreteColour = None, fig = None):
        
        if ax is None:
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
        
        
        super().visualise(ax = ax, cax = cax, householdsPerSuburb = householdsPerSuburb, workDistribution = workDistribution,\
                          showInfrastructure = showInfrastructure, logScale = logScale, saveName = saveName, complete = False,\
                          maxHouseholds = maxHouseholds, discreteColour = discreteColour, fig = fig)
        
        ax.set_ylim(-0.25 * self.layers * self.weight, 0.25 * self.layers * self.weight)
        
        if complete:
            self.completeVisualisation(ax, cax, householdsPerSuburb, logScale, saveName, maxHouseholds, discreteColour, fig) 
    
    

#%%

class TTW:
    # This class takes as input distances between suburbs and the distribution of work and can simulate
    # a TTW matrix using Monte Carlo, by growing the suburbs through the addition of households and/by
    # the relocation of existing households between suburbs
    # It cas also be created directly from an input TTW matrix (such as from census data)
    # Maximum likelihood calculations can also be performed by accepting input parameters
    def __init__(self, networkTopology, workDistribution = None, TTWArray = None):
        self.topology = networkTopology
        distances = networkTopology.distances
        self.arraySize = len(distances)
        
       
        if TTWArray is None:
            if workDistribution is not None:
                assert abs(np.sum(workDistribution) - 1) < 1E-5  # workDistribution should sum to 1 (as a probability distribution)
            self.households = pd.DataFrame(columns = [0,1], dtype = int)  # Columns are home suburb & work suburb
            self.householdsPerSuburb = np.zeros(self.arraySize)
            self.workPerSuburb = np.zeros(self.arraySize)
            if workDistribution is not None and len(workDistribution) < self.arraySize:
                # If more suburbs are provided than the size of the work distribution list, the remaining suburbs are given as having work distribution of 0
                self.workDistribution = np.append(workDistribution, [0 for i in range(len(workDistribution), len(distances))])
            else:
                self.workDistribution = workDistribution
        else:
            # If TTWArray is provided, then households are created according to the matrix
            totalHouseholds = np.sum(TTWArray)
            householdsHome = []
            householdsWork = []
            for i,row in enumerate(TTWArray):
                for j,n in enumerate(row):
                    householdsHome += [i]*n
                    householdsWork += [j]*n
            self.households = pd.DataFrame()
            self.households[0] = householdsHome
            self.households[1] = householdsWork
            self.getTTWArray()
            self.workDistribution = np.sum(TTWArray, 0) / totalHouseholds

        # Paramaters used in the Monte Carlo simulation. Overridden by the inputs for the maximum likelihood calculation  
        self.parameters = {"c_p": 1/6,  # r = 0.5
                           "c_w": 17.5,
                           "c_l": 1,
                           "c_a": 1,  # Firm agglomeration multiplier
                           "alpha_p": 1,
                           "alpha_w": 1,
                           "alpha_l": 2,  # 2
                           "alpha_a": 1,  # Agglomeration potentential parameter
                           "beta": 5,
                           "cell_capacity": 50,
                           "fixed_energy": 6,
                           "c_d": 1}  # Distance multiplier
       
    def initialiseRandomHouseholds(self, numberOfHouseholds, exactWorkDistribution = True):
        # Initialise a given number of households according to the work distribution
        # The home suburb is selected at random
        # If the work distribution is None, then the work suburb is also selected at random
        # Otherwise, if exactWorkDistribution = True, the work distribution generated each time will be identical, else it is drawn probabilistically
        self.households[0] = np.random.choice(self.arraySize, numberOfHouseholds)        
        
        workDistribution = self.workDistribution        
        if workDistribution is None:
            # Work suburb drawn randomly. Simulates the case where each household agent is associated with its
            # own work agent, and each is able to relocate
            self.households[1] = np.random.choice(self.arraySize, numberOfHouseholds)
            
        elif exactWorkDistribution:
            workPerSuburb = np.round(workDistribution * numberOfHouseholds).astype(int)
            while np.sum(workPerSuburb) != numberOfHouseholds:
                # If there are differences due to rounding, adjust to make the total number of households equal the specified amount by adjusting the maximum value (usually at the CBD)
                maxSuburb = np.argmax(workPerSuburb)
                if np.sum(workPerSuburb) < numberOfHouseholds:
                    workPerSuburb[maxSuburb] += 1  # Add 1
                if np.sum(workPerSuburb) > numberOfHouseholds:
                    workPerSuburb[maxSuburb] -= 1  # Subtract 1
            householdsWork = []
            for j,n in enumerate(workPerSuburb):
                householdsWork += [j]*n
            self.households[1] = householdsWork
        else:  # Draw work suburb probabilistically
            self.households[1] = np.random.choice(self.arraySize, numberOfHouseholds, p = workDistribution)
        
        self.getTTWArray()
    
    def randomiseHomeSuburb(self):
        # Can be used to rerun relocation simulation with households having the same work suburbs (partially redundant since the assignment of work to households can now be non-probabilistic)
        self.households[0] = np.random.choice(self.arraySize, len(self.households))
        self.getTTWArray()

    def randomiseWorkSuburb(self):
        self.households[1] = np.random.choice(self.arraySize, len(self.households))
        self.getTTWArray()
    
    def randomiseEdges(self):
        # Shuffle the edges between household home suburb and work suburb, while maintaining the total distributions
        workSuburbs = self.households[1].values
        np.random.shuffle(workSuburbs)
        self.households[1] = workSuburbs
        self.getTTWArray()
        
    def getParameters(self, parameters = None):
        if parameters is None:
            parameters = self.parameters
        elif len(parameters) < len(self.parameters):
            for key, value in self.parameters.items():
                parameters.setdefault(key, value)
        
        return parameters

    
    def energyOfSingleHousehold(self, homeSuburb = None, workSuburb = None, householdsPerSuburb = None, congestion = None, params = None):
        # Calculates the energy of a single household according to the three components of 
        # attraction to other households, attraction to work and local repulsion (overcrowding)
        # homeSuburb and workSuburb can be single integers or None
        # In the case of None, the full array is returned
        
		 # Set up parameters
        # Supplied parameters (eg from Maximum Likelihood) replace those taken from the initially set values
        params = self.getParameters(params)

        if congestion is None:
            distances = self.topology.distances * params['c_d']
        elif congestion:  # Congestion = True. Recalculate congestion this round
            distances = self.topology.getDistanceMatrixWithCongestion(self.TTWArray)
            self.congestionDistances = distances
        else:  # Congestion = False. Use previously calculated congestion this round
            distances = self.congestionDistances
            
        if householdsPerSuburb is None:
            householdsPerSuburb = self.householdsPerSuburb
        
        
        # Three components of energy
        household_energy = householdsPerSuburb / (distances[homeSuburb] ** params['alpha_p'])
        household_energy = -params['c_p'] * np.sum(household_energy, axis = -1)  # Use nansum since if a suburb is disconnected the distances to/from that suburb is nan

        if self.workDistribution is None:
            localOccupants = householdsPerSuburb[homeSuburb] + self.workPerSuburb[homeSuburb]
        else:
            localOccupants = householdsPerSuburb[homeSuburb]
        
#        local_energy = params['c_l'] * (localOccupants ** params['alpha_l'] /  ## Changed from 2
#                                params['cell_capacity'] + params['fixed_energy'])
        
        # Local energy based on Gaussian kernel. 'alpha_l' plays the role of the standard deviation
        workDistribution = np.atleast_2d(self.workDistribution)
#        workDistribution = np.zeros_like(self.workDistribution)  # Setting up a Gaussian centred on only the CBD
#        workDistribution[np.argmax(self.workDistribution)] = 1
        
        local_energy = workDistribution[homeSuburb][workSuburb] * np.exp(-distances[homeSuburb][workSuburb] ** 2 / (2 * params['alpha_l'] ** 2))
        local_energy = np.sum(local_energy, axis = -1)
        local_energy = params.get('local_energy', local_energy)  # For maximum likelihood of all individual local cost values
        local_energy = params['c_l'] * local_energy
        # CBD index = 72 after filtering the suburbs without rent data
        
        work_energy = -params['c_w'] / (distances[homeSuburb][workSuburb] ** params['alpha_w'])
#        if np.any(np.isnan(work_energy)):
#            if type(work_energy) == float:
#                work_energy = 0  # Replace nan with 0, since this means that one of the two suburbs is disconnected
#            else:
#                work_energy[np.isnan(work_energy)] = 0
        
        if homeSuburb is None or workSuburb is None:
            # Change into columns
            household_energy = household_energy.reshape((-1,1))
            local_energy = local_energy.reshape((-1,1))
            work_energy = work_energy.reshape((self.arraySize if homeSuburb is None else homeSuburb,
                                               self.arraySize if workSuburb is None else workSuburb))

            
#        household_energy = -params['c_p'] * np.sum(householdsPerSuburb / (distances ** params['alpha_p']) , 1, keepdims = True)
#        local_energy = params['c_l'] * (householdsPerSuburb.reshape(-1,1) ** params['alpha_l'])
#        work_energy = -params['c_w'] / (distances ** params['alpha_w'])
        return household_energy + local_energy + work_energy

    def energyChangeGlobal_newHousehold(self, homeSuburb, workSuburb, householdsPerSuburb, params = None):
        # Calculates the change in total energy across all households between the current state and
        # one more household added at the specified location
        
        # Set up parameters
        # Supplied parameters (eg from Maximum Likelihood) replace those taken from the initially set values
        
        params = self.getParameters(params)
        
        distances = self.topology.distances  # What about congestion?
        
        # Three components of energy
        # Household energy is long range and changes for all suburbs
        household_energy_distribution_old = -params['c_p'] * np.nansum(householdsPerSuburb / (distances ** params['alpha_p']) , 1)  # Dividing a row vector by a matrix repeats the operation for each row of the matrix. Summed across rows
        newHouseholdsPerSuburb = np.copy(householdsPerSuburb)
        newHouseholdsPerSuburb[homeSuburb] += 1
        household_energy_distribution_new = -params['c_p'] * np.nansum(newHouseholdsPerSuburb / (distances ** params['alpha_p']) , 1)
        
        # Local repulsion only changes for one suburb
        local_energy_distribution_old = params['c_l'] * (householdsPerSuburb[homeSuburb] ** params['alpha_l'] / params['cell_capacity'] + params['fixed_energy'])
        local_energy_distribution_new = params['c_l'] * (newHouseholdsPerSuburb[homeSuburb] ** params['alpha_l'] / params['cell_capacity'] + params['fixed_energy'])
        
        # For household energy and local repulsion the energy is compounded by the number of people in those suburbs
        # For work energy, the change only occurs for the new household
        household_energy_old = household_energy_distribution_old * householdsPerSuburb
        household_energy_new = household_energy_distribution_new * newHouseholdsPerSuburb
        local_energy_old = local_energy_distribution_old * householdsPerSuburb[homeSuburb]
        local_energy_new = local_energy_distribution_new * newHouseholdsPerSuburb[homeSuburb]
        work_energy = -params['c_w'] / distances[homeSuburb][workSuburb] ** params['alpha_w']  # Same as the change for a single agent
        if np.isnan(work_energy):
            work_energy = 0  # Replace nan with 0, since this means that one of the two suburbs is disconnected
        
        # Sum of change in all energy components
        H = np.sum(household_energy_new) - np.sum(household_energy_old) + work_energy + local_energy_new - local_energy_old
        return H

    def energyChangeGlobal_edgeSwap(self, household_1, household_2, params = None):        
        # Calculates the change in global energy for an edge swap operation
        # The global change is the same as the change only considering the two households being swapped.
        # No other agents are affected by the edge swap
        # Household and local energy components stay the same. Only the work component changes
        
        params = self.getParameters(params)
        
        distances = self.topology.distances
        
        home_1, work_1 = household_1
        home_2, work_2 = household_2
        
        work_energy_old = -params['c_w'] / (distances[[home_1, home_2], [work_1, work_2]] ** params['alpha_w'])
        work_energy_new = -params['c_w'] / (distances[[home_2, home_1], [work_1, work_2]] ** params['alpha_w'])
        
        return np.sum(work_energy_new - work_energy_old)  # Sum up the two suburbs

    def globalEnergy(self, householdsPerSuburb, distances = None, params = None):
        # Set up parameters
        # Supplied parameters (eg from Maximum Likelihood) replace those taken from the initially set values
        params = self.getParameters(params)
        
        if distances is None:
            distances = self.topology.distances
        
        # Three components of energy
        # Household energy
        household_energy_distribution = -params['c_p'] * np.nansum(householdsPerSuburb / (distances ** params['alpha_p']) , 1)  # Dividing a row vector by a matrix repeats the operation for each row of the matrix. Summed across rows
        household_energy = household_energy_distribution * householdsPerSuburb
        
        # Local repulsion
        local_energy_distribution = params['c_l'] * (householdsPerSuburb ** params['alpha_l'] / params['cell_capacity'] + params['fixed_energy'])
        local_energy = local_energy_distribution * householdsPerSuburb
        
        # Work energy
        work_energy_distribution = -params['c_w'] / (distances ** params['alpha_w'])
        work_energy = work_energy_distribution * self.TTWArray
        
        return (np.sum(household_energy), np.nansum(work_energy), np.sum(local_energy))

    def energyOfSingleFirm(self, workSuburb, workPerSuburb, params = None):
        
        params = self.getParameters(params)
        
        distances = self.topology.distances
#        agglomeration_energy = - c_a * np.dot(workPerSuburb, np.exp(-alpha_a * distances[workSuburb]))
        
        agglomeration_energy = workPerSuburb / (distances[workSuburb] ** params['alpha_a'])
        agglomeration_energy = -params['c_a'] * np.nansum(agglomeration_energy)
#        print('a:',agglomeration_energy)
        
        localOccupants = self.householdsPerSuburb[workSuburb] + workPerSuburb[workSuburb]        
        local_energy = params['c_l'] * (localOccupants ** params['alpha_l'] /  ## Changed from 2
                                params['cell_capacity'] + params['fixed_energy'])
#        print('l:', local_energy)

        return local_energy + agglomeration_energy
    
    def probability_new_household(self,homeSuburb, workSuburb, useEnergyOfSingleHousehold = True):
        if useEnergyOfSingleHousehold:
            # Initial energy of the new household is equal to 0
            # Final energy is given by the energy experienced by the new household
            newHouseholdsPerSuburb = np.copy(self.householdsPerSuburb)
            newHouseholdsPerSuburb[homeSuburb] += 1
            e = self.energyOfSingleHousehold(homeSuburb, workSuburb, newHouseholdsPerSuburb)
        else:
            # Uses energyChangeGlobal_newHousehold, which already considers the change in energy from adding a single household
            e = self.energyChangeGlobal_newHousehold(homeSuburb, workSuburb, self.householdsPerSuburb)
        
        P = 1 if e < 0 else np.exp(-1 * e * self.parameters["beta"])
        return P
    
    def probability_relocation(self, household, newHomeSuburb, useEnergyOfSingleHousehold = True, congestion = False):
        if useEnergyOfSingleHousehold:
            # Initial energy is found from the household at the current location
            # Final energy is found from the household at the new location
            initialE = self.energyOfSingleHousehold(household[0], household[1], self.householdsPerSuburb, congestion)
            newHouseholdsPerSuburb = np.copy(self.householdsPerSuburb)
            newHouseholdsPerSuburb[household[0]] -= 1
            newHouseholdsPerSuburb[newHomeSuburb] += 1
            finalE = self.energyOfSingleHousehold(newHomeSuburb, household[1], newHouseholdsPerSuburb, congestion)
        else:
            # Using energyChangeGlobal_newHousehold. The (negative) initialE is taken as the change in energy of removing the household from the existing suburb
            oneLessHouseholdsPerSuburb = np.copy(self.householdsPerSuburb)
            oneLessHouseholdsPerSuburb[household[0]] -= 1
            initialE = self.energyChangeGlobal_newHousehold(household[0], household[1], oneLessHouseholdsPerSuburb)
            finalE = self.energyChangeGlobal_newHousehold(newHomeSuburb, household[1], self.householdsPerSuburb)
        
        e = finalE - initialE
        P = 1 if e < 0 else np.exp(-1 * e * self.parameters["beta"])
        return P

    def probability_relocate_firm(self, oldSuburb, newSuburb): 
        initialE = self.energyOfSingleFirm(oldSuburb, self.workPerSuburb)
        newWorkPerSuburb = np.copy(self.workPerSuburb)
        newWorkPerSuburb[oldSuburb] -= 1
        newWorkPerSuburb[newSuburb] += 1
        finalE = self.energyOfSingleFirm(newSuburb, newWorkPerSuburb)
        
        e = finalE - initialE
        P = 1 if e < 0 else np.exp(-1 * e * self.parameters["beta"])
        return P

    def probability_edge_swap(self, household_1, household_2):
        e = self.energyChangeGlobal_edgeSwap(household_1, household_2)
        P = 1 if e < 0 else np.exp(-1 * e * self.parameters["beta"])
        return P

    def addHousehold(self, homeSuburb, workSuburb):
        self.households = self.households.append(pd.DataFrame([[homeSuburb, workSuburb]]), ignore_index = True).astype(int)
        self.householdsPerSuburb[homeSuburb] += 1
        self.TTWArray[homeSuburb][workSuburb] += 1

    def relocateHousehold(self, householdNumber, newHomeSuburb):  # Home suburb
        oldHomeSuburb = self.households.iloc[householdNumber, 0]
        workSuburb = self.households.iloc[householdNumber, 1]
        self.households.iloc[householdNumber, 0] = newHomeSuburb
        self.householdsPerSuburb[oldHomeSuburb] -= 1
        self.householdsPerSuburb[newHomeSuburb] += 1
        self.TTWArray[oldHomeSuburb][workSuburb] -= 1
        self.TTWArray[newHomeSuburb][workSuburb] += 1
    
    def relocateFirm(self, householdNumber, newWorkSuburb):  # Work suburb
        oldWorkSuburb = self.households.iloc[householdNumber, 1]
        homeSuburb = self.households.iloc[householdNumber, 0]
        self.households.iloc[householdNumber, 1] = newWorkSuburb
        self.workPerSuburb[oldWorkSuburb] -= 1
        self.workPerSuburb[newWorkSuburb] += 1
        self.TTWArray[homeSuburb][oldWorkSuburb] -= 1
        self.TTWArray[homeSuburb][newWorkSuburb] += 1
        
    def performEdgeSwap(self, household_1_Number, household_2_Number):
        household_1_home_suburb = self.households.iloc[household_1_Number,0]
        household_2_home_suburb = self.households.iloc[household_2_Number,0]
        self.relocateHousehold(household_1_Number, household_2_home_suburb)
        self.relocateHousehold(household_2_Number, household_1_home_suburb)
    
    def tick_new_household(self, useEnergyOfSingleHousehold = True):
        # Proposes to add a new household with a home suburb generated randomly and work suburb generated according
        # to the workDistribution probability density.
        # If the random number generated is less than the probability value then the new household is added
        homeSuburb = np.random.randint(self.arraySize)
        workSuburb = np.random.choice(self.arraySize, 1, p = self.workDistribution)[0]
        # Select whether to use single household or global energy
        P = self.probability_new_household(homeSuburb, workSuburb, useEnergyOfSingleHousehold)
        if P == 1:
            self.addHousehold(homeSuburb, workSuburb)
        else:
            if np.random.random() < P:
                self.addHousehold(homeSuburb, workSuburb)
        return P
                
    def tick_relocation(self, useEnergyOfSingleHousehold = True, congestion = False):
        # Proposes to relocate one household. The new home suburb is generated randomly
        householdNumber = np.random.randint(len(self.households))
        household = self.households.iloc[householdNumber,:]
        while True:  # Repeat until a new suburb is selected
            newHomeSuburb = np.random.randint(self.arraySize)
            if newHomeSuburb != household[0]:
                break
        # Select whether to use single household or global energy
        P = self.probability_relocation(household, newHomeSuburb, useEnergyOfSingleHousehold, congestion)
        if P == 1:
            self.relocateHousehold(householdNumber, newHomeSuburb)
        else:
            if np.random.random() < P:
                self.relocateHousehold(householdNumber, newHomeSuburb)
        return (P, household[0], newHomeSuburb)

    def tick_relocate_firm(self):
        # Proposes to relocate one firm. The new work suburb is generated randomly
        householdNumber = np.random.randint(len(self.households))
        household = self.households.iloc[householdNumber,:]
        while True:  # Repeat until a new suburb is selected
            newWorkSuburb = np.random.randint(self.arraySize)
            if newWorkSuburb != household[1]:
                break
        P = self.probability_relocate_firm(household[1], newWorkSuburb)
        if P == 1:
            self.relocateFirm(householdNumber, newWorkSuburb)
        else:
            if np.random.random() < P:
                self.relocateFirm(householdNumber, newWorkSuburb)
        return P

    def tick_edge_swap(self):
        # Proposes an edge swap between two randomly selected households
        # This can be seen as the households swapping living suburbs or swapping work locations
        household_1_Number = np.random.randint(len(self.households))
        household_1 = self.households.iloc[household_1_Number,:]
        while True:
            household_2_Number = np.random.randint(len(self.households))
            household_2 = self.households.iloc[household_2_Number,:]
            if not np.any(household_1 == household_2):  # Redraw if either the home or work suburbs are the same
                break
        P = self.probability_edge_swap(household_1, household_2)
        if P == 1:
            self.performEdgeSwap(household_1_Number, household_2_Number)
        else:
            if np.random.random() < P:
                self.performEdgeSwap(household_1_Number, household_2_Number)
        return P

        
    def getTTWArray(self):
        distributionTable = np.zeros((self.arraySize, self.arraySize), dtype = int)
        for row in range(self.arraySize):
            distributionTable[row] = np.bincount(self.households[self.households[0] == row][1], minlength = self.arraySize)
        self.TTWArray = distributionTable
        self.householdsPerSuburb = np.sum(distributionTable, 1, dtype = int)
        self.workPerSuburb = np.sum(distributionTable, 0, dtype = int)
        self.totalHouseholds = np.sum(self.householdsPerSuburb, dtype = int)
    
    def runSimulation(self, addHouseholds = True, relocation = True, edgeSwaps = False, ticks = 5E4,\
                      relocPerNewHousehold = 10, verbose = True, simulatedAnnealing = False,\
                      globalDistributionFunction = None, distributionTicks = 1000, \
                      useEnergyOfSingleHousehold = True, congestionUpdate = None):
        # globalDistributionFunction may be a single function or a list of functions
        if globalDistributionFunction is not None:
            assert ticks > distributionTicks
            if type(globalDistributionFunction) == list:
                self.globalDistribution = np.zeros([distributionTicks, len(globalDistributionFunction)])
            else:
                self.globalDistribution = np.zeros(distributionTicks)
        ticks = int(ticks)  # Cast as int
        
        if simulatedAnnealing:
            self.originalBeta = self.parameters["beta"]
        
#        params = getParameters(None)  # Should use this instead of starting each run from nothing
        
        for t in range(ticks):
            if t % 1000 == 0 and verbose:
                print("Step: " + str(t))
            if simulatedAnnealing and t % 20000 == 0 and t != 0:
                self.parameters["beta"] += 1
                if verbose:
                    print("Increasing beta by 1 to " + str(self.parameters["beta"]))
            
            if congestionUpdate is None:
                congestion = None
            elif t % congestionUpdate == 0:
                congestion = True
            else:
                congestion = False
            
            if addHouseholds:
                P = self.tick_new_household(useEnergyOfSingleHousehold)
                if relocation and len(self.households) > 0:
                    for i in range(relocPerNewHousehold):
                        self.tick_relocation(useEnergyOfSingleHousehold)
            elif relocation:
                (P, oldHomeSuburb, newHomeSuburb) = self.tick_relocation(useEnergyOfSingleHousehold, congestion)

            if edgeSwaps:
                assert np.sum(self.workDistribution > 0) > 1  # Make sure there is not just one work suburb
                P = self.tick_edge_swap()
            
            if self.workDistribution is None:
                self.tick_relocate_firm()
            
            if t >= ticks - 100 and verbose:
                print('P: ' + str(P))
                if (not addHouseholds) and relocation and P == 1:
                    print("{} -> {}".format(oldHomeSuburb, newHomeSuburb))
            if t >= ticks - distributionTicks and globalDistributionFunction is not None:
                try:
                    # If globalDistributionFunction is a list of functions
                    for i, function in enumerate(globalDistributionFunction):
                        self.globalDistribution[t - ticks + distributionTicks, i] = function()
                except:
                    self.globalDistribution[t - ticks + distributionTicks] = globalDistributionFunction()
    
        if simulatedAnnealing:
            self.finalBeta = self.parameters["beta"]
            self.parameters["beta"] = self.originalBeta
    
    def visualise(self, householdsPerSuburb = None, ax = None, cax = None, showInfrastructure = False,\
                  showWork = False, logScale = False, saveName = None,\
                  maxHouseholds = None, discreteColour = None, fig = None):
        
        if householdsPerSuburb is None:
            householdsPerSuburb = self.householdsPerSuburb
        else:
            householdsPerSuburb = householdsPerSuburb.ravel()
        
        if showWork:
#            work = self.workDistribution
            work = self.workPerSuburb
        else:
            work = None
        
        self.topology.visualise(ax, cax, householdsPerSuburb, work, showInfrastructure,\
                                logScale, saveName, True, maxHouseholds, discreteColour, fig)

    def animate(self, frames, ax = None, cax = None, ticks = 5E4, verbose = False, **kwargs):
        if ax is None:
            ax = plt.axes()
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
        
        ticksPerRound = ticks // frames
        
#        plt.ion()
        plt.show()
        for i in range(frames):
            self.runSimulation(ticks = ticksPerRound, verbose = verbose, **kwargs)
            ax.cla()
            cax.cla()
            self.visualise(ax = ax, cax = cax)
            print("Frame: {} of {}".format(i, frames))
#            plt.draw()
            plt.pause(0.001)
    
    def negLogLikelihood(self, params, param_keys, fixed_params = {}, TTWArray = None):

        params = dict(zip(param_keys, params))  # Combine with the keys to turn the list into a dictionary
        for key, value in fixed_params.items():
            params.setdefault(key, value)
        params = self.getParameters(params)
#        params['beta'] = 1  # Manually ignoring beta
#        params['cell_capacity'] = 1
#        params['fixed_energy'] = 0
        
        # local energy params
        local_energy_keys = []
        local_energy_values = []
        for key, value in params.items():
            if key.startswith('local_energy'):
                try:
                    local_energy_keys.append(int(key[-3:]))
                    local_energy_values.append(value)
                except ValueError:  #  'local_energy' comes from fixed_params and does not need to be recombined
                    pass
        local_energy = list(zip(local_energy_keys, local_energy_values))
        if len(local_energy) > 0:
            local_energy_sorted = sorted(local_energy)
            local_energy = np.array(local_energy_sorted)[:,1]
            params.update({'local_energy': local_energy})
        
        if TTWArray is None:
            TTWArray = self.TTWArray
            
        householdsPerSuburb = self.householdsPerSuburb
        
        
        H = self.energyOfSingleHousehold(None, None, householdsPerSuburb, params = params)

        Z = np.sum(np.exp(-params['beta'] * H), axis = 0, keepdims = True)  # Sum for each workplace
    
        lnP = np.multiply(TTWArray, -params['beta'] * H - np.log(Z))
        return -np.sum(lnP)
    
    
    def maximiseLikelihood(self, initParams, bounds, trials = 100, useDE = False, fixed_params = {}, TTWArray = None):
        # InitParams and bounds are input as dictionaries
        
        keys = initParams.keys()
        
        if 'local_energy' in keys:
            initParams = initParams.copy()
            bounds = bounds.copy()
            local_energy = initParams['local_energy']
            local_energy_dict = dict(zip(('local_energy_{:03d}'.format(i) for i in range(len(local_energy))), local_energy))
            local_energy_bounds_dict = dict(zip(('local_energy_{:03d}'.format(i) for i in range(len(local_energy))), [bounds['local_energy']] * len(local_energy)))
            initParams.update(local_energy_dict)
            initParams.pop('local_energy')
            bounds.update(local_energy_bounds_dict)
            bounds.pop('local_energy')
            keys = initParams.keys()
            
        finalParams = []
        loglikelihood = [] 
        
        for i in range(trials): 
            if useDE:
                results = differential_evolution(self.negLogLikelihood, [bounds[x] for x in keys],
                                                 args = (keys, fixed_params, TTWArray))
            else:
                # Take init as drawn from a normal distribution with mean and variance set as the initParams value
                if trials == 1:
                    init = [initParams[key] for key in keys]
                else:
                    init = [np.maximum(np.random.normal(initParams[key], initParams[key]), bounds[key][0]) for key in keys]  # Did not include a check against the bound maximum because None values are sometimes used. The minimizer should automatically correct this anyway
                results = minimize(self.negLogLikelihood, init, args = (keys, fixed_params, TTWArray),
                                   bounds = [bounds[x] for x in keys])
            
            if results.success:
                finalParams.append(results.x)
                loglikelihood.append(results.fun)

        idx = np.nanargmin(loglikelihood)
        
        return dict(zip(keys, finalParams[idx])), loglikelihood[idx]
        
    
    def probabilityFromParameters(self, params):
        params = self.getParameters(params)
        print(params)
        
        householdsPerSuburb = self.householdsPerSuburb
        workDistribution = np.atleast_2d(self.workDistribution)
        
        H = self.energyOfSingleHousehold(None, None, householdsPerSuburb, params = params)
        
        exps = np.exp(-params['beta'] * H)
        P = exps / np.sum(exps, axis = 0, keepdims = True)
        P = P * workDistribution  # Converting P(i|j) to P(i,j)
        assert not np.any(np.isnan(P))
        
        self.mllProbability = P
        
        return P
    
    def arrayFromProbability(self, P = None, households = None, minimiseRandomness = True):
        if P is None:
            P = self.mllProbability
        
        if households is None:
            households = self.totalHouseholds
        
        suburbs = self.arraySize
        if minimiseRandomness:
            # First assign some of the households to minimise randomness
            TTWArray_output = np.floor(P * households).astype(int)
        else:
            TTWArray_output = np.zeros_like(P)
        
        # Probabilistically assign remaining households
        households_distributed = np.sum(TTWArray_output).astype(int)
        householdTTWChoices = np.random.choice(range(suburbs**2), size = households - households_distributed, p = P.ravel())
        householdPerTTW_output = np.bincount(householdTTWChoices, minlength = suburbs**2)
        TTWArray_output += householdPerTTW_output.reshape((suburbs, suburbs))  # Generate a TTWArray_output
        
        self.mllArray = TTWArray_output
        self.mllHHDist = np.sum(TTWArray_output, axis = 1)
        return self.mllArray
        
    def hellingerDistance(self, actualDistribution, mllDistribution):
        distance = (np.sqrt(actualDistribution) - np.sqrt(mllDistribution)) ** 2
        distance = np.sqrt(np.sum(distance)) / np.sqrt(2)
        return distance
    
    def averageTimetoWork(self, congestion = False):
        distances = self.topology.getDistanceMatrixWithCongestion(self.TTWArray, A = int(congestion))
        return np.sum(self.TTWArray * distances) / self.totalHouseholds
    
    def differenceInAverageTimeToWork(self, congestion = False):  # For two city simulation
        distances = self.topology.getDistanceMatrixWithCongestion(self.TTWArray, A = int(congestion))
        workIndices = np.flatnonzero(self.workDistribution)
        averageTimeToWork_1 = np.sum(self.TTWArray[:,workIndices[0]] * distances[:,workIndices[0]]) / np.sum(self.TTWArray[:,workIndices[0]])
        averageTimeToWork_2 = np.sum(self.TTWArray[:,workIndices[1]] * distances[:,workIndices[1]]) / np.sum(self.TTWArray[:,workIndices[1]])
        
        return abs(averageTimeToWork_1 - averageTimeToWork_2)
    
    def entropy(self, useTTWArray = True):
        if useTTWArray:
            probability = self.TTWArray / self.totalHouseholds
        else:
            probability = self.householdsPerSuburb / self.totalHouseholds
        log_probability = np.log2(probability)
        log_probability[log_probability == -np.inf] = 0
        return -np.sum(probability * log_probability)

    def save(self, saveName):
        with open(saveName + '.pkl', 'wb') as f:
            pickle.dump(self, f)

def load(fileName):
    if fileName.split('.')[-1] == "pkl":
        with open(fileName, 'rb') as f:
            return pickle.load(f)
    else:
        with open(fileName + '.pkl', 'rb') as f:
            return pickle.load(f)

#def sweepParameters(TTWClass, trials = 1, testPoints = 10, saveName = None):
#    rs = np.linspace(0.05, 0.95, tests)
#    entropyHH = np.zeros([tests,trials])
#    entropyTTW = np.zeros([tests,trials])
#    timeToWork = np.zeros([tests,trials])
#    household_energy = np.zeros([tests,trials])
#    work_energy = np.zeros([tests,trials])
#    local_energy = np.zeros([tests,trials])


### Testing
if __name__ == "__main__":

    layers = 25
#    t = gridTopology(layers = layers, weight = 1, constant = 1, diagonals = False)
#    t = hexagonalTopology(5, weight = 1)
    t = oneDimTopology(layers)
#    t.createLatticeInfrastructure()
    
#    t.randomiseWeightOfInfrastructure(theta = 1)
    cityGap = 11
#    workDistribution = [0] * (layers + 1) * (layers // 2) + [1]
#    workDistribution = [0] * layers * (layers // 2) + [0] * ((layers - cityGap) // 2 - 1) + [0.5] + [0] * cityGap + [0.5]
    workDistribution = [0] * ((layers - cityGap) // 2 - 1) + [0.5] + [0] * cityGap + [0.5]  # For 1D topology
#    workDistribution = None
    a = TTW(t, workDistribution = workDistribution)
    a.initialiseRandomHouseholds(200)
    
    ax = plt.axes()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    
    a.visualise(ax=ax, cax=cax, showWork = True, showInfrastructure = False)
    plt.show()
#    
    for i in range(5):
        a.runSimulation(addHouseholds = False, ticks = 500, verbose = False)
        a.visualise(showWork = True, showInfrastructure = False)
        a.parameters['beta'] += 1
        plt.show()
    
    a.runSimulation(addHouseholds = False, relocation = False, edgeSwaps = True, ticks = 5000, verbose = False)
    

    
#    a.animate(10)



    
##    t.createPreferentialAttachmentInfrastructure(linksToCreate = 100)
##    t = sydneyTopology()
#    workDistribution = np.array([0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.01, 0.01,\
#                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005])
#     
#    a = TTW(t, workDistribution = workDistribution)
#    a.initialiseRandomHouseholds(500, exactWorkDistribution = True)
#    for i in range(1):
#        a.runSimulation(addHouseholds = False, relocation = True, ticks = 5000, verbose=False)
#        t.growOptimalNetwork(a.householdsPerSuburb, a.globalEnergy)
##    t.removeRandomInfrastructure(300)
#    a.visualise(showInfrastructure=True, saveName='T', showWork = True)

    # Try Scale Free Network
#    G = nx.barabasi_albert_graph(300, 1)
#    t = topology(G)
#    degree = dict(nx.degree(G))
#    highestDegreeIndex = max(degree, key = lambda x: degree[x])
#    workDistribution = [0] * highestDegreeIndex + [1]
#    a = TTW(t, workDistribution = workDistribution)
##    a.initialiseRandomHouseholds(500)
#    a.runSimulation(addHouseholds = False, relocation = True, ticks = 50000, verbose=True, congestionUpdate = 500)
#    a.visualise(showWork = True, saveName = 'Test2')



#    t = gridTopology(diagonals = True)
#    t = urbanStreet(layers = 21, numberOfNodes = 75, diagonals = True)
#    t.createLatticeInfrastructure()
#    a = TTW(t, workDistribution = [0] * 112 + [1])
#    a = TTW(t, workDistribution = [0] * 220 + [1])
#    a.initialiseRandomHouseholds(500)
#    for i in range(10):
#        a.runSimulation(False, ticks = 5000)
#        a.visualise(showWork=True, saveName = 'Test'+str(i))