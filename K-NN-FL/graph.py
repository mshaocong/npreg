# -*- coding: utf-8 -*- 
import numpy as np 

"""
kneighbors_graph(X, n_neighbors, metric)
Input: 
    the dataset 'X'; each row is an observation
    IMPORTANT parameter 'n_neighbors'
    the distance function 'metric'
Output:
    the graph
    the incidence matrix G
"""
def kneighbors_graph(X, n_neighbors, metric):
    nrows = X.shape[0]
    dist = metric
    
    #Generate the graph structure
    graph = {}
    for i in range(nrows): 
        xi = X[i, :]  
        distance_to_xi = dist(xi, X)
        nearest_k_points = np.argsort(distance_to_xi)[1:(n_neighbors+1)]  
        xi_connect_to = []
        
        for point in nearest_k_points:
            if point in graph.keys(): 
                if i not in graph[point]:
                    xi_connect_to.append(point)
            else:
                xi_connect_to.append(point)
        graph[i] = xi_connect_to
    
    #Find the incident matrix
    G = np.zeros((n_neighbors*nrows, nrows)) 
    for i in graph:
        fill_row = i*n_neighbors
        for j in graph[i]:
            if j > i:
                G[fill_row,i] = 1
                G[fill_row,j] = -1
            else:
                G[fill_row,i] = -1
                G[fill_row,j] = 1 
            fill_row += 1 
    G = G.astype('float32')
    return (graph, G)

"""
Input: 
    the dataset 'X'; each row is an observation
    the distance function 'dist'
Output:
    the graph
    the incidence matrix G 
"""
#def epsilon_neighbors_graph(X, dist, epsilon):
#    pass