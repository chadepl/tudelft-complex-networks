import networkx as nx
import csv
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import math
import operator

def read_graph_from_file(path):
    edges_per_t = {}
    nodes = []
    with open(path) as file_handle:
        csv_reader = csv.reader(file_handle, delimiter = ' ')
        next(csv_reader) # skipping the header
        for row in csv_reader:  
            a = row[0]
            b = row[1]
            t = row[2]
            if a not in nodes:
                nodes.append(a)
            if b not in nodes:
                nodes.append(b)                
            if t in nodes:
                edges_per_t[t].append((a, b))
            else:
                edges_per_t[t] = [(a, b)] 
    return nodes, edges_per_t

# Aggregates all the edges. If there are duplicate edges 
# (multiple interactions in time) it collapses them. 
def generate_aggregated_graph(nodes, edges_per_t, repeated_edges_behavior):
    G = nx.DiGraph()
    temp_edges = []
    for k, v in edges_per_t.items():
        temp_edges = temp_edges + v
    temp_edges = list(dict.fromkeys(temp_edges))
    # setup graph
    G.add_nodes_from(nodes)
    for e in temp_edges:
        G.add_edge(e[0], e[1])
    
    return G

# Computes the degree of each node in the graph
# First column is the id and second is the degree
def get_graph_degrees(G):
    G_degrees = np.zeros((len(G.degree()), 2))
    position = 0
    for i, v in G.degree():
        G_degrees[position, 0] = i
        G_degrees[position, 1] = v
        position += 1
    return G_degrees

# Recomputes the edges_per_t so aggregating them with 
# a given granularity (This is relative to the first 
# timestamp of the dataset) 
def aggregate_edges_by_granularity(edges_per_t, granularity):
    new_edges = {}
    min_t = min([k for k in edges_per_t.keys()])
    max_t = max([k for k in edges_per_t.keys()])
    
    if(granularity == 'sec'):
        granularity_factor = 1
    elif(granularity == 'min'):
        granularity_factor = 60
    elif(granularity == 'hour'):
        granularity_factor = 3600
    elif(granularity == 'day'):
        granularity_factor = 3600 * 24
    
    for k,v in edges_per_t.items():
        second_since_start = float(k) - float(min_t) # the start is the first element's timestamp
        bin_id = round(second_since_start/granularity_factor)
        
        if bin_id in new_edges:
            new_edges[bin_id] = new_edges[bin_id] + v
        else:
            new_edges[bin_id] = [] + v
    
    return new_edges
    


#########################
# INFORMATION SPREADING #
#########################

