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
        #next(csv_reader) # skipping the header
        for row in csv_reader:  
            a = row[0]
            b = row[1]
            t = row[2]
            if a not in nodes:
                nodes.append(a)
            if b not in nodes:
                nodes.append(b)                
            if t in edges_per_t.keys():
                edges_per_t[t].append((a, b))
            else:
                edges_per_t[t] = [(a, b)] 
    return nodes, edges_per_t


# Aggregates all the edges. If there are duplicate edges 
# (multiple interactions in time) it collapses them. 
def generate_aggregated_graph(nodes, edges_per_t, repeated_edges_behavior=[], directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    temp_edges = []
    for k, v in edges_per_t.items():
        temp_edges = temp_edges + v
    temp_edges = list(dict.fromkeys(temp_edges))
    # setup graph
    G.add_nodes_from(nodes)
    for e in temp_edges:
        G.add_edge(e[0], e[1])

    return G


# Computes the directed aggregated graph with a weight equal to the number
# of interactions that each node has
def generate_weighted_aggregated_graph(nodes, edges_per_t, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    temp_edges = []
    for v in edges_per_t.values():
        temp_edges += v
    weight_dict = {i: temp_edges.count(i) for i in set(temp_edges)}
    # setup graph
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([k+(weight_dict[k],) for k in weight_dict.keys()])
    return G


# Calculates the aggregated "popularity" (number of interactions until time t) of each node
# both for inlinks and outlinks for the given granularity
def get_popularity_per_time(edges_per_t):
    in_node_popularity = {}
    out_node_popularity = {}
    in_last_checked = {}
    out_last_checked = {}

    sorted_edges = dict(sorted(edges_per_t.items(), key=lambda s: s[0]))

    for k, v in sorted_edges.items():

        for e in v:
            if e[0] not in out_node_popularity.keys():
                out_node_popularity[e[0]] = {k: 1}
            else:
                if k not in out_node_popularity[e[0]].keys():
                    out_node_popularity[e[0]][k] = out_node_popularity[e[0]][out_last_checked[e[0]]] + 1
                else:
                    out_node_popularity[e[0]][k] += 1
            out_last_checked[e[0]] = k
            if e[1] not in in_node_popularity.keys():
                in_node_popularity[e[1]] = {k: 1}
            else:
                if k not in in_node_popularity[e[1]].keys():
                    in_node_popularity[e[1]][k] = in_node_popularity[e[1]][in_last_checked[e[1]]] + 1
                else:
                    in_node_popularity[e[1]][k] += 1
            in_last_checked[e[1]] = k
    return in_node_popularity, out_node_popularity


# Makes the popularity (not bar in the current version) chart given the nodes specified in the nodes list
# and a list of dictionaries for each node with the timestep as key and the number of interactions as value
def make_popularity_bar_chart(users, nodes):
    y_pos = np.arange(len(users[0].keys()))
    for ind, user in enumerate(users):
        popularity = list(user.values())
        plt.plot(y_pos, popularity, label='user '+nodes[ind])
    plt.xticks(y_pos[0:len(users[0].keys()):200], list(users[0].keys())[0:len(users[0].keys()):200])
    plt.xticks(rotation=45)
    plt.xlabel('Day')
    plt.ylabel('Aggregated Number of interactions')
    plt.title('Popularity of top 5 users')
    plt.grid(True)
    plt.legend()

    plt.show()


# Finds the popularity distribution of the top 5 most popular nodes
# The _popularity arguments are the output of get_popularity_per_time function
# The _sorted arguments are a list of node - num of interactions pairs sorted in descending order
def check_popularity_patterns(in_node_popularity, out_node_popularity, in_degree_sorted, out_degree_sorted, time):
    top5_in_nodes = list(map(lambda x: x[0], in_degree_sorted[0:5]))
    top5_out_nodes = list(map(lambda x: x[0], out_degree_sorted[0:5]))
    top5_in = []
    top5_out = []
    for n in top5_in_nodes:
        for ind, t in enumerate(time):
            if t not in in_node_popularity[n].keys():
                in_node_popularity[n][t] = 0 if ind == 0 else in_node_popularity[n][time[ind-1]]
        top5_in.append((n, in_node_popularity[n]))
    for n in top5_out_nodes:
        for ind, t in enumerate(time):
            if t not in out_node_popularity[n].keys():
                out_node_popularity[n][t] = 0 if ind == 0 else out_node_popularity[n][time[ind-1]]
        top5_out.append((n, out_node_popularity[n]))
    return top5_in, top5_out


# Computes the degree of each node in the graph
# First column is the id and second is the degree
def get_graph_degrees(G):
    G_degrees = np.zeros((len(G.degree(weight='weight')), 2))
    position = 0
    for i, v in G.degree(weight='weight'):
        G_degrees[position, 0] = i
        G_degrees[position, 1] = v
        position += 1
    return G_degrees


# Computes the in-degree of each node in the graph
# First column is the id and second is the degree
def get_graph_in_degrees(G):
    G_degrees = np.zeros((len(G.in_degree(weight='weight')), 2))
    position = 0
    for i, v in G.in_degree(weight='weight'):
        G_degrees[position, 0] = i
        G_degrees[position, 1] = v
        position += 1
    return G_degrees


# Computes the out-degree of each node in the graph
# First column is the id and second is the degree
def get_graph_out_degrees(G):
    G_degrees = np.zeros((len(G.out_degree(weight='weight')), 2))
    position = 0
    for i, v in G.out_degree(weight='weight'):
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

    granularity_factor = 1 if granularity == 'sec' else 60 if granularity == 'min' else 3600 if granularity == 'hour' \
        else 3600 * 24

    for k, v in edges_per_t.items():
        second_since_start = float(k) - float(min_t)  # the start is the first element's timestamp
        bin_id = math.floor(second_since_start/granularity_factor)
        
        if bin_id in new_edges:
            new_edges[bin_id] = new_edges[bin_id] + v
        else:
            new_edges[bin_id] = [] + v
    
    return new_edges
