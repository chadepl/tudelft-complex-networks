import copy
from collections import Counter
import networkx as nx
import numpy as np


class Algorithm:
    counts = Counter()
    l = 3
    WEIGHT = 0.1
    motifs = {}

    def setup_motifs(self):
        s1 = nx.DiGraph()
        s1.add_edge(1, 2, weight=0.1)
        s1.add_edge(2, 3, weight=0.2)

        s2 = nx.DiGraph()
        s2.add_edge(1, 2, weight=0.1)
        s2.add_edge(1, 3, weight=0.2)
        s2.add_edge(2, 3, weight=0.3)

        s3 = nx.DiGraph()
        s3.add_edge(1, 2, weight=0.1)
        s3.add_edge(1, 3, weight=0.2)
        s3.add_edge(3, 1, weight=0.3)

        s4 = nx.DiGraph()
        s4.add_edge(1, 2, weight=0.1)
        s4.add_edge(3, 2, weight=0.2)

        s5 = nx.DiGraph()
        s5.add_edge(1, 2, weight=0.1)
        s5.add_edge(1, 3, weight=0.2)

        self.motifs = {
            'S1': s1,
            'S2': s2,
            'S3': s3,
            'S4': s4,
            'S5': s5
        }

    def main_algorithm(self, edges, times, delta):
        self.setup_motifs()
        start = 1
        self.counts = Counter()

        for end in range(len(edges)):
            while times[start] + delta < times[end]:
                self.decrement_count(edges[start])
                start += 1
            self.increment_count(edges[end])

        return self.check_isomorphism(self.counts)

    def decrement_count(self, edge):
        self.counts[self.edge_to_di_graph(edge)] -= 1
        suffix_list = self.get_list_of_keys_sorted(self.l-1)
        for suffix in suffix_list:
            if self.length(suffix) < self.l - 1:
                self.counts[self.concat(edge, suffix, is_sufix=True)] -= self.counts[suffix]
            else:
                break

    def increment_count(self, edge):
        prefix_list = self.get_list_of_keys_sorted(self.l)
        prefix_list.reverse()
        for prefix in prefix_list:
            if self.length(prefix) < self.l:
                self.counts[self.concat(edge, prefix, is_sufix=False)] += self.counts[prefix]
            else:
                break
        self.counts[self.edge_to_di_graph(edge)] += 1

    def concat(self, edge, graph, is_sufix=False):
        graph_weights = [d['weight'] for (u, v, d) in graph.edges(data=True)]

        if is_sufix:
            new_weight = max(graph_weights) + self.WEIGHT
        else:
            new_weight = min(graph_weights) - self.WEIGHT
        new_graph = copy.deepcopy(graph)
        new_graph.add_edge(edge[0], edge[1], weight=new_weight)
        return new_graph

    def edge_to_di_graph(self, edge):
        graph = nx.DiGraph()
        graph.add_edge(edge[0], edge[1], weight=self.WEIGHT)
        return graph

    def check_isomorphism(self, graph_list):
        mcount = dict(zip(self.motifs.keys(), list(map(int, np.zeros(len(self.motifs))))))
        for graph in graph_list:
            mot_match = list(map(lambda mot_id: nx.is_isomorphic(graph, self.motifs[mot_id]), self.motifs.keys()))
            match_keys = [list(self.motifs.keys())[i] for i in range(len(self.motifs)) if mot_match[i]]
            if len(match_keys) == 1:
                mcount[match_keys[0]] += 1
        return mcount

    def get_list_of_keys_sorted(self, limit):
        suffix_list = [elem for elem in self.counts.keys() if len(elem.edges)<limit]
        suffix_list.sort(key=lambda s: len(s.edges))
        return suffix_list

    @staticmethod
    def length(graph):
        return len(graph.edges)