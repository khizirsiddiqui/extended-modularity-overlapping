"""
Extended modularity implemented in python


Ref:
Extending the definition of modularity to directed graphs with overlapping communities
- V Nicosia, G Mangioni, V Carchiolo and M Malgeri
J. Stat. Mech. (2009) P03024
"""

from itertools import combinations 
import numpy as np

def extended_modularity(graph, communities, weight='weight', p=0.2):
    q = 0.0
    degrees = dict(graph.degree(weight=weight))
    m = sum(degrees.values())
    n = graph.number_of_nodes()
    alpha = {}

    for community in communities:
        for nd in community:
            alpha[nd] = alpha.get(nd, 0) + 1

    def F(i, j, pr=p):
        return 1./((1. + np.exp(-(pr*(2./alpha[i] - 1.))))*(1. + np.exp(-(pr*(2./alpha[j] - 1.)))))

    for nd1, nd2 in combinations(graph.nodes, 2):
        if graph.has_edge(nd1, nd2):
            e = graph[nd1][nd2]
            wt = e.get(weight, 1)
        else:
            wt = 0
        for community in communities:
            beta_out = 0.0
            for j in graph.nodes:
                if j in community: continue
                if j == nd1: continue
                if j == nd2: continue
                beta_out = beta_out + F(nd1, j)
            beta_out = beta_out / n

            beta_in = 0.0
            for i in graph.nodes:
                if i in community: continue
                if i == nd1: continue
                if i == nd2: continue
                beta_in = beta_in + F(i, nd2)
            beta_in = beta_in / n

            q = q + F(nd1, nd2)*wt - float(beta_in * beta_out * degrees[nd1] * degrees[nd2] / m)

    return q / m
