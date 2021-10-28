import networkx as nx
import argparse
import pprint
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm
from networkx.algorithms import bipartite
import time

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

def get_ollivier_curvature(G):
    orc = OllivierRicci(G, alpha = 0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    # pprint.pprint([e for e in orc.G.edges.data()])
    # print("The Ollivier-Ricci curvature of edge (0,1) is %f" % orc.G[0][1]["ricciCurvature"])
    # print("The Ollivier-Ricci curvature of edge (0,1) is %f" % orc.G[1][2]["ricciCurvature"])
    return sum([e[2]['ricciCurvature'] for e in orc.G.edges.data()]) / len(orc.G.edges())

def get_forman_curvature(G):
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    return sum([e[2]['formanCurvature'] for e in frc.G.edges.data()]) / len(frc.G.edges())

def sample(G, n_samples):
    H = nx.to_scipy_sparse_matrix(G)
    nodes = list(G)
    nodes.sort()
    n = H.shape[0]
    curvature = []
    max_iter = 10000
    iter = 0
    idx = 0

    while idx < n_samples:

        # if in max_iter we cannot sample a triangle check the diameter of the 
        # component, must be at least 3 to sample triangles
        if iter == max_iter:
            d = nx.algorithms.distance_measures.diameter(G)
            if d < 3: return None

        iter = iter + 1

        b = random.randint(0, n-1)
        c = random.randint(0, n-1)
        if b == c: continue
        
        path = nx.shortest_path(G, source=nodes[b], target=nodes[c])
        if len(path) < 3: continue

        middle = len(path) // 2
        m = nodes.index(path[middle])
        l_bc = len(path) - 1
        
        # sample reference node
        a = random.choice([l for l in list(range(n)) if l not in [m,b,c]])
            
        path = nx.shortest_path(G, source=nodes[a], target=nodes[b])
        l_ab = len(path) - 1
           
        path = nx.shortest_path(G, source=nodes[a], target=nodes[c])
        l_ac = len(path) - 1

        path = nx.shortest_path(G, source=nodes[a], target=nodes[m])
        l_am = len(path) - 1

        idx = idx + 1
        curv = (l_am**2 + l_bc**2 / 4 - (l_ab**2 + l_ac**2) / 2) / (2 * l_am)
        curvature.append(curv)
    
    return curvature

def sectional_curvature(G):
    components = [G.subgraph(c) for c in nx.connected_components(G)]
    nodes = [c.number_of_nodes()**3 for c in components]
    total = np.sum(nodes)
    weights = [n/total for n in nodes]
    curvs = [0]

    for idx,c in enumerate(components):
        weight = weights[idx]
        n_samples = int(1000 * weight)
        if n_samples > 0 and c.number_of_nodes() > 3:
            curv = sample(c, n_samples)
            if curv is not None:
                curvs.extend(curv)
    
    return np.mean(curvs), total

start_time = time.time()
G = nx.gnm_random_graph(500, 500*499/2, directed=True)
U = G.to_undirected()
ollivier = get_ollivier_curvature(G)
print("ollivier: ", ollivier)
forman = get_forman_curvature(G)
print("forman: ", forman)
sectional, _ = sectional_curvature(U)
print("sectional: ", sectional)
print("--- %s seconds ---" % (time.time() - start_time))