import networkx as nx
import argparse
import pprint
import numpy as np

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from generate_graph import *

parser = argparse.ArgumentParser(
    description="Computing local curvatures for Graphs"
)

parser.add_argument(
    "--depth", type = int, help="depth of the generated tree"
)
parser.add_argument(
    "--branch", type = int, help="the maximum branching factor for the generated tree"
)

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

def get_parallelogram_curvature(G):
    parallelograms_to_check = len(G) * 100
    d = dict(nx.shortest_path_length(G))
    sum_curvature = 0
    for _ in range(parallelograms_to_check):
        while True:
            m = random.sample(G.nodes(), 1)[0]
            if G.degree[m] >= 2:
                break
        b, c = random.sample(list(G.neighbors(m)), 2)
        while True:
            a = random.sample(G.nodes(), 1)[0]
            if a not in [m, b, c]:
                break
        sum_curvature += (d[a][m] ** 2 + d[b][c] ** 2 / 4.0 - (d[b][a] ** 2 + d[a][c] ** 2) / 2.0) / (2.0 * d[a][m])
    return sum_curvature / parallelograms_to_check

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

if __name__ == '__main__':
    args = parser.parse_args()
    G = generate_tree(args.depth, args.branch,seed=1)
    ollivier = get_ollivier_curvature(G)
    forman = get_forman_curvature(G)
    print("ollivier: ", ollivier)
    print("forman: ", forman)
    #   Sectional Curvature estimation based on our implementation for HyperKGQA
    sectional, _ = sectional_curvature(G)
    #   Sectional Curvature estimation based on Global Graph Curvature
    parallelogram = get_parallelogram_curvature(G)
    print("sectional: {}, parallelogram: {}".format(sectional, parallelogram))
