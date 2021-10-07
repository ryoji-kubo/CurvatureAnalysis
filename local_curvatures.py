import networkx as nx
import argparse
import pprint
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm
from networkx.algorithms import bipartite

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from generate_graph import *

parser = argparse.ArgumentParser(
    description="Computing local curvatures for Graphs"
)

parser.add_argument('--nargs', nargs='*', default='all', help = 'tree, balanced_tree')
parser.add_argument('--seed', type=int, default=None)
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
    # max_iter = 10000
    # iter = 0
    for _ in range(parallelograms_to_check):
        while True:
            # if iter == max_iter:
            #     d = nx.algorithms.distance_measures.diameter(G)
            #     if d < 3: return 0

            m = random.sample(G.nodes(), 1)[0]
            if G.degree[m] >= 2:
                break
            # iter = iter+1

        b, c = random.sample(list(G.neighbors(m)), 2)
        # iter = 0
        while True:
            # if iter == max_iter:
            #     d = nx.algorithms.distance_measures.diameter(G)
            #     if d < 3: return 0
            a = random.sample(G.nodes(), 1)[0]
            if a not in [m, b, c]:
                break
            # iter = iter+1

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
    random.seed(args.seed)

    if 'tree' in args.nargs or 'all' in args.nargs:
        df = get_tree_dataframe()

        for index, row in df.iterrows():
            print(f'{index+1}/{len(df.index)}')
            G = generate_tree(row['depth'],row['branch'],seed=args.seed)
            print(f'Generated Tree, Size: {G.order()}')

            if G.order()<=500:
                plt.figure()
                plt.title(f'Tree with Size {G.order()}')
                nx.draw(G)
                plt.savefig(f'Figures/Tree with Size {G.order()}')

            df.loc[index, 'number of nodes']=G.order()
            ##  Calculate the local curvatures
            ollivier = get_ollivier_curvature(G)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(G)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(G)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
            #   Sectional Curvature estimation based on Global Graph Curvature
            # parallelogram = get_parallelogram_curvature(G)
            # df.loc[index, 'parallelogram']=parallelogram
            # print("parallelogram: ", parallelogram)
        df.to_csv('Results/tree_curvature.csv', index=False)

    if 'balanced_tree' in args.nargs or 'all' in args.nargs:
        df = get_balanced_tree_dataframe()
        for index, row in df.iterrows():
            print(f'{index+1}/{len(df.index)}')
            G = nx.balanced_tree(row['r-ary'],row['height'])
            if G.order()<=500:
                plt.figure()
                plt.title(f'Balanced Tree with Size {G.order()}')
                nx.draw(G)
                plt.savefig(f'Figures/Balanced Tree with Size {G.order()}')
            print(f'Generated Balanced Tree, Size: {G.order()}')
            df.loc[index, 'number of nodes']=G.order()
            ##  Calculate the local curvatures
            ollivier = get_ollivier_curvature(G)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(G)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(G)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
        df.to_csv('Results/balanced_tree_curvature.csv', index=False)
    
    if 'star_graph' in args.nargs or 'all' in args.nargs:
        df = get_star_graph_dataframe()
        for index, row in df.iterrows():
            print(f'{index+1}/{len(df.index)}')
            G = nx.star_graph(row['number of nodes'])
            if G.order()<=500:
                plt.figure()
                plt.title(f'Star Graph with Size {G.order()}')
                nx.draw(G)
                plt.savefig(f'Figures/Star Graph with Size {G.order()}')
            print(f'Generated Star Graph, Size: {G.order()}')
            ##  Calculate the local curvatures
            ollivier = get_ollivier_curvature(G)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(G)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(G)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
        df.to_csv('Results/star_graph_curvature.csv', index=False)
    
    if 'scale_free' in args.nargs or 'all' in args.nargs:
        df = get_scale_free_dataframe()
        for index, row in df.iterrows():
            beta = row['beta']
            alpha = gamma = row['alpha gamma']
            print(f'{index+1}/{len(df.index)}')
            G = nx.scale_free_graph(row['number of nodes'], alpha=alpha,beta=beta,gamma=gamma,seed=args.seed)
            D = from_multigraph_to_graph(G)
            U = D.to_undirected()
            if D.order()<=500:
                plt.figure()
                plt.title(f'Scale Free with Size {D.order()} and Beta {beta}')
                nx.draw(D, node_size=50)
                plt.savefig(f'Figures/Scale_Free/Scale Free with Size {D.order()} and Beta {beta}.png')
            print(f'Generated Scale Free, Size: {D.order()} and Beta {beta}')

            ##  Calculate the local curvatures

            ## to do: scale_free_graph is MultiDiGraph and that is not compatible with get_ollivier_curvature
            ollivier = get_ollivier_curvature(D)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(D)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(U)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
        df.to_csv('Results/Scale_Free/scale_free_curvature.csv', index=False)
    
    if 'bipartite' in args.nargs or 'all' in args.nargs:
        df = get_bipartite_dataframe()
        for index, row in df.iterrows():
            print(f'{index+1}/{len(df.index)}')
            G = bipartite.random_graph(row['number of nodes in first bipartite set'], row['number of nodes in second bipartite set'], 0.5, seed=args.seed)
            if G.order()<=500:
                plt.figure()
                plt.title(f'Bipartite with Size {G.order()}')
                nx.draw(G)
                plt.savefig(f'Figures/Bipartite with Size {G.order()}')
            print(f'Results/Generated Bipartite Size: {G.order()}')
            df.loc[index, 'total number of nodes']=G.order()
            ##  Calculate the local curvatures
            ollivier = get_ollivier_curvature(G)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(G)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(G)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
        df.to_csv('Results/bipartite_curvature.csv', index=False)

    if 'dag' in args.nargs or 'all' in args.nargs:
        df = get_dag_dataframe()
        for index, row in df.iterrows():
            print(f'{index+1}/{len(df.index)}')
            G = get_random_dag(row['number of nodes'], seed = args.seed)
            U = G.to_undirected()
            if U.order()<=500:
                plt.figure()
                plt.title(f'DAG with Size {G.order()}')
                nx.draw(G, node_size=50)
                plt.savefig(f'Figures/DAG/DAG with Size {G.order()}')
            print(f'Generated DAG, Size: {G.order()}')
            ##  Calculate the local curvatures
            ollivier = get_ollivier_curvature(G)
            df.loc[index, 'ollivier']=ollivier
            print("ollivier: ", ollivier)
            forman = get_forman_curvature(G)
            df.loc[index, 'forman']=forman
            print("forman: ", forman)
            #   Sectional Curvature estimation based on our implementation for HyperKGQA
            sectional, _ = sectional_curvature(U)
            df.loc[index, 'sectional']=sectional
            print("sectional: ", sectional)
        df.to_csv('Results/DAG/dag_curvature.csv', index=False)
