import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.algorithms.dag import *

from utils import *


def generate_tree(depth = 2, maxbranch = 3, create_using = None, seed = None, draw = False, save_file = 'hierarchy.png'):
    '''
    generate a balanced tree with maximum depth and maximum branching factor (the actual branching factor will be random for each vertex)
    if creating a directed balanced tree, set create_using = nx.DiGraph
    '''
    random.seed(seed)
    d = 0
    G = nx.empty_graph(1, create_using)
    leaves = [0]
    while d<depth:
        next_leaves = []
        for leaf in leaves:
            branch = random.randint(1, maxbranch)
            children = [G.order()+i for i in range(branch)]
            G.add_edges_from([(leaf, child) for child in children])
            next_leaves.extend(children)
        d+=1
        leaves = next_leaves
    if draw:
        pos = hierarchy_pos(G,0)
        nx.draw(G, pos=pos, with_labels=True)
        plt.savefig(save_file)
    return G

def get_random_dag(n, p = 0.5, seed = None):
    '''generates random DAGs, taken from https://gist.github.com/flekschas/0ea70dec4d92bc706e61'''
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True, seed = seed)
    random_dag = nx.DiGraph([(u, v) for (u, v) in random_graph.edges() if u < v])
    random.seed()
    while random_dag.order() < n:
        node1 = random.randint(0,random_graph.order())
        node2 = random.randint(0,random_graph.order())
        while node1 == node2 or (node1, node2) in random_dag.edges():
            node1 = random.randint(0,random_graph.order())
            node2 = random.randint(0,random_graph.order())
        random_dag.add_edge(node1, node2)
        if is_directed_acyclic_graph(random_dag)==False:
            random_dag.remove_edge(node1, node2)
    while random_dag.order() > n:
        node = random.randint(0,random_dag.order())
        random_dag.remove_node(node)
    return random_dag