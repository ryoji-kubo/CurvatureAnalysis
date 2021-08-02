import networkx as nx
import matplotlib.pyplot as plt
import random

from utils import *


def generate_tree(depth = 2, maxbranch = 3, seed = None, create_using = None, draw = False, save_file = 'hierarchy.png'):
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

