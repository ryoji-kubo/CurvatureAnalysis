import networkx as nx
import random
import pandas as pd
import numpy as np
from networkx.algorithms.dag import *

"""
retrieved from: Aug 2nd 2021
https://newbedev.com/can-one-get-hierarchical-graphs-from-networkx-with-python-3
"""

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def get_tree_dataframe():
    big_list = []
    #depth = 2, 4, 6
    #branch = 3, 5, 7
    for i in range(3):
        for j in [3,5,7]:
            big_list.append([2*(i+1), j, 0, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['depth','branch','ollivier','forman','sectional','number of nodes'])
    return df

def get_balanced_tree_dataframe():
    big_list = []
    #height= 2, 4, 6
    #r-ary = 2, 3, 4
    for i in range(3):
        for j in [2,3,4]:
            big_list.append([2*(i+1), j, 0, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['height','r-ary','ollivier','forman','sectional','number of nodes'])
    return df

def get_star_graph_dataframe():
    big_list = []
    for i in [100, 1000, 10000]:
        big_list.append([i, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['number of nodes','ollivier','forman','sectional'])
    return df

def get_scale_free_dataframe():
    big_list = []
    beta_list = np.linspace(0,1,10,endpoint=False)
    beta_list = np.delete(beta_list,[0])
    for i in [10, 20, 100, 500, 1000, 10000]:
        for j in beta_list:
            big_list.append([i, j, (1-j)/2, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['number of nodes','beta','alpha gamma','ollivier','forman','sectional'])
    return df

def get_bipartite_dataframe():
    big_list = []
    small_list = [10, 100, 1000]
    for i in range(3):
        for j in range(3-i):
            index = i+j
            big_list.append([small_list[i], small_list[index], 0, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['number of nodes in first bipartite set', 'number of nodes in second bipartite set','ollivier','forman','sectional','total number of nodes'])
    return df

def get_dag_dataframe():
    big_list = []
    p_list = np.linspace(0,1,10,endpoint=False)
    p_list = np.delete(p_list,[0])
    for i in [10,20,100,500,1000]:
        for j in p_list:
            big_list.append([i, j, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['number of nodes','p','ollivier','forman','sectional'])
    return df

def get_random_digraph_dataframe():
    big_list = []
    nodes = [20,100,500,1000]
    for n in nodes:
        # edge_list created to get the number of edges spaced out from n/10*(n/10 - 1) ~ nC2 
        edge_list = np.linspace(n/10, n, num=10, dtype= int)
        # print(f'{n}: ', edge_list)
        edge_list = 1/2*np.multiply(edge_list,edge_list-1)
        # print(f'{n}: ', edge_list)
        for edge in edge_list:
            big_list.append([n,edge,0,0,0])
    columns = ['number of nodes','number of edges','ollivier','forman','sectional']
    df = pd.DataFrame(big_list, columns=columns)
    return df

def get_dag_tree_dataframe():
    big_list = []
    nodes = [20,100,500,1000]
    rates = [1,5,10]
    for n in nodes:
        for r in rates:
            edge_list = np.linspace(n/10, n, num=10, dtype= int)
            for e in edge_list:
                big_list.append([n,r,e,0,0,0])
    columns = ['number of nodes','rate','new edges','ollivier','forman','sectional']
    df = pd.DataFrame(big_list, columns=columns)
    return df

def get_random_tree_dataframe():
    big_list = []
    nodes = [20,100,500,1000]
    for n in nodes:
        edge_list = np.linspace(n/10, n, num=10, dtype= int)
        for e in edge_list:
            big_list.append([n,e,0,0,0])
    columns = ['number of nodes','new edges','ollivier','forman','sectional']
    df = pd.DataFrame(big_list, columns=columns)
    return df

def from_multigraph_to_graph(M):
    G = nx.DiGraph()
    for edge in M.edges():
        if edge[0]==edge[1]:
            continue
        if edge in G.edges():
            continue
        G.add_edge(*edge)
    return G

def add_random_edges_dag(G,n):
    if is_directed_acyclic_graph(G) == False:
        raise ValueError('Expected DAG')
    e = 0
    while e < n:
        node1 = random.choice(list(G.nodes()))
        node2 = random.choice(list(G.nodes()))
        while node1 == node2 or (node1, node2) in G.edges():
            node1 = random.choice(list(G.nodes()))
            node2 = random.choice(list(G.nodes()))
        G.add_edge(node1, node2)
        e+=1
        if is_directed_acyclic_graph(G)==False:
            G.remove_edge(node1, node2)
            e-=1
    return G
