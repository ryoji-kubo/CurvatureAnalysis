import networkx as nx
import random
import pandas as pd
import numpy as np
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
    beta_list =beta_list = np.linspace(0,1,10,endpoint=False)
    beta_list = np.delete(beta_list,[0])
    for i in [100, 1000, 10000]:
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
    for i in [10, 100, 1000, 2000]:
        big_list.append([i, 0, 0, 0])
    df = pd.DataFrame(big_list, columns=['number of nodes','ollivier','forman','sectional'])
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