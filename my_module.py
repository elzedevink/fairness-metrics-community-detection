import os
import csv
import sys
import random
import pathlib
import warnings
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from cdlib import NodeClustering

import gensim
from sklearn.cluster import KMeans
from fairwalk import FairWalk
from node2vec import Node2Vec

warnings.filterwarnings('error', category=np.RankWarning)

def requires_directory():
    python_file = sys.argv[0]
    if python_file == 'generate_networks.py':
        return False
    return True

# for parsing directory and seed information when calling separate code files
class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Description for my parser')
        parser.add_argument('-d', '--directory', help='Give directory name', 
            required=requires_directory(), type=str)
        parser.add_argument('-n', '--network', help='Give network name', required=False, 
            default=None)
        
        p = parser.parse_args()
        self.directory = p.directory
        self.network = p.network


def print_network_info(G, net_name, gt_coms=None):
    # print network data
    print(net_name)
    print('num nodes', len(G.nodes))
    print('num edges', len(G.edges))
    print('avg degree', np.average([G.degree[n] for n in G.nodes]))
    print('highest degree', max([G.degree[n] for n in G.nodes]))
    print('lowest degree', min([G.degree[n] for n in G.nodes]))
    if gt_coms:
        print('num communities', len(gt_coms))
        print('largest community', max([len(c) for c in gt_coms]))
        print('smallest community', min([len(c) for c in gt_coms]))
    return


# The graph is stored in two csv files:
# - 'name'_edges.csv: adjlist of nodes
# - 'name'_nodes.csv: list of nodes and their communities
def store_network(G, name, path):
    file_str = path + '/' + name

    nx.write_adjlist(G, file_str+'_edges.csv', delimiter=',')
    community = nx.get_node_attributes(G, 'ground_truth')
    
    n = len(G.nodes)
    num_com = max(list(community.values()))+1

    com_csv = [['id', 'ground_truth']]
    for i in range(n):
        com_csv.append([i, community[i]])
    
    with open(file_str + '_nodes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(com_csv)

    print(f'Stored {file_str} with {n} nodes and {num_com} communities')


# returns for a given graph and algorithm the community assignments
# returns the communities as a list of lists of nodes
def get_assignment(df, alg):
    df = df[alg]

    if df.eq(-1).all():
        # invalid method
        return None

    assignment = []
    for com in range(df.max()+1):
        ids = df.index[df == com]
        assignment.append(list(ids))

    return assignment


def get_node_clustering_dict(G, df):
    algs = df.columns

    node_clustering_dict = {}
    for alg in algs:
        assignment = get_assignment(df, alg)
        if not assignment:
            node_clustering_dict[alg] = None
            continue

        node_clustering_dict[alg] = NodeClustering(assignment, graph=G, method_name=alg)
    
    return node_clustering_dict


# returns networkx network from given directory, network name
def get_network_communities(dir_path, net_name, request_node_clustering=True):
    graph_path = dir_path+'/'+net_name
    try:
        H = nx.read_adjlist(graph_path+'_edges.csv', nodetype=int, delimiter=',')
        G = nx.Graph()
        G.add_nodes_from(sorted(H.nodes(data=True)))
        G.add_edges_from(H.edges(data=True))
    except FileNotFoundError:
        print(f'Network "{net_name}" does not exist.')
        exit()

    df = pd.read_csv(graph_path+'_nodes.csv', index_col=0)

    if request_node_clustering:
        return G, get_node_clustering_dict(G, df)
    else:
        comm_assignment_dict = df.to_dict('index')

    return G, comm_assignment_dict


def save_or_show_fig(figure_str, save_fig):
    if save_fig:
        print(f'save fig: {figure_str}')
        plt.savefig(figure_str, bbox_inches='tight', dpi=300)
    else:
        print(f'show fig: {figure_str}')
        plt.show()
    plt.close()


def create_paths(paths):
    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # print(a)


def jaccard_sim(a, b):
    a = set(a)
    b = set(b)

    return len(a.intersection(b)) / len(a.union(b))


def find_highest_set_zero(matrix):
    max_val = 0
    max_coord_options = []

    # Find the highest value in the matrix. If tied, add to options
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > max_val:
                max_val = matrix[i][j]
                max_coord_options = [(i, j)]
            elif matrix[i][j] == max_val:
                max_coord_options.append((i, j))

    if max_coord_options == []:
        return -1, None

    # random choice if tie
    max_coord = random.choice(max_coord_options)

    # Set the corresponding row and column to -1
    row, col = max_coord
    for i in range(len(matrix)):
        matrix[i][col] = -1
    for j in range(len(matrix[row])):
        matrix[row][j] = -1
    
    return max_val, max_coord


def iterative_mapping(source, target):
    mapping = []

    source_coms = source.communities
    target_coms = target.communities

    sim_matrix = []

    for source_com in source_coms:
        JS = [jaccard_sim(source_com, target_com) for target_com in target_coms]
        sim_matrix.append(JS)

    for _ in range(len(source_coms)):
        max_value, max_coord = find_highest_set_zero(sim_matrix)

        # more coms in source than in target, map to None, invalid
        if max_coord == None:
            to_map = set(range(len(source_coms)))
            already_mapped = set([source for source, target in mapping])
            still_to_map = list(to_map.difference(already_mapped))

            for s in still_to_map:
                mapping.append((s,None))
            break

        mapping.append(max_coord)

    # sorts mapping by source
    return sorted(mapping)


def calc_mapped_scores(G, gt_com, pred_com):
    """
        Given the ground-truth community (list) and predicted community (list),
        return the mapped scores in a dictionary
    """
    # no mapping
    if pred_com == None:
        mapping_scores = {
            'F1': 0,
            'FCCN': 0,
            'FCCE': 0
        }
        return mapping_scores

    set_gt_com = set(gt_com)
    set_pred_com = set(pred_com)

    TP = len(set_gt_com & set_pred_com)
    FN = len(set_gt_com) - TP
    FP = len(set_pred_com) - TP

    FCCN = TP / (TP + FN)
    F1 = (2 * TP) / (2 * TP + FP + FN)

    # calculate FCCE: Fraction of Correctly Classified Edges
    gt_SG = G.subgraph(gt_com)
    gt_edges = gt_SG.edges()

    edge_overlap = 0
    for edge_source, edge_target in gt_edges:
        if edge_source in pred_com and edge_target in pred_com:
            edge_overlap += 1

    if len(gt_edges) == 0:
        FCCE = 1
    else:
        FCCE = edge_overlap/len(gt_edges)

    mapping_scores = {
        'FCCN': FCCN,
        'F1': F1,
        'FCCE': FCCE
    }
    return mapping_scores


def calc_fairness_metric(x, y):
    """
    Calculate fairness metric for given attribute values and metric scores
    :param x: attribute values
    :param y: metric scores
    :return fm: fairness metric
    """
    try:
        if np.max(x) == np.min(x):
            return None
            
        x_norm = (x-np.min(x)) / (np.max(x) - np.min(x))
        a, b = np.polyfit(x_norm, y, 1)

        fm = (2 * np.arctan(a)/np.pi)

    except:
        fm = None

    return fm


def deepwalk_walks(H, walk_length, num_walks):
    '''
    Create random walks with deepwalk's method
    '''
    G = nx.Graph()
    G.add_nodes_from([str(n) for n in list(H.nodes)])
    G.add_edges_from([(str(e0), str(e1)) for e0, e1 in list(H.edges)])

    source_nodes = list(G)
    
    walks = []

    for _ in range(num_walks):

        for source in source_nodes:
            walk = [source]

            while len(walk) < walk_length:
                options = [n for n in G.neighbors(walk[-1])]
                next_node = np.random.choice(options)
                walk.append(next_node)
            walks.append(walk)

    return walks

def transform_sklearn_labels_to_communities(labels: list):
    pred_coms = [[] for i in range(len(np.unique(labels)))]
    for idx, label in enumerate(labels):
        pred_coms[label].append(idx)
    return pred_coms

def model_to_pred(G, model, n_clusters):
    emb_df = pd.DataFrame([model.wv.get_vector(str(n)) 
        for n in G.nodes()], index=G.nodes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_df)
    pred_coms = transform_sklearn_labels_to_communities(labels=kmeans.labels_)
    pred = NodeClustering(communities=pred_coms, graph=G)
    
    return pred

def walk_family(G, variant, n_clusters, dimensions, walk_length, num_walks):
    if variant == 'deepwalk':
        walks = deepwalk_walks(G, walk_length, num_walks) # walks is a list of walks (list of nodes)
        model = gensim.models.Word2Vec(walks, size=dimensions)

    elif variant == 'fairwalk':
        n = len(G.nodes())
        node2group = {node: group for node, group in zip(G.nodes(), 
            (5*np.random.random(n)).astype(int))}
        nx.set_node_attributes(G, node2group, 'group')

        model = FairWalk(G, dimensions=dimensions, walk_length=walk_length, 
            num_walks=num_walks, quiet=True)  # Use temp_folder for big graphs
        # model = model.fit(window=10, min_count=1, batch_words=4)
        model = model.fit()
    elif variant == 'node2vec':
        emb = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
            num_walks=num_walks, p=1, q=1, quiet=True)
        model = emb.fit() # model is a gensim.model.Word2Vec

    pred_coms = model_to_pred(G, model, n_clusters)
    return pred_coms


def remove_prefix(s, prefix):
    return s[len(prefix):] if s.startswith(prefix) else s


def path_prefix(dir_path, category, create=False):

    if category == 'data':
        input_prefix = 'data/'
    elif category == 'data_applied_methods':
        input_prefix = 'data_applied_methods/'
    elif category == 'results':
        input_prefix = 'results/'
    elif category == 'figures':
        input_prefix = 'figures/'
    else:
        print('Error: input correct category')
        exit()

    if input_prefix in dir_path:
        correct = dir_path
    else:
        correct = input_prefix + dir_path

    if not os.path.exists(correct):
        if not create:
            print('Directory does not exist:', correct)
            exit()
        else:
            os.makedirs(correct)
    return correct