import os
import csv
import networkx as nx
from cdlib import ensemble, NodeClustering, algorithms as alg

from my_module import *


def check_validity(G, com):
    """
    Checks whether the graph and community assignments are valid, meaning:
        - G is fully a fully connected, undirected graph
        - Every node in G is in a community
        - No node is in multiple communities
    This is only used in testing, to double-check that the CDMs work properly
    All synthethic graphs are fully connected and undirected
    All CDMs are crisp (no-overlap) and fully covering (all nodes in a community).

    :param G: NetworkX graph
    :param com: NodeClustering object containing community assignments
    :return: True if valid. False if invalid
    """
    combined_coms = [node for comm in com.communities for node in comm]
    if G.is_directed():
        print('Error: graph not directed')
        return False
    elif not nx.is_connected(G):
        print('Error: graph not fully connected')
        return False
    elif len(combined_coms) != len(G):
        if len(combined_coms) < len(G):
            print(f'Error: Not enough nodes placed in community by {com.method_name}')
        else:
            if len(combined_coms) != len(set(combined_coms)):
                if len(combined_coms) > len(set(combined_coms)):
                    print(len(combined_coms))
                    print(len(set(combined_coms)))
                    print(f'Error: Overlap detected by {com.method_name}')
                else:
                    print(f'Error: Not fully covering the graph by {com.method_name}')
        return False

    return True

def add_community_alg(comm_assignment_dict, com):
    """
    Add community assignment from a new CDM to dictionary

    :param comm_assignment_dict: dictionary of community assignments
    :param com: NodeClustering object containing community assignment to be added, if com is not
        a NodeClustering object, this cdm raised an error.
    :return: changed comm_assignment_dict with 'com' added
    """
    if not isinstance(com, NodeClustering):
        # crash when running method or overlap/not fully covering
        # com is string of the failed method
        print(f'Did not add {com}')

        for node in comm_assignment_dict.keys():
            comm_assignment_dict[node][com] = -1
        return comm_assignment_dict
    
    for community in range(len(com.communities)):
        node_list_per_com = com.communities[community]
        for node in node_list_per_com:
            comm_assignment_dict[node][com.method_name] = community

    return comm_assignment_dict


def save_new_assignment(G, file_path, comm_assignment_dict):
    """
    Store the network with community assignments in an easily readable way for getting 
        results and Gephi analysis.
    
    :param G: NetworkX graph
    :param file_path: path including network name to store to
    :param comm_assignment_dict: community assignments for all nodes, CDMs
    """
    print(f'saving {file_path}')
    method_list = list(comm_assignment_dict[0].keys())

    com_csv = [['id']]
    com_csv[0].extend(method_list)

    for n in list(G.nodes()):
        row = [n]
        row.extend(list(comm_assignment_dict[n].values()))
        com_csv.append(row)
        
    with open(file_path+'_nodes.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(com_csv)

    nx.write_adjlist(G, file_path+'_edges.csv', delimiter=',')


def get_cdms(num_gt_coms):
    # settings for all in walking family
    dimensions = 128
    walk_length = 80
    num_walks = 10

    # alphabetically sorted in category
    cdm_parameter = [
        # optimization
        (alg.greedy_modularity, [], 'CNM'),
        (alg.pycombo, [], 'Combo'),
        (alg.leiden, [], 'Leiden'),
        (alg.louvain, [], 'Louvain'),
        (alg.paris, [], 'Paris'),
        (alg.rb_pots, [], 'RB-C'),
        (alg.rber_pots, [], 'RB-ER'),
        (alg.significance_communities, [], 'Significance'),

        # spectral
        (alg.eigenvector, [], 'Eigenvector'),
        (alg.r_spectral_clustering, [ensemble.Parameter(name='method', 
            start='regularized_with_kmeans'), ensemble.Parameter(name='n_clusters', 
            start=num_gt_coms)], 'RSC-K'),
        (alg.r_spectral_clustering, [ensemble.Parameter(name='method', 
            start='sklearn_spectral_embedding'), ensemble.Parameter(name='n_clusters', 
            start=num_gt_coms)], 'RSC-SSE'),
        (alg.r_spectral_clustering, [ensemble.Parameter(name='method', 
            start='vanilla'), ensemble.Parameter(name='n_clusters', 
            start=num_gt_coms)], 'RSC-V'),
        (alg.spectral, [ensemble.Parameter(name='kmax', start=num_gt_coms)], 'Spectral'),

        # representational
        (walk_family, [ensemble.Parameter(name='variant', start='deepwalk'), 
            ensemble.Parameter(name='n_clusters', start=num_gt_coms), 
            ensemble.Parameter(name='dimensions', start=dimensions), 
            ensemble.Parameter(name='walk_length', start=walk_length), 
            ensemble.Parameter(name='num_walks', start=num_walks)], 'Deepwalk'),
        (walk_family, [ensemble.Parameter(name='variant', start='fairwalk'), 
            ensemble.Parameter(name='n_clusters', start=num_gt_coms), 
            ensemble.Parameter(name='dimensions', start=dimensions), 
            ensemble.Parameter(name='walk_length', start=walk_length), 
            ensemble.Parameter(name='num_walks', start=num_walks)], 'Fairwalk'),
        (walk_family, [ensemble.Parameter(name='variant', start='node2vec'), 
            ensemble.Parameter(name='n_clusters', start=num_gt_coms), 
            ensemble.Parameter(name='dimensions', start=dimensions), 
            ensemble.Parameter(name='walk_length', start=walk_length), 
            ensemble.Parameter(name='num_walks', start=num_walks)], 'Node2Vec'),

        # dynamics
        (alg.infomap, [], 'Infomap'),
        (alg.spinglass, [], 'Spinglass'),
        (alg.walktrap, [], 'Walktrap'),

        # propagation
        (alg.async_fluid, [ensemble.Parameter(name='k', start=num_gt_coms)], 'Fluid'),
        (alg.label_propagation, [], 'Label Propagation'),
        
        # other
        (alg.em, [ensemble.Parameter(name='k', start=num_gt_coms)], 'EM'),
        (alg.sbm_dl, [], 'SBM'),
        (alg.sbm_dl_nested, [], 'SBM - Nested'),
    ]
    print(f'Number of methods: {len(cdm_parameter)}')

    return cdm_parameter


def apply_cdms(net_name, dir_path_input, dir_path_output):
    """
    Applies the set of CDMs to the given network. The assignments are stored 
        in comm_assignment_dict. Then save_new_assignment() is called which saves them.

    :param net_name: network name
    :param dir_path_input: directory of network name
    :param dir_path_output: directory where the network with applied CDMs goes
    """
    print(f'\nload {net_name}')

    # read G and dictionary of ground-truth communities
    G, comm_assignment_dict = get_network_communities(dir_path_input, net_name, 
        request_node_clustering=False)

    num_gt_coms = max([comm_assignment_dict[node]['ground_truth'] 
        for node in comm_assignment_dict]) + 1
    
    cdm_parameter = get_cdms(num_gt_coms)
    
    cdms = [cdm for cdm, _, _ in cdm_parameter]
    parameters = [param for _, param, _ in cdm_parameter]
    method_names = [name for _, _, name in cdm_parameter]

    for i, method in enumerate(cdms):
        values = {}
        
        for param in parameters[i]:
            values[param.name] = param.start
        try:
            com = method(G, **values)
            com.method_name = method_names[i]

            if not check_validity(G, com):
                # not valid
                comm_assignment_dict = add_community_alg(comm_assignment_dict, 
                    method_names[i])
                continue

            print(f'\tadded\t{com.method_name}')
            comm_assignment_dict = add_community_alg(comm_assignment_dict, com)

        except Exception as e:
            print(repr(e))
            comm_assignment_dict = add_community_alg(comm_assignment_dict, method_names[i])
    
    file_path = dir_path_output+'/'+net_name
    save_new_assignment(G, file_path, comm_assignment_dict)