import os
import json
import pickle
import pandas as pd
import networkx as nx
from my_module import *
from cdlib import NodeClustering, evaluation


def get_gt_info(G, ground_truth):
    """
    Provides dictionary of ground-truth community values for size, density, conductance

    :param G: NetworkX graph
    :param ground_truth: NodeClustering object with ground-truth communities
    :return info_dict: dict containing lists of community attribute values
    """
    sizes = [len(com) for com in ground_truth.communities]
    densities = evaluation.internal_edge_density(G, ground_truth, summary=False)
    conductances = evaluation.conductance(G, ground_truth, summary=False)

    info_dict = {
        'size': sizes,
        'density': densities,
        'conductance': conductances
    }
    
    return info_dict


def get_metric_scores(G, ground_truth, cdm):
    """
    Returns dictionary with metric scores FCCN, F1, FCCE

    :param G: NetworkX graph
    :param ground_truth: NodeClustering object containing the ground-truth communities
    :param cdm: NodeClustering object of the CDM's community assignments
    :return metric_results: dictionary of the metric scores
    """
    pred_coms = cdm.communities

    # mapping is sorted by gt coms: 0, ..., |C|-1
    mapping = iterative_mapping(ground_truth, cdm)

    mapped_metrics = []
    for gt, pred in mapping:
        if pred != None:
            pred_com = cdm.communities[pred]
        else:
            pred_com = None
        scores = calc_mapped_scores(G, ground_truth.communities[gt], pred_com)
        mapped_metrics.append(scores)

    metric_scores = {
        'FCCNs': [metrics['FCCN'] for metrics in mapped_metrics],
        'F1s': [metrics['F1'] for metrics in mapped_metrics],
        'FCCEs': [metrics['FCCE'] for metrics in mapped_metrics]
    }
    return metric_scores


def get_fm_results(metric_scores, x, ground_truth, cdm):
    """
    Returns dictionary with fairness metric scores for several mapped/global metrics

    :param metric_scores: Dictionary containing metric scores
    :param x: attribute values of communities
    :return fm_results: dictionary of the fairness metric scores
    """
    fm_scores = {
        'Phi_FCCN': calc_fairness_metric(x, metric_scores['FCCNs']),
        'Phi_F1': calc_fairness_metric(x, metric_scores['F1s']),
        'Phi_FCCE': calc_fairness_metric(x, metric_scores['FCCEs']),

        'nmi': ground_truth.normalized_mutual_information(cdm).score,
        'ari': ground_truth.adjusted_rand_index(cdm).score,
        'nf1': ground_truth.nf1(cdm).score,
        'rmi': ground_truth.rmi(cdm, norm_type='normalized').score
    }
    return fm_scores

def result_hub(data_path, net_name, res_path, show=False, store=True):
    """
    Collects results and puts them in result_dict to be stored
    Results are gathered for all communities and stored once per network

    :param data_path: path to where network data is stored
    :param net_name: name of network
    :param res_path: path to where results are going to be stored
    """
    print(f'\nNetwork: {net_name}')

    G, node_clustering_dict = get_network_communities(data_path, net_name)
    
    ground_truth = node_clustering_dict['ground_truth']
    gt_info = get_gt_info(G, ground_truth)

    attributes = ['size', 'density', 'conductance']

    all_result_dict = {
        'size': {},
        'density': {},
        'conductance': {},
    }

    for cdm_name in node_clustering_dict:
        if cdm_name == 'ground_truth':
            continue
        cdm = node_clustering_dict[cdm_name]

        if not cdm:
            # invalid method
            fm_scores = {
                'Phi_FCCN': None,
                'Phi_F1': None,
                'Phi_FCCE': None,

                'nmi': None,
                'ari': None,
                'nf1': None,
                'rmi': None,
            }
            for attr in attributes:
                all_result_dict[attr][cdm_name] = fm_scores
        else:
            metric_scores = get_metric_scores(G, ground_truth, cdm)

            for attr in attributes:
                x = np.array(gt_info[attr])
                fm_scores = get_fm_results(metric_scores, x, ground_truth, cdm)
                all_result_dict[attr][cdm_name] = fm_scores

    for attr in attributes:
        filename = f'{res_path}/res_{attr[:4]}_{net_name}.pkl'
        if show:
            print(attr)
            print(json.dumps(all_result_dict[attr], indent=4))
        if store:
            print(f'saving {filename}')
            with open(filename, 'wb') as handle:
                pickle.dump(all_result_dict[attr], handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f'NOT saving {filename}\n')


if __name__ == '__main__':
    p = Parser()
    dir_name = p.directory
    net_name = p.network
    dir_path_input = path_prefix(dir_name, 'data_applied_methods')
    res_path = f'results/{remove_prefix(dir_name, "data_applied_methods/")}'

    print('start get_results.py')
    print(f'dir_path_input = {dir_path_input}, res_path = {res_path}, net_name = {net_name}')
    
    if not os.path.exists(dir_path_input):
        print('Directory does not exist:', dir_path_input)
        print('Did you forget "data_applied_methods"?')
        exit()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if net_name == None:
        # get all network names in dir_path_input        
        files = os.listdir(dir_path_input)
        net_names = sorted(list(set([file[:-10] for file in files 
            if file[-10:] == '_nodes.csv'])), reverse=False)

        print('Networks run:', net_names)
        for net_name in net_names:
            result_hub(dir_path_input, net_name, res_path)
    else:
        # use only the provided net_name (using -n ...)
        result_hub(dir_path_input, net_name, res_path)