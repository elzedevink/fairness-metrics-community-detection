import os

from generate_networks import generate_LFR
from set_communities import *
from get_results import *
from create_figures import *

# generate networks with LFR benchmark
def generate_nets(path_data, n, graphs_per_category):
    mus = [0.2, 0.4, 0.6]
    generate_LFR(path_data, n, graphs_per_category, mus)


# apply community detection methods
def apply_detection_methods(path_data, path_data_cdms):
    files = os.listdir(path_data)
    net_names = sorted(list(set([file[:-10] for file in files 
        if file[-10:] == '_nodes.csv'])))
    
    print('Networks run:', net_names)
    for net_name in net_names:
        apply_cdms(net_name, path_data, path_data_cdms)


# calculate fairness metric Phi and CDM evaluation metrics
def calculate_results(path_data_cdms, path_results, show, store):
    files = os.listdir(path_data_cdms)
    net_names = sorted(list(set([file[:-10] for file in files 
        if file[-10:] == '_nodes.csv'])))
    
    print('Networks run:', net_names)
    for net_name in net_names:
        result_hub(path_data_cdms, net_name, path_results, show=show, store=store)


def create_figure(res_path, fig_path, combined, acc_measure='nmi'):
    """
    :param res_path: directory where results are found 
    :param fig_path: directory where figures will be stored
    :param combined: create figures per network or combine all results from 
        given directory res_path 
    :param acc_measure: accuracy metric from ['nmi', 'ari', 'nf1', 'rmi']
    """
    separate_by_mu = True # separate results based on mixing parameter mu

    if combined:
        use_results_combined_networks(res_path, fig_path, acc_measure, separate_by_mu)
    else:
        use_results_per_network(res_path, fig_path, acc_measure)


def LFR_example():
    # synthethic data
    dir_name = 'synthetic/example_LFR_1000'
    path_data = 'data/' + dir_name
    path_data_cdms = 'data_applied_methods/' + dir_name
    path_results = 'results/' + dir_name
    path_figures = 'figures/' + dir_name

    paths = [path_data, path_data_cdms, path_results, path_figures]
    create_paths(paths)

    generate_nets(path_data)
    apply_detection_methods(path_data, path_data_cdms)
    calculate_results(path_data_cdms, path_results, show=False, store=True)
    
    # create figures for combined results of same mu using error bars of standard deviation
    create_figure(res_path=path_results, fig_path=path_figures, combined=True)

    # create a figure for each synthetic network
    # create_figure(res_path=path_results, fig_path=path_figures, combined=False)


def real_world_results():
    # real-world data
    real_path_data = 'data/real_world'
    real_path_data_cdms = 'data_applied_methods/real_world'
    real_path_results = 'results/real_world'
    real_path_figures = 'figures/real_world'

    real_paths = [real_path_data, real_path_data_cdms, real_path_results, real_path_figures]
    create_paths(real_paths, n=1000, graphs_per_category=2)

    apply_detection_methods(real_path_data, real_path_data_cdms)
    calculate_results(real_path_data_cdms, real_path_results, show=False, store=True)
    create_figure(res_path=real_path_results, fig_path=real_path_figures, combined=False)


if __name__ == '__main__':
    LFR_example()
    real_world_results()    