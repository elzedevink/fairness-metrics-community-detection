import os

from generate_networks import generate_LFR
from set_communities import *
from get_results import *

# generate networks with LFR benchmark
def generate_nets(path_data):
    n = 1000
    graphs_per_category = 1 # 5
    mus = [0.2] # [0.2, 0.4, 0.6]
    generate_LFR(path_data, n, graphs_per_category, mus)


# apply community detection methods
def apply_detection_methods(path_data, path_data_cdms):
    files = os.listdir(path_data)
    net_names = sorted(list(set([file[:-10] for file in files 
        if file[-10:] == '_nodes.csv'])))
    
    print('Networks run:', net_names)
    for net_name in net_names:
        apply_cdms(net_name, path_data, path_data_cdms)


def show_results(path_data_cdms, path_results):
    files = os.listdir(path_data_cdms)
    net_names = sorted(list(set([file[:-10] for file in files 
        if file[-10:] == '_nodes.csv'])))
    
    print('Networks run:', net_names)
    for net_name in net_names:
        result_hub(path_data_cdms, net_name, path_results, show=True, store=True)


if __name__ == '__main__':
    # synthethic data
    path_data = 'data/synthetic/LFR_large_246'
    path_data_cdms = 'data_applied_methods/synthetic/LFR_large_246'
    path_results = 'results/synthetic/LFR_large_246'

    generate_nets(path_data)
    apply_detection_methods(path_data, path_data_cdms)
    show_results(path_data_cdms, path_results)

    # real-world data
    real_path_data = 'data/real_world'
    real_path_data_cdms = 'data_applied_methods/real_world'
    real_path_results = 'results/real_world'

    apply_detection_methods(real_path_data, real_path_data_cdms)
    show_results(real_path_data_cdms, real_path_results)