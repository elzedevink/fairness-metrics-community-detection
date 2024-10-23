import os
import pickle
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl

from my_module import *

attributes = ['size', 'density', 'conductance']
fairness_metrics = ['Phi_FCCN', 'Phi_F1', 'Phi_FCCE']

optimization_cdms = {
    'name': 'optimization',
    'methods': ['CNM', 'Combo', 'Leiden', 'Louvain', 'Paris', 'RB-C', 
        'RB-ER', 'Significance'],
    'marker_idx': 0
}
spectral_cdms = {
    'name': 'spectral',
    'methods': ['Eigenvector', 'RSC-K', 'RSC-SSE', 'RSC-V', 'Spectral'],
    'marker_idx': 1
}
representational_cdms = {
    'name': 'representational',
    'methods': ['Deepwalk', 'Fairwalk', 'Node2Vec'],
    'marker_idx': 2
}
dynamics_cdms = {
    'name': 'dynamics',
    'methods': ['Infomap', 'Spinglass', 'Walktrap'],
    'marker_idx': 3
}
propagation_cdms = {
    'name': 'propagation',
    'methods': ['Fluid', 'Label Propagation'],
    'marker_idx': 4
}
miscellaneous_cdms = {
    'name': 'miscellaneous',
    'methods': ['EM','SBM', 'SBM - Nested'],
    'marker_idx': 5
}

cdm_groups = [optimization_cdms, spectral_cdms, representational_cdms, 
    dynamics_cdms, propagation_cdms, miscellaneous_cdms]


def get_cdm_group(cdm_name):
    for group in cdm_groups:
        if cdm_name in group['methods']:
            return group
    print('no group found')
    return None


def fm_accuracy_figure(results, metric, attribute, acc_measure, fig_path, 
        net_name=None, error_bar=None, mu=None):
    """
    """
    alpha = 1

    colors = sns.color_palette('bright')
    markers = ['o', 's', '^', '*', 'D', '>']
    marker_sizes = [13, 13, 13, 15, 13, 13]
    fontsize_ticks = 24
    fontsize_label = 26

    cdms = list(results.keys())
    x = [results[cdm][metric] for cdm in cdms]
    y = [results[cdm][acc_measure] for cdm in cdms]
    metric_str = metric[4:]

    fig, ax = plt.subplots()

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # vertical line

    ax.set_xlim(-.7, .7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f'$\\Phi^{{{metric_str}}}_{{{attribute}}}$', fontsize=fontsize_label+4)
    ax.set_ylabel(acc_measure.upper(), fontsize=fontsize_label)        
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    
    for i in range(len(x)):
        color_idx = i%10
        group = get_cdm_group(list(cdms)[i])
        marker = markers[group['marker_idx']]
        markersize = marker_sizes[group['marker_idx']]
        
        if error_bar:
            x_err = [error_bar[cdm][metric] for cdm in cdms]
            y_err = [error_bar[cdm][acc_measure] for cdm in cdms]
            marker, _, bars = ax.errorbar(x[i], y[i], xerr=x_err[i], yerr=y_err[i], lw=2, 
                marker=marker, markersize=markersize, color=colors[color_idx], 
                alpha=alpha, label=list(cdms)[i])
                        
            [bar.set_alpha(0.5) for bar in bars]

        else: # no error bar
            ax.scatter(x[i], y[i], marker=marker, s=markersize**2,
                color=colors[color_idx], alpha=alpha, label=list(cdms)[i])

    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05, 1, 0.2), mode='expand', 
        ncol=4, borderaxespad=0)
    
    create_paths([f'{fig_path}/{acc_measure}/'])
    if error_bar:
        fig_pre = f'{fig_path}/{acc_measure}/'
    else:
        fig_pre = f'{fig_path}/{acc_measure}/{net_name}_'

    if mu:
        fig_str = f'{fig_pre}{attribute[:4]}_mu{mu}_{metric}.png'
    else:
        fig_str = f'{fig_pre}{attribute[:4]}_{metric}.png'

    save_or_show_fig(fig_str, True)


# filters out CDMs that have encountered an error when assigning communities
def error_method_filter(result_dict, attr):
    cdms = result_dict.keys()

    to_delete = []
    for cdm in cdms:
        phi_F1 = result_dict[cdm]['Phi_F1']
        if phi_F1 == None:
            to_delete.append(cdm)

    if to_delete:
        print(f'Deleted {len(to_delete)} from {attr} results')
        print(to_delete)

    for cdm in to_delete:
        del result_dict[cdm]

    return result_dict


def use_results_per_network(res_path, fig_path, acc_measure):
    files = os.listdir(res_path)
    net_names = sorted(list(set([file[9:-4] for file in files 
        if file[-4:] == '.pkl'])))
    print('Networks run:', net_names)

    for net_name in net_names:
        for attr in attributes:
            with open(f'{res_path}/res_{attr[:4]}_{net_name}.pkl', 'rb') as handle:
                result_dict = pickle.load(handle)

            result_dict = error_method_filter(result_dict, attr)
            if result_dict == {}:
                print(f'No results for {net_name}, regarding {attr}')
                continue

            cdms = result_dict.keys()

            for fairness_metric in fairness_metrics:
                fm_accuracy_figure(result_dict, fairness_metric, attr, acc_measure, 
                    fig_path, net_name=net_name)


def combine_results(file_names, res_path):
    result_dict_list = []
    for file_name in file_names:
        with open(res_path+'/'+file_name, 'rb') as handle:
            result_dict = pickle.load(handle)
        result_dict_list.append(result_dict)

    return result_dict_list


def get_avg_std_results(result_dict_list, cdms, metrics):
    avg_dict = {}
    std_dict = {}
    for cdm in cdms:
        
        cdm_metric_score = {metric:[] for metric in metrics}

        for i in range(len(result_dict_list)):
            if not result_dict_list[i][cdm][metrics[1]]:
                continue
            for metric in metrics:
                cdm_metric_score[metric].append(result_dict_list[i][cdm][metric])

        cdm_metric_avg = {metric: np.mean(cdm_metric_score[metric]) for metric in metrics}
        cdm_metric_std = {metric: np.std(cdm_metric_score[metric]) for metric in metrics}

        avg_dict[cdm] = cdm_metric_avg
        std_dict[cdm] = cdm_metric_std

    return avg_dict, std_dict


def create_combined_fig(attr, file_names, res_path, fig_path, acc_measure, mu=None):
    result_dict_list = combine_results(file_names, res_path)

    cdms = result_dict_list[0].keys()
    metrics = fairness_metrics.copy()
    metrics.append(acc_measure)
            
    avg_dict, std_dict = get_avg_std_results(result_dict_list, cdms, metrics)

    for fm_metric in metrics[:-1]: # don't include acc_measure
        fm_accuracy_figure(avg_dict, fm_metric, attr, acc_measure, fig_path, 
            error_bar=std_dict, mu=mu)


def use_results_combined_networks(res_path, fig_path, acc_measure, separate_by_mu):

    for attr in attributes:
        print(attr)

        files = os.listdir(res_path)
        file_names_total = sorted([file for file in files if attr[:4] in file])

        # separate results by mu value
        if separate_by_mu:
            mus = list(set([int(f[f.find('mu')+2]) for f in file_names_total]))

            for mu in mus:
                file_names = [file_name for file_name in file_names_total 
                    if 'mu'+str(mu) in file_name]
                create_combined_fig(attr[:4], file_names, res_path, fig_path,
                    acc_measure=acc_measure, mu=mu)
        else:
            create_combined_fig(attr[:4], file_names_total, res_path, fig_path,
                acc_measure=acc_measure)
    return