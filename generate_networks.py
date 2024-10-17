import random
import numpy as np
import networkx as nx
from cdlib import NodeClustering, evaluation

from my_module import *


def generate_LFR(dir_path_LFR, n, graphs_per_category, mus):
    """
        Generate LFR graphs with the following parameters:
        - n: number of nodes
        - tau1: Power law exponent for degree distribution (>1)
        - tau2: Power law exponent for community size distribution (>1)
        - mu: mixing parameter, fraction of inter-community edges for each node (in 
            interval [0,1])
        - average_degree: average degree (either this or min_deg needs to be given)
        - max_degree: maximum degree
        - min_community: minimum community size
    """
    
    tau1 = 2
    tau2 = 2.5
    average_degree = 20
    max_degree = 100
    min_community = 20 
    seed = 0

    for mu in mus:
        success = 0

        while success < graphs_per_category:
            random_state = np.random.RandomState(seed)

            try:
                G = nx.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, 
                    average_degree=average_degree, max_degree=max_degree, 
                    min_community=min_community, seed=random_state)

                communities = list({frozenset(G.nodes[v]["community"]) for v in G})
                communities.sort(key=len, reverse=True)
                
                for node in G.nodes():
                    for i, c in enumerate(communities):
                        if node in c:
                            G.nodes[node]['ground_truth'] = i
                    del G.nodes[node]['community']

                success += 1
                store_network(G, f'SG_mu{int(mu*10)}_s{seed}', dir_path_LFR)

            except Exception as error:
                print(error)

            seed += 1

if __name__ == '__main__':
    dir_path_LFR = 'data/synthetic/LFR_small_246'
    generate_LFR(dir_path_LFR, 1000, 50, [0.2, 0.4, 0.6])

    dir_path_LFR = 'data/synthetic/LFR_large_246'
    generate_LFR(dir_path_LFR, 10000, 5, [0.2, 0.4, 0.6])
