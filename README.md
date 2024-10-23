# Group Fairness Metrics for Community Detection Methods in Social Networks

[de Vink, E., Saxena, A.: Group fairness metrics for community detection methods in social networks (2024).](https://arxiv.org/abs/2410.05487)

## Files
**generate_networks.py**: Generates LFR benchmark networks

**set_communities.py**: Applies community detection methods to network data

**get_results.py**: Calculates fairness metric Phi for size, density, and conductance

**create_figures.py**: gathers results, creates figures

**my_module.py**: Helper functions

**main.py**: Shows example of LFR network generation, CDM application and and prints fairness metric and performance values

## Data folders
**data**: network data in csv files showing ground-truth communities in 'networkname_nodes.csv' and edge adjacency list in 'networkname_edges.csv'

**data_applied_methods**: same as **data** but with community assignments by the community detection methods

**results**: contains fairness metric Phi and performance values

**figures**: contains figures displaying the results