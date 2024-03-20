import networkx as nx
import random
import json
import numpy as np
import argparse
from tqdm import tqdm
import os
from graph import generate_random_graph, max_cut_approximation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Max-Cut Approximation')
    parser.add_argument('--node_bound', type=int, default=50, help='Number of nodes')
    parser.add_argument('--probability', type=float, default=0.5, help='Probability of edge creation')
    parser.add_argument('--method', type=str, default='sdp', choices=['sdp', 'ip'], help='Method to use for the approximation')
    parser.add_argument('--data_folder', type=str, default='data', help='Folder to save the graph and solution')
    parser.add_argument('--num_examples', type=int, default=1000, help='Number of samples to generate')
    args = parser.parse_args()

    data_folder = args.data_folder

    for i in tqdm(range(args.num_examples)):
        num_nodes = random.randint(10, args.node_bound)
        data_path = os.path.join(data_folder, 'node_num_' + str(num_nodes))
        os.makedirs(data_path, exist_ok=True)
        file_name = os.path.join(data_path, 'graph_' + str(i) + '.json')

        prob = args.probability
        G = generate_random_graph(num_nodes, prob, weight_range=(1, 10))
        s = np.random.choice(G.nodes())
        t = np.random.choice(G.nodes())
        while t == s:
            t = np.random.choice(G.nodes())
        s, t = int(s), int(t)
        prob_data = {
            'graph': G,
            's': s,
            't': t
        }
        method = args.method
        rounded_cost, assignment, sol = max_cut_approximation(prob_data, method=method)
        if sol is not None:
            sol = sol.tolist()
        rounded_cost = float(rounded_cost)
        cut_set = [node for node in G.nodes() if assignment[node] == assignment[s]]

        # Save the graph and its solution
        data_to_save = {'graph': nx.node_link_data(G), 's': s, 't': t, 'cut_set': cut_set, 'method': method,
                        'rounded_cost': rounded_cost, 'sol_value': sol}

        with open(file_name, 'w') as f:
            json.dump(data_to_save, f, indent=4)
