import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import numpy as np
from maxcut import ip, sdp_v2
import argparse
from tqdm import tqdm
import os

def generate_random_graph(nodes, probability, weight_range=(1, 10)):
    # Generate the Erdős-Rényi graph
    G = nx.erdos_renyi_graph(nodes, probability)

    # Assign random weights to each edge
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(*weight_range)

    return G

def max_cut_approximation(prob_data, method='ip', verbose=False):
    G, s, t = prob_data['graph'], prob_data['s'], prob_data['t']
    if method == 'ip':
        rounded_cost, assignments, sol = ip(G, s, t, verbose=verbose)
    elif method == 'sdp':
        rounded_cost, assignments, sol = sdp_v2(G, s, t, verbose=verbose)
    else:
        raise ValueError(f"Invalid method: {method}")
    return rounded_cost, assignments, sol

def visualize_graph(prob_data, cut_set=None):
    """Visualize the graph with an optional cut set highlighted."""
    G, s, t =prob_data['graph'], prob_data['s'], prob_data['t']
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    if cut_set:
        nx.draw_networkx_nodes(G, pos, nodelist=cut_set, node_color='lightgreen')
    # label the source node and target node
    nx.draw_networkx_nodes(G, pos, nodelist=[s], node_color='green', label= 's' + str(s))
    nx.draw_networkx_nodes(G, pos, nodelist=[t], node_color='blue', label = 't' + str(t))

    # label the edges with weights
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    plt.show()

def save_graph_solution(graph, solution, file_name):
    """Saves the graph and solution to a file."""
    data = {
        'graph': nx.node_link_data(graph),
        'solution': list(solution)
    }
    with open(file_name, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    data_folder = 'data/test_data'

    for i in tqdm(range(1)):
        num_nodes = random.randint(10, 20)
        data_path = os.path.join(data_folder, 'node_num_' + str(num_nodes))
        os.makedirs(data_path, exist_ok=True)
        file_name = os.path.join(data_path, 'graph_' + str(i) + '.json')

        prob = 0.5
        G = generate_random_graph(num_nodes, prob, weight_range=(1, 10))
        s = int(np.random.choice(G.nodes()))
        t = int(np.random.choice(G.nodes()))
        while t == s:
            t = np.random.choice(G.nodes())

        prob_data = {
            'graph': G,
            's': s,
            't': t
        }
        method = 'sdp'
        rounded_cost, assignment, sol = max_cut_approximation(prob_data, method=method)
        if sol is not None:
            sol = sol.tolist()
        rounded_cost = float(rounded_cost)
        cut_set = [node for node in G.nodes() if assignment[node] == assignment[s]]

        visualize_graph(prob_data, cut_set=cut_set)

        # Save the graph and its solution
        data_to_save = {'graph': nx.node_link_data(G), 's': s, 't': t, 'cut_set': cut_set, 'method': method,
                        'rounded_cost': rounded_cost, 'sol_value': sol}

        with open(file_name, 'w') as f:
            json.dump(data_to_save, f, indent=4)
