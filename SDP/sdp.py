import networkx as nx
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

G = nx.Graph()
# Add nodes and weighted edges as needed. Here's an example:
start_node = 0
end_node = 4
for i in range(start_node, end_node):
    for j in range(i+1, end_node+1):
        G.add_edge(i, j, weight=np.random.randint(1, 10))

# G.add_edge('s', 'a', weight=2)
# G.add_edge('a', 'b', weight=3)
# G.add_edge('b', 't', weight=4)
# G.add_edge('s', 't', weight=1)

n = G.number_of_nodes()
node_list = list(G.nodes())
W = np.zeros((n, n))
for i, u in enumerate(node_list):
    for j, v in enumerate(node_list):
        if G.has_edge(u, v):
            W[i, j] = G[u][v]['weight']

# Define the SDP variable
X = cp.Variable((n, n), symmetric=True)

# The objective is to maximize the sum of the weights of the edges between the subsets
objective = cp.Minimize(cp.trace(W @ X))

# Constraints
constraints = [X >> 0, cp.diag(X) == 1]
# Adding a constraint to ensure s and t are in different subsets
s_index = node_list.index(start_node)
t_index = node_list.index(end_node)
constraints.append(X[s_index, t_index] == -1)  # This enforces s and t to be in different sets

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=False)

print(f"Maximum cut value (relaxed): {-problem.value}")


# use X.value to infer the partitioning
eigenvalues, eigenvectors = np.linalg.eigh(X.value)

vector = eigenvectors[:, np.argmax(eigenvalues)]
assignments = vector > 0

# compute the ground truth cut value
cut_value = 0
for u, v, data in G.edges(data=True):
    if assignments[node_list.index(u)] != assignments[node_list.index(v)]:
        # This edge is part of the cut
        cut_value += data['weight']

print(f"Maximum cut value (rounded): {cut_value}")

# Draw the graph
# Assign colors based on the partition
colors = ['blue' if assign else 'red' for assign in assignments]
pos = nx.spring_layout(G)  # Generates a layout for the graph
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800, edge_color='black')

# Highlight the edge between 's' and 't' if it exists
if G.has_edge('s', 't'):
    nx.draw_networkx_edges(G, pos, edgelist=[('s', 't')], edge_color='green', width=2)

# Label edge weights
edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


plt.title('Visualizing the Cut')
plt.show()