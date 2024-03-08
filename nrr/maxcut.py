import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = \
    ['FreeSans'] + plt.rcParams['font.sans-serif']


def ip(g, s, t):
    kernel = pywraplp.Solver.CreateSolver('SCIP')
    variables = {}
    if not kernel:
        raise RuntimeError
    else:
        print('>> RUNNING IP w/ SCIP')

    for node in g.nodes():
        name = 'n {}'.format(node)
        variables[name] = kernel.BoolVar(name)

    for edge in g.edges():
        name = 'e {}|{}'.format(edge[0], edge[1])
        variables[name] = kernel.BoolVar(name)

    name = 'n {}'.format(s)
    kernel.Add(variables[name] == 1)
    name = 'n {}'.format(t)
    kernel.Add(variables[name] == 0)
    for edge in g.edges():
        name_x0 = 'n {}'.format(edge[0])
        name_x1 = 'n {}'.format(edge[1])
        name_e = 'e {}|{}'.format(edge[0], edge[1])
        kernel.Add(
            variables[name_e] <= variables[name_x0] + variables[name_x1])
        kernel.Add(
            variables[name_e] <= 2 - variables[name_x0] - variables[name_x1])

    cost = 0
    for edge in g.edges():
        name_e = 'e {}|{}'.format(edge[0], edge[1])
        w = g.edges[edge]['weight']
        cost = cost + w * variables[name_e]

    kernel.Maximize(cost)
    status = kernel.Solve()
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        cost = kernel.Objective().Value()
        print('Cost: {:.2f}'.format(cost))
        x = []
        for node in np.sort(g.nodes()):
            name = 'n {}'.format(node)
            variable = variables[name]
            x.append(variable.solution_value())
            print('Node {}: {}'.format(node, variable.solution_value()))
        return cost, x
    else:
        print("No solution found.")
        return None, None

def sdp(g, s, t):
    print('RUNNING SDP w/ MOSEK')
    n = g.number_of_nodes()
    node_list = list(g.nodes())
    W = np.zeros((n, n))
    for i, u in enumerate(node_list):
        for j, v in enumerate(node_list):
            if g.has_edge(u, v):
                W[i, j] = g[u][v]['weight']
    X = cp.Variable((n, n), symmetric=True)
    objective = cp.Minimize(cp.trace(W @ X))
    constraints = [X >> 0, cp.diag(X) == 1]
    s_index = node_list.index(s)
    t_index = node_list.index(t)
    constraints.append(X[s_index, t_index] == -1)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    eigenvalues, eigenvectors = np.linalg.eigh(X.value)
    vector = eigenvectors[:, np.argmax(eigenvalues)]
    assignments = vector > 0
    rounded_cost = 0
    for u, v, data in g.edges(data=True):
        if assignments[node_list.index(u)] != assignments[node_list.index(v)]:
            rounded_cost += data['weight']
    print('Relaxed Cost: {:.2f}'.format(-problem.value))
    print('Rounded Cost: {:.2f}'.format(rounded_cost))
    for node in np.sort(g.nodes()):
        print('Node {}: {}'.format(node, assignments[node]))
    return rounded_cost, assignments

def sdp_v2(g, s, t):
    print('>> RUNNING SDP V2 w/ MOSEK')
    n = g.number_of_nodes()
    Y = cp.Variable((n, n), symmetric=True)
    objective = cp.Maximize(0.5 * sum(
        g.edges[i, j]['weight'] * (1 - Y[i, j]) for (i, j) in g.edges()))
    constraints = [Y >> 0, cp.diag(Y) == 1]
    constraints.append(Y[s, t] == -1)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    u = np.random.randn(n)
    z = np.sign(Y.value @ u)
    relaxed_cost = problem.value
    rounded_cost = 0
    for (i, j) in g.edges():
        w = g.edges[i, j]['weight']
        rounded_cost = rounded_cost + w * (1 - z[i] * z[j])
    rounded_cost *= 1 / 2
    print('Relaxed Cost: {:.2f}'.format(relaxed_cost))
    print('Rounded Cost: {:.2f}'.format(rounded_cost))
    for node in np.sort(g.nodes()):
        print('Node {}: {}'.format(node, z[node]))
    return rounded_cost, z

def compare_solutions(g, s, t, x_ip, x_sdp):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    visualize_solution(ax, g, s, t, x_ip)
    ax.set_title('Integer Programming')
    ax.axis('equal')
    ax = fig.add_subplot(1, 2, 2)
    visualize_solution(ax, g, s, t, x_sdp)
    ax.set_title('Semidefinite Programming')
    ax.axis('equal')
    plt.show()

def visualize_solution(ax, g, s, t, x):
    pos = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, -1.0),
        3: (1.0, -1.0),
        4: (2.0, -0.5),
    }
    node_colors = []
    for node in g.nodes():
        if node == s:
            node_colors.append('#E23F44')
        elif x[node] == x[s]:
            node_colors.append('#FFADB0')
        elif node == t:
            node_colors.append('#0047AB')
        else:
            node_colors.append('#0096FF')
    edge_colors = []
    for edge in g.edges():
        u, v = edge
        if x[u] == x[v]:
            edge_colors.append('0')
        else:
            edge_colors.append('0.8')
    nx.draw(
        g, ax=ax, pos=pos,
        node_size=1.5e3,
        node_color=node_colors,
        edge_color=edge_colors,
        width=2,
        with_labels=True,
        font_color='w',
        font_size=18,
    )


def main():
    edges = [
        (0, 1),
        (0, 2),
        (1, 4),
        (1, 3),
        (2, 3),
        (3, 4),
    ]
    weights = {
        (0, 1): {'weight': 1.0},
        (0, 2): {'weight': 1.0},
        (1, 4): {'weight': 1.0},
        (1, 3): {'weight': 1.0},
        (2, 3): {'weight': 1.0},
        (3, 4): {'weight': 1.0},
    }
    g = nx.from_edgelist(edges)
    nx.set_edge_attributes(g, weights)
    s = 0
    t = 4
    _, x_ip = ip(g, s, t)
    _, x_sdp = sdp(g, s, t)
    _, x_sdp2 = sdp_v2(g, s, t)
    compare_solutions(g, s, t, x_ip, x_sdp)
    compare_solutions(g, s, t, x_ip, x_sdp2)


if __name__ == '__main__':
    main()

