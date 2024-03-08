import networkx as nx
import matplotlib.pyplot as plt

edges = [
    [0, 1],
    [0, 2],
    [1, 4],
    [1, 3],
    [2, 3],
    [3, 4],
]

g = nx.from_edgelist(edges)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pos = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, -1.0),
    3: (1.0, -1.0),
    4: (2.0, -0.5),
}
nx.draw(
    g, ax=ax, pos=pos, 
    node_size=2e3, 
    with_labels=True, 
    font_color='w',
    font_family='FreeSans',
    font_size=18,
)
ax.axis('equal')
fig.tight_layout()
plt.savefig('sample_network.png')
#plt.show()
