import networkx as nx
import numpy as np


def __create_edges(current_node, new_nodes):
    node_list = np.array([current_node, *new_nodes])
    edges = [(node, neighbor) for node in node_list for neighbor in node_list[node_list != node]]
    return edges


def __add_cliques_to_network(graph, distribution):
    G = graph.copy()
    graph_len = len(G)
    new_node_list = np.arange(graph_len, sum(distribution))

    G.add_nodes_from(new_node_list)

    a = 0
    for i in range(len(distribution)):
        if distribution[i] == 1:
            continue

        house_nodes = new_node_list[a : a + distribution[i] - 1]
        edges = __create_edges(i, house_nodes)
        G.add_edges_from(edges, connection_type="intra")
        a += distribution[i] - 1

    return G


def watts_strogatz_clique_graph(distribution, k, p):
    try:
        n = len(distribution)
        G = nx.watts_strogatz_graph(n, k, p)
        nx.set_edge_attributes(G, values="inter", name="connection_type")
        G = __add_cliques_to_network(graph=G, distribution=distribution)
    except TypeError:
        n = distribution
        G = nx.watts_strogatz_graph(n, k, p)
        nx.set_edge_attributes(G, values="inter", name="connection_type")
    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def __draw_with_colors(graph):
        plt.figure(figsize=(9, 9))
        pos = nx.spring_layout(graph)

        nx.draw_networkx_nodes(graph, pos=pos)
        nx.draw_networkx_labels(graph, pos=pos)

        inter_edges = [
            edge
            for edge, attribute in nx.get_edge_attributes(graph, "connection_type").items()
            if attribute == "inter"
        ]
        intra_edges = [
            edge
            for edge, attribute in nx.get_edge_attributes(graph, "connection_type").items()
            if attribute == "intra"
        ]

        nx.draw_networkx_edges(graph, pos=pos, edgelist=inter_edges, edge_color="red")
        nx.draw_networkx_edges(graph, pos=pos, edgelist=intra_edges, edge_color="black")

        plt.axis("off")
        plt.savefig("graph.png")

    __distribution = 100

    G = watts_strogatz_clique_graph(__distribution, k=4, p=0.1)
    __draw_with_colors(G)
