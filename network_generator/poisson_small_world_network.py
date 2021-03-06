import networkx as nx
import numpy as np
from math import ceil
from random import choices, choice
from typing import Tuple, List
from itertools import chain


def __get_nearest_neighbors_connections(node: int,
                                        num_connections: int,
                                        number_of_nodes: int) -> List[Tuple[int, int]]:
    connections_to_right = ceil(num_connections / 2)
    connections_to_left = num_connections - connections_to_right

    right_connections = np.array(
        [i for i in range(node + 1, node + connections_to_right + 1)])
    right_connections[right_connections >
                      number_of_nodes - 1] -= number_of_nodes

    left_connections = np.array(
        [i for i in range(node - connections_to_left, node)])
    left_connections[left_connections < 0] += number_of_nodes

    neighbor_to_connect = np.concatenate([right_connections, left_connections])

    return [sorted([node, neighbor]) for neighbor in neighbor_to_connect]


def __is_rewired(probability: float) -> bool:
    return choices([True, False], weights=[probability, 1 - probability])[0]


def __rewire_connection(node_connection: Tuple[int, int], probability: float, number_of_nodes: int):
    if __is_rewired(probability):
        nodes = list(range(number_of_nodes))
        node, _ = node_connection
        # A node cannot rewire with itself
        nodes.remove(node)
        rewired_neighbor = choice(nodes)
        return (node, rewired_neighbor)

    return node_connection


def __create_edges(number_of_nodes: int, lam: int, rewiring_probability: float):
    if rewiring_probability < 0:
        raise ValueError("Rewiring probability must be greater than 0")

    connectivity_distribution = np.random.poisson(
        lam=lam, size=number_of_nodes)

    connectivity = [__get_nearest_neighbors_connections(node, connections, number_of_nodes)
                    for node, connections in enumerate(connectivity_distribution)]

    connectivity_without_repeated_edges = {tuple(item)
                                           for item in chain.from_iterable(connectivity)}
    if rewiring_probability > 0:
        rewired_connectivity = [
            __rewire_connection(
                connection, rewiring_probability, number_of_nodes)
            for connection in connectivity_without_repeated_edges]
        return rewired_connectivity

    return list(connectivity_without_repeated_edges)


def poisson_small_world_graph(n: int, D: int, p: float, Graph_to_update: nx.Graph = None) -> nx.Graph:
    """Returns a Poisson small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes
    D : int
        The average degree of the network. It is the same lambda of poisson
        distribution

    p : float
        The probability of rewiring each edge

    Graph_to_update: nx.Graph
        A graph to update the degree distribution following Poisson distribution.
    """
    if Graph_to_update:
        G = Graph_to_update.copy()
        G.remove_edges_from(G.edges())
    else:
        G = nx.Graph()
        G.add_nodes_from(list(range(n)))
    edges = __create_edges(n, D, p)
    G.add_edges_from(edges)

    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def __draw_circular(graph):
        pos = nx.circular_layout(graph)
        nx.draw_networkx(graph, pos=pos)
        plt.show()

    G = nx.watts_strogatz_graph(50, 2, 0.1)
    H = poisson_small_world_graph(50, 8, .5, G)
    __draw_circular(H)
