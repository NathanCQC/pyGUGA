# type: ignore
"""Module for GUGA graph construction using NetworkX as the backend."""

from __future__ import annotations
import networkx as nx

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


def nx_build_graph(df: DataFrame, arcs: tuple[int, ...] = (0, 1, 2, 3)):
    """Build a NetworkX graph from the Distinct Row Table for the GUGA graph.

    Args:
        df: A pandas DataFrame representing the Distinct Row Table for the GUGA graph.
        arcs: A tuple of integers representing the arcs to include in the graph.

    Returns:
        A NetworkX DiGraph representing the GUGA graph.

    """
    if not set(arcs).issubset({0, 1, 2, 3}):
        raise ValueError("arcs must be a subset of {0, 1, 2, 3}")

    G = nx.DiGraph()  # directed graph

    color_map = {"0": "red", "1": "orange", "2": "green", "3": "blue"}  # not working

    def arcs_to_str(prefix: str, arcs: tuple[int, ...]):
        return [f"{prefix}{arc}" for arc in arcs]

    for j in df.index:
        u_val: int = df["u"][j]
        a_val: int = df["a"][j]
        G.add_node(
            j,
            layer=u_val,
            subgroup=a_val,
            abc=(df["a"][j], df["b"][j], df["c"][j]),
        )

        for x in arcs_to_str("l", arcs):
            l_val = df[x][j]
            if l_val != "None":
                color = color_map[x[-1]]
                G.add_edge(l_val, j, color=color)

        for x in arcs_to_str("k", arcs):
            k_val = df[x][j]
            if k_val != "None":
                color = color_map[x[-1]]
                G.add_edge(j, k_val, color=color)
    return G


def n_paths_to_node(G: nx.Graph, node: int, source: int = 0) -> int:
    """Calculate the number of paths to a node in a NetworkX graph.

    Args:
        G: NetworkX graph
        node: Target node
        source: Source node (default=0)

    Returns:
        The number of paths from the source node to the target node.

    """
    paths = nx.all_simple_paths(G, source=node, target=source)
    return len(list(paths))
