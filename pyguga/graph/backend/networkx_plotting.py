# type: ignore
"""Plotting functions for GUGA graphs using NetworkX."""

import networkx as nx
import matplotlib.pyplot as plt


def custom_subgroup_layout_horizontal(G: nx.DiGraph, padding: int = 50):
    """Automatically calculate node spacing to avoid overlapping.

    Positions nodes in a 2D grid by (u, a) with dynamic spacing.

    Args:
        G: NetworkX graph with node attributes 'layer' and 'subgroup'
        padding: Additional space between nodes

    Returns:
    - pos: Dictionary mapping nodes to positions

    """
    pos = {}

    # Gather nodes by (layer, subgroup)
    groups = {}
    for node, attrs in G.nodes(data=True):
        group = (attrs["layer"], attrs["subgroup"])
        groups.setdefault(group, []).append(node)

    # Sort nodes within each group by their original index (j)
    for group in groups:
        groups[group].sort()

    # Determine unique layers and subgroups
    unique_layers = sorted({attrs["layer"] for _, attrs in G.nodes(data=True)})
    unique_subgroups = sorted({attrs["subgroup"] for _, attrs in G.nodes(data=True)})

    # Calculate spacing based on the number of layers and subgroups
    # You can adjust the base_spacing as needed
    base_x_spacing = 1000.0
    base_y_spacing = 1000.0
    base_vertical_offset = 50

    # Optionally, adjust spacing based on the maximum number of nodes in any group
    max_nodes_in_group = max(len(nodes) for nodes in groups.values())
    vertical_offset = base_vertical_offset * (max_nodes_in_group / 2) + padding

    x_spacing = base_x_spacing + padding
    y_spacing = base_y_spacing + padding

    # Sort groups to layout from left to right and bottom to top
    sorted_keys = sorted(groups.keys(), key=lambda x: (x[0], x[1]))

    for u_val, a_val in sorted_keys:
        # Base coordinates for this group
        x_coord = unique_layers.index(u_val) * x_spacing
        y_coord = unique_subgroups.index(a_val) * y_spacing

        # Offset vertically if multiple nodes share the same (u, a)
        for i, node in enumerate(groups[(u_val, a_val)]):
            pos[node] = (x_coord, y_coord + i * vertical_offset)

    return pos


def nx_draw(G: nx.DiGraph, abc=False):
    """Draw a GUGA graph with subgroup layout.

    Args:
        G: NetworkX DiGraph representing the GUGA graph
        abc: If True, label nodes with their abc attributes

    """
    layout = custom_subgroup_layout_horizontal(G)
    plt.figure(figsize=(12, 8))

    # Extract edge colors
    edges = G.edges(data=True)
    edge_colors = [edge[2]["color"] for edge in edges]

    # Create labels with abc attributes
    if abc:
        labels = {node: f"{attrs['abc']}" for node, attrs in G.nodes(data=True)}
    else:
        labels = {node: f"{node}" for node in G.nodes()}

    nx.draw(
        G,
        pos=layout,
        labels=labels,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        arrows=True,
        arrowstyle="-",
        arrowsize=20,
    )
    plt.title("ALL GUGA Graph")
    plt.show()
