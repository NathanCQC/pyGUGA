# type: ignore
"""Test the GUGA graph Dimension."""

import pytest
from pyguga.graph import nx_build_graph, n_paths_to_node
from pyguga.graph import DistinctRowTable


@pytest.mark.parametrize("n_spatial_orbs", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_full_fock_space_dim(n_spatial_orbs: int):
    """Test the full Fock space dimension."""
    drt = DistinctRowTable(n_spatial_orbs)
    graph = nx_build_graph(drt.df)
    df_u = drt.df_orbital_level(n_spatial_orbs)
    spin_multiplicity = df_u["b"] + 1
    # This could be parallelized
    node_path_list = {j: n_paths_to_node(graph, j) for j in df_u.index}
    spin_multiplicity_node_path_list = {
        j: spin_multiplicity[j] * node_path_list[j] for j in df_u.index
    }
    fermionic_fock_space_dim = sum(spin_multiplicity_node_path_list.values())
    assert fermionic_fock_space_dim == 2 ** (2 * n_spatial_orbs)
