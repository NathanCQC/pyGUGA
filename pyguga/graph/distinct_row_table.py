"""Distinct Row Table for Shavitt GUGA graph."""

import pandas as pd


class DistinctRowTable:
    """Class to generate the Distinct Row Table for the Shavitt GUGA graph."""

    def __init__(self, n_spatial: int) -> None:
        """Initialize the Distinct Row Table for the Shavitt GUGA graph.

        This will generate the Distinct Row Table for the Shavitt GUGA graph
        for all possible head nodes of the graph for a given number of spatial
        orbitals.

        Args:
            n_spatial: Number of spatial orbitals.

        """
        self._nspatorbs = n_spatial
        self._abc_node_dict = self._get_node_dict(n_spatial)
        self._ki_dict = self._get_downward_chaining_indices()
        self._li_dict = self._get_upward_chaining_indices()
        self._nnode = list(self._abc_node_dict.values())[-1]
        self._distinct_row_table = self._get_distinct_row_table()
        self._csf_list: list[None] = []
        self._n_csfs = None

    def _get_node_dict(self, n_tot: int) -> dict[tuple[int, int, int, int], int]:
        """Return a dictionary of all possible nodes in the graph.

        Args:
            n_tot: Number of spatial orbitals.

        Returns:
            A dictionary with keys as 4-tuples (a, b, c, n) and values as integers
            representing the nodes of the graph.

        """

        def all_tuples_sum_n(n: int):
            """Return all 3-element tuples (a, b, c) such that a+b+c = n."""
            results: list[tuple[int, int, int]] = []
            for a in range(n + 1):
                for b in range(n + 1):
                    for c in range(n + 1):
                        if a + b + c == n:
                            results.append((a, b, c))
            return list(reversed(results))

        abc_node_dict: dict[tuple[int, int, int, int], int] = {}
        j = 0
        for n in range(n_tot + 1):
            for a, b, c in all_tuples_sum_n(n):
                abc_node_dict[(a, b, c, n)] = j
                j += 1

        return abc_node_dict

    def _get_downward_chaining_indices(self) -> dict[int, list[int | str]]:
        """Return the downward chaining indices for all nodes in the graph.

        Calculates the lower nodes which the current node is connected to in the
        Shavitt GUGA graph.

        Returns:
            A dictionary with keys as integers representing the nodes of the graph
            and values as lists of integers or strings representing the downward
            chaining indices.

        """
        ki_dict: dict[int, list[int | str]] = {}
        for j, j_abc in zip(
            self._abc_node_dict.values(), self._abc_node_dict.keys(), strict=True
        ):
            k0_abc = (j_abc[0], j_abc[1], j_abc[2] - 1, j_abc[3] - 1)
            k1_abc = (j_abc[0], j_abc[1] - 1, j_abc[2], j_abc[3] - 1)
            k2_abc = (j_abc[0] - 1, j_abc[1] + 1, j_abc[2] - 1, j_abc[3] - 1)
            k3_abc = (j_abc[0] - 1, j_abc[1], j_abc[2], j_abc[3] - 1)

            k0 = self._abc_node_dict.get(k0_abc, "None")
            k1 = self._abc_node_dict.get(k1_abc, "None")
            k2 = self._abc_node_dict.get(k2_abc, "None")
            k3 = self._abc_node_dict.get(k3_abc, "None")

            ki_dict[j] = [k0, k1, k2, k3]

        return ki_dict

    def _get_upward_chaining_indices(self) -> dict[int, list[int | str]]:
        """Return the upward chaining indices for all nodes in the graph.

        Calculates the upper nodes which the current node is connected to in the
        Shavitt GUGA graph.

        Returns:
            A dictionary with keys as integers representing the nodes of the graph
            and values as lists of integers or strings representing the upward
            chaining indices.

        """
        li_dict: dict[int, list[int | str]] = {}
        for j, j_abc in zip(
            self._abc_node_dict.values(), self._abc_node_dict.keys(), strict=True
        ):
            l0_abc = (j_abc[0], j_abc[1], j_abc[2] + 1, j_abc[3] + 1)
            l1_abc = (j_abc[0], j_abc[1] + 1, j_abc[2], j_abc[3] + 1)
            l2_abc = (j_abc[0] + 1, j_abc[1] - 1, j_abc[2] + 1, j_abc[3] + 1)
            l3_abc = (j_abc[0] + 1, j_abc[1], j_abc[2], j_abc[3] + 1)

            l0 = self._abc_node_dict.get(l0_abc, "None")
            l1 = self._abc_node_dict.get(l1_abc, "None")
            l2 = self._abc_node_dict.get(l2_abc, "None")
            l3 = self._abc_node_dict.get(l3_abc, "None")

            li_dict[j] = [l0, l1, l2, l3]

        return li_dict

    def _get_distinct_row_table(self) -> pd.DataFrame:
        """Return the Distinct Row Table for the Shavitt GUGA graph.

        Returns:
            A pandas DataFrame representing the Distinct Row Table for the GUGA graph.

        """
        uirrep: list[int] = []
        for u in reversed(range(self._nspatorbs + 1)):
            a = 1
            for abcN in self._abc_node_dict.keys():
                if abcN[3] == u:
                    uirrep.append(a)
                    a = a + 1
        j_values = list(self._abc_node_dict.values())
        data: dict[str, list[int | str] | list[int]] = {
            "a": [int(i[0]) for i in self._abc_node_dict.keys()],
            "b": [int(i[1]) for i in self._abc_node_dict.keys()],
            "c": [int(i[2]) for i in self._abc_node_dict.keys()],
            "u": [i[3] for i in self._abc_node_dict.keys()],
            "uirrep": uirrep,
            "k0": [i[0] for i in self._ki_dict.values()],  # downward chaining indices
            "k1": [i[1] for i in self._ki_dict.values()],
            "k2": [i[2] for i in self._ki_dict.values()],
            "k3": [i[3] for i in self._ki_dict.values()],
            "l0": [i[0] for i in self._li_dict.values()],  # upward chaining indices
            "l1": [i[1] for i in self._li_dict.values()],
            "l2": [i[2] for i in self._li_dict.values()],
            "l3": [i[3] for i in self._li_dict.values()],
        }

        return pd.DataFrame(data, index=j_values)

    @property
    def df(self) -> pd.DataFrame:
        """Return the Distinct Row Table as a pandas DataFrame.

        Where the columns are:
        a, b, c, u, uirrep, k0, k1, k2, k3, l0, l1, l2, l3.
        """
        return self._distinct_row_table

    def df_orbital_level(self, u_level: int) -> pd.DataFrame:
        """Return the Distinct Row Table for a given orbital level."""
        return self.df[self.df["u"] == u_level]
