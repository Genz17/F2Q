from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper


def _walk_string(
    i: int,
    mapping: Dict[int, Tuple[str, int]],
    nqubits: int,
    nstrings: int,
) -> str:
    string = ["I"] * nqubits
    while i in mapping:
        op, i = mapping[i]
        string[i - nstrings] = op
    return "".join(string)


class TernaryTreeMapper(FermionicMapper):
    """Fixed ternary-tree fermion-to-qubit mapper.

    This implements a static ternary-tree Majorana layout and exposes it
    through the standard Qiskit Nature `pauli_table()` interface.
    """

    @classmethod
    def majorana_table(cls, register_length: int) -> list[str]:
        nqubits = register_length
        nstrings = 2 * nqubits + 1

        # mapping[node] = (branch_label, parent_node)
        mapping: Dict[int, Tuple[str, int]] = {}

        # Start from the root and grow a ternary tree.
        free_slots: list[Tuple[int, str]] = [
            (nstrings, "Z"),
            (nstrings, "Y"),
            (nstrings, "X"),
        ]

        # Add internal ternary-tree nodes.
        # This reproduces the structure in your original code.
        for node in range(nstrings + 1, 3 * nqubits + 1):
            parent, branch = free_slots.pop(0)
            mapping[node] = (branch, parent)
            free_slots.extend([(node, "X"), (node, "Y"), (node, "Z")])

        # Attach the leaf nodes corresponding to Majorana operators.
        for leaf in range(nstrings):
            parent, branch = free_slots.pop(0)
            mapping[leaf] = (branch, parent)

        # Only the first 2*nqubits leaves are used as Majorana strings.
        return [
            _walk_string(i, mapping, nqubits, nstrings)
            for i in range(2 * nqubits)
        ]

    @classmethod
    @lru_cache(maxsize=32)
    def pauli_table(cls, register_length: int) -> list[tuple[Pauli, Pauli]]:
        majoranas = cls.majorana_table(register_length)
        return [
            (Pauli(majoranas[2 * i]), Pauli(majoranas[2 * i + 1]))
            for i in range(register_length)
        ]
