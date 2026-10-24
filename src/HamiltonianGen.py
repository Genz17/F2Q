from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import PauliList
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    SquareLattice,
)
import numpy

class PauliSymplectic:

    def __init__(self, Xrep, Zrep, qNum=None, coeffs=None):
        self.Xrep = numpy.array(Xrep, dtype=numpy.uint8, copy=True)
        self.Zrep = numpy.array(Zrep, dtype=numpy.uint8, copy=True)
        self.qNum = self.Xrep.shape[1] if qNum is None else int(qNum)
        self.coeffs = (
            None if coeffs is None
            else numpy.abs(numpy.asarray(coeffs, dtype=complex)).reshape(-1)
        )

    @classmethod
    def from_sparse_pauli_op(cls, operator):
        # Keep the logical column order used by the original implementation.
        paulis = PauliList(operator.paulis)
        return cls(
            numpy.fliplr(paulis.x), numpy.fliplr(paulis.z),
            operator.num_qubits, operator.coeffs,
        )

    def copy(self):
        return type(self)(self.Xrep, self.Zrep, self.qNum, self.coeffs)

    def restrict(self, qubits):
        qubits = numpy.asarray(qubits, dtype=int)
        return type(self)(
            self.Xrep[:, qubits], self.Zrep[:, qubits], len(qubits), self.coeffs
        )

    def cnot(self, i, j):
        self.Xrep[:, j] ^= self.Xrep[:, i]
        self.Zrep[:, i] ^= self.Zrep[:, j]

    def h(self, i):
        temp = self.Xrep[:, i].copy()
        self.Xrep[:, i] = self.Zrep[:, i]
        self.Zrep[:, i] = temp

    def s(self, i):
        self.Zrep[:, i] ^= self.Xrep[:, i]

    def apply_actions(self, actions, qubit_map=None):
        for name, args in actions:
            if qubit_map is not None:
                args = tuple(int(qubit_map[index]) for index in args)
            getattr(self, name)(*args)

    def term_weights(self):
        return (self.Xrep | self.Zrep).sum(axis=1, dtype=numpy.int64)

    def weight(self):
        return int(self.term_weights().sum(dtype=numpy.int64))

    def qdriftWeight(self):
        if self.coeffs is None:
            raise ValueError
        return float(numpy.dot(self.coeffs, self.term_weights()))

    @property
    def term_count(self):
        return int(self.Xrep.shape[0])


def HoppingHam(r,N):
    Ham = FermionicOp({}, num_spin_orbitals=N)
    for i in range(N):
        for j in range(N):
            if abs(i-j) <= r and i != j:
                Ham = Ham + FermionicOp({"+_%d -_%d" % (i, j): 1}, num_spin_orbitals=N)
    return Ham

def H2D(Lx, Ly):
    N = Lx * Ly
    Ham = FermionicOp({}, num_spin_orbitals=N)

    for i in range(N):
        for j in range(N):
            xi, yi = i % Lx, i // Lx
            xj, yj = j % Lx, j // Lx

            dx = abs(xi - xj)
            dy = abs(yi - yj)
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                Ham = Ham + FermionicOp({"+_%d -_%d" % (i, j): 1}, num_spin_orbitals=N)

    return Ham

def HubbardHam2D(Lx, Ly, periodic=False):
    if periodic:
        square_lattice = SquareLattice(
            rows=Lx, cols=Ly, boundary_condition=BoundaryCondition.PERIODIC
        )
    else:
        square_lattice = SquareLattice(
            rows=Lx, cols=Ly, boundary_condition=BoundaryCondition.OPEN
        )
    H = FermiHubbardModel(
        square_lattice.uniform_parameters(
            uniform_interaction=-1,
            uniform_onsite_potential=0,
        ),
        onsite_interaction=4,
    ).second_q_op().simplify()

    return H
