from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper
from qiskit.quantum_info import PauliList
from ttMapper import TernaryTreeMapper
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    SquareLattice,
)
import numpy

class PauliSymplectic():
    def __init__(self, Xrep, Zrep, qNum, coeffs=None):
        self.Xrep = Xrep
        self.Zrep = Zrep
        self.qNum = qNum
        if type(coeffs) != type(None):
            self.coeffs = numpy.abs(coeffs)

    def cnot(self,i,j):
        # control on i, act on j
        self.Xrep[:,j] ^= self.Xrep[:,i] 
        self.Zrep[:,i] ^= self.Zrep[:,j] 

    def h(self,i):
        temp = self.Xrep[:,i].copy()
        self.Xrep[:,i] = self.Zrep[:,i]
        self.Zrep[:,i] = temp

    def s(self,i):
        self.Zrep[:, i] ^= self.Xrep[:, i]

    def weight(self):
        self.sumRep = self.Xrep | self.Zrep
        return self.sumRep.sum()

    def qdriftWeight(self):
        self.sumRep = self.Xrep | self.Zrep
        temp = self.sumRep.sum(axis=1)
        return (self.coeffs * temp).sum()


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

if __name__ == "__main__":
    H = HoppingHam(1, 4)
    mapper = BravyiKitaevMapper()
    #square_lattice = SquareLattice(
    #    rows=2, cols=2, boundary_condition=BoundaryCondition.OPEN
    #)
    #H = FermiHubbardModel(
    #    square_lattice.uniform_parameters(
    #        uniform_interaction=-1,
    #        uniform_onsite_potential=0,
    #    ),
    #    onsite_interaction=4,
    #).second_q_op().simplify()
    mapped_op = mapper.map(H)
    print(mapped_op.paulis)
    print(mapped_op.coeffs)


    ps1 = PauliSymplectic(numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32),
                         numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32),
                         len(mapped_op.paulis[0]), mapped_op.coeffs)
    print(ps1.qdriftWeight())

    #H = HubbardHam2D(2,2,periodic = False).simplify()
    #mapped_op = mapper.map(H)
    #ps2 = PauliSymplectic(numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32),
    #                     numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32),
    #                     len(mapped_op.paulis[0]))
    #mapper = BravyiKitaevMapper()
    #mapper = TernaryTreeMapper()
    #mapped_op = mapper.map(H)
    #print(mapped_op)

    #mapper = JordanWignerMapper()
    #mapped_op = mapper.map(H)
    #print(mapped_op)
    #ps = PauliSymplectic(numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32),
    #                     numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32),
    #                     len(mapped_op.paulis[0]))
    #print(ps.weight())
    #ps.cnot(1,3)
    #print(ps.weight())
