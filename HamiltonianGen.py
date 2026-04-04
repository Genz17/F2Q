from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import PauliList
import numpy

class PauliSymplectic():
    def __init__(self, Xrep, Zrep, qNum):
        self.Xrep = Xrep
        self.Zrep = Zrep
        self.qNum = qNum

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


def HoppingHam(r,N):
    Ham = FermionicOp({}, num_spin_orbitals=N)
    for i in range(N):
        for j in range(N):
            if abs(i-j) < r and i != j:
                Ham = Ham + FermionicOp({"+_%d -_%d" % (i, j): 1}, num_spin_orbitals=N)
    return Ham

if __name__ == "__main__":
    H = HoppingHam(3, 4)
    mapper = JordanWignerMapper()
    mapped_op = mapper.map(H)
    ps = PauliSymplectic(numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32),
                         numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32),
                         len(mapped_op.paulis[0]))
    print(ps.weight())
    ps.cnot(1,3)
    print(ps.weight())
