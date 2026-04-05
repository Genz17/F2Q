from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper
from qiskit.quantum_info import PauliList
from ttMapper import TernaryTreeMapper
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

def HubbardHam2D(Lx, Ly, t=1.0, U=4.0, periodic=False):
    """
    2D spinful Hubbard model on an Lx x Ly square lattice.

    Spin-orbital indexing:
        site s = x + Lx*y
        up   -> 2*s
        down -> 2*s + 1
    """
    n_sites = Lx * Ly
    N = 2 * n_sites  # two spin orbitals per site

    Ham = FermionicOp({}, num_spin_orbitals=N)

    def site(x, y):
        return x + Lx * y

    def orb(x, y, spin):
        # spin = 0 for up, 1 for down
        return 2 * site(x, y) + spin

    for x in range(Lx):
        for y in range(Ly):
            # on-site interaction: U n_up n_down
            up = orb(x, y, 0)
            dn = orb(x, y, 1)
            Ham = Ham + FermionicOp(
                {f"+_{up} -_{up} +_{dn} -_{dn}": U},
                num_spin_orbitals=N
            )

            # nearest neighbors: only add right and up to avoid double counting
            for dx, dy in [(1, 0), (0, 1)]:
                xn, yn = x + dx, y + dy

                if periodic:
                    xn %= Lx
                    yn %= Ly
                else:
                    if xn >= Lx or yn >= Ly:
                        continue

                for spin in [0, 1]:
                    i = orb(x, y, spin)
                    j = orb(xn, yn, spin)

                    # -t (a_i^\dagger a_j + a_j^\dagger a_i)
                    Ham = Ham + FermionicOp(
                        {
                            f"+_{i} -_{j}": -t,
                            f"+_{j} -_{i}": -t,
                        },
                        num_spin_orbitals=N
                    )

    return Ham

if __name__ == "__main__":
    H = HoppingHam(3, 4)
    #mapper = BravyiKitaevMapper()
    mapper = TernaryTreeMapper()
    mapped_op = mapper.map(H)
    print(mapped_op)

    mapper = JordanWignerMapper()
    mapped_op = mapper.map(H)
    print(mapped_op)
    #ps = PauliSymplectic(numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32),
    #                     numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32),
    #                     len(mapped_op.paulis[0]))
    #print(ps.weight())
    #ps.cnot(1,3)
    #print(ps.weight())
