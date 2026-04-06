from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper, ParityMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import *
from local_solver import localSolver
from HamiltonianGen import PauliSymplectic
from ttMapper import TernaryTreeMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
import numpy
import copy

class globalSolver():
    def __init__(self, H, localSize, SEARCH_DEPTH):
        self.H = H
        self.qNum = self.H.qNum
        self.localSize = localSize
        self.sd = SEARCH_DEPTH

    def __call__(self, iterNum = 50000, strategy = 'random'):
        wList = []
        failed_vectors = set()
        if strategy == 'random':
            for i in range(iterNum):
                random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False)
                old_weight = self.H.weight()
                print(i, old_weight)
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints], self.H.Zrep[:, random_ints],self.localSize) # take space
                ls = localSolver(ps, searchDepth=self.sd)
                ls(show=False)
                self.H.Xrep[:,random_ints] = ls.H.Xrep
                self.H.Zrep[:,random_ints] = ls.H.Zrep
                new_weight = self.H.weight()
        if strategy == 'rmax':
            for i in range(iterNum):
                active_matrix = self.H.Xrep | self.H.Zrep
                qubit_weights = active_matrix.sum(axis=0) # 得到长度为 qNum 的一维数组
                p = qubit_weights + 1e-3
                p = p / p.sum()
                for _ in range(2000):
                    random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False, p=p)
                    key = tuple(sorted(int(x) for x in random_ints))
                    if key not in failed_vectors:
                        break
                    else:
                        print("repeated sampling")
                else:
                    print("No new subset found after many retries.")
                    break
                old_weight = self.H.weight()
                print(random_ints, i, old_weight)
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints], self.H.Zrep[:, random_ints],self.localSize) # take space
                ls = localSolver(ps, searchDepth=self.sd)
                ls(show=False)
                self.H.Xrep[:,random_ints] = ls.H.Xrep
                self.H.Zrep[:,random_ints] = ls.H.Zrep
                new_weight = self.H.weight()
                if new_weight >= old_weight:
                    failed_vectors.add(key)
                else:
                    failed_vectors.clear()
            
if __name__ == "__main__":

    qNum = 20

    jw = []
    bk = []
    tt = []
    op = []
    molecules = (
    ("LiH", "H 0 0 0; Li 0 0 1.6"),
    ("H_2O", "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"),
    ("CH_4", "6 2.5369 0 0; 1 3.0739 0.31 0; 1 2 -0.31 0; 1 2.2269 0.5369 0; 1 2.8469 -0.5369 0; "),
    ("O_2", "O 0.616 0.0 0.0; O -0.616 0.0 0.0"),
    ("NaF", "11 3 0 0; 9 2 0 0;"),
    ("CO_2", "8 3.732 0.5 0; 8 2 -0.5 0; 6 2.866 0 0;"),
    )

    for j in range(6,8):
        print("==================")
        numpy.random.seed(42)
        #H = HoppingHam(j-1, j)
        H = HubbardHam2D(j,j)
        
        #astr = ''
        #astrList = ['H %.5f 0 0;' % (0.735 * i) for i in range(j)]
        #for a in astrList:
        #    astr+=a
        #print(astr)

        #H = PySCFDriver(
        #atom=astr,
        #basis="sto3g",
        #charge=0,
        #spin=j%2,
        #unit=DistanceUnit.ANGSTROM,
        #).run().hamiltonian.second_q_op()

        #print("now we are at ", m)
        #mol = m[1]
        #H = PySCFDriver(
        #atom = mol,
        #basis="sto3g",
        #charge=0,
        #spin=0,
        #unit=DistanceUnit.ANGSTROM,
        #).run().hamiltonian.second_q_op()


        mapper = JordanWignerMapper()
        mapped_op = mapper.map(H)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))
        jw.append(ps.weight()/len(mapped_op.paulis))

        mapper = BravyiKitaevMapper()
        mapped_op = mapper.map(H)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))
        bk.append(ps.weight()/len(mapped_op.paulis))


        mapper = TernaryTreeMapper()
        mapped_op = mapper.map(H)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))
        tt.append(ps.weight()/len(mapped_op.paulis))

        l = numpy.asarray([jw[-1], bk[-1], tt[-1]])
        print(l)
        if numpy.argmin(l) == 0:
            mapper = JordanWignerMapper()
        elif numpy.argmin(l) == 1:
            mapper = BravyiKitaevMapper()
        elif numpy.argmin(l) == 2:
            mapper = TernaryTreeMapper()
        else:
            print(numpy.argmin(l))
            mapper = None

        mapper = JordanWignerMapper()
        mapped_op = mapper.map(H)
        print(len(mapped_op.paulis))
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))

        SEARCH_DEPTH = 3
        gs = globalSolver(ps, 6, SEARCH_DEPTH)
        w_b = gs(strategy = 'rmax')
        op.append(w_b/len(mapped_op.paulis))
        print('jw=',jw)
        print('bk=',bk)
        print('tt=',tt)
        print('op=',op)
