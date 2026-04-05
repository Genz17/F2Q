from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import *
from local_solver import localSolver
from HamiltonianGen import PauliSymplectic
from cmab import NonStationaryCombinatorialBandit
from ucb import ExactUCBBandit
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
                random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False, p=p)
                old_weight = self.H.weight()
                print(i, old_weight)
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints], self.H.Zrep[:, random_ints],self.localSize) # take space
                ls = localSolver(ps, searchDepth=self.sd)
                ls(show=False)
                self.H.Xrep[:,random_ints] = ls.H.Xrep
                self.H.Zrep[:,random_ints] = ls.H.Zrep
                new_weight = self.H.weight()
        if strategy == 'cmab':
            bandit = NonStationaryCombinatorialBandit(
                num_qubits=self.qNum,
                num_select=self.localSize,
                epsilon=0.25,   # 25% 概率全局瞎逛防止卡死，75% 概率追着热点打
                alpha=0.9,      # 50% 的遗忘率，因为局部优化完一次，这块区域的潜力瞬间归零
                initial_q=100.0  # 假设初期一次优化能降 30，鼓励算法先把所有 qubit 摸一遍
            )

            for i in range(iterNum):
                #bandit.epsilon = min(i * 0.001, 0.5)
                random_ints = bandit.select_block()
                old_weight = self.H.weight()
                print(i, old_weight)
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints], self.H.Zrep[:, random_ints], self.localSize)
                ls = localSolver(ps, searchDepth=self.sd)
                ls(show=False)
                self.H.Xrep[:,random_ints] = ls.H.Xrep
                self.H.Zrep[:,random_ints] = ls.H.Zrep

                new_weight = self.H.weight()

                reward = old_weight - new_weight
                bandit.update(random_ints, reward)

        if strategy == 'ucb':
            bandit = ExactUCBBandit(self.qNum, self.localSize, alpha=0.8, c=10.0)
            for i in range(iterNum):
                arm_idx, random_ints = bandit.select_block()

                old_weight = self.H.weight()
                print(i, old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints], self.H.Zrep[:, random_ints], self.localSize)
                ls = localSolver(ps, searchDepth=self.sd)
                ls(show=False)

                self.H.Xrep[:,random_ints] = ls.H.Xrep
                self.H.Zrep[:,random_ints] = ls.H.Zrep

                new_weight = self.H.weight()
                reward = old_weight - new_weight

                bandit.update(arm_idx, reward)
        return new_weight
            
if __name__ == "__main__":

    qNum = 20

    jw = []
    bk = []
    tt = []
    op = []

    for j in range(6,7):
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



        #x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        #z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        #ps = PauliSymplectic(x_mat,
        #                     z_mat,
        #                     len(mapped_op.paulis[0]))

        #SEARCH_DEPTH = 4
        #gs = globalSolver(ps, 3, SEARCH_DEPTH)
        #w_s = gs(strategy = 'random')
        #print(w_b,w_s)
