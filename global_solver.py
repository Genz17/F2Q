from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import *
from local_solver import localSolver
from HamiltonianGen import PauliSymplectic
from cmab import NonStationaryCombinatorialBandit
import numpy
import copy

class globalSolver():
    def __init__(self, H, localSize, SEARCH_DEPTH):
        self.H = H
        self.qNum = self.H.qNum
        self.localSize = localSize
        self.sd = SEARCH_DEPTH

    def __call__(self, iterNum = 200, strategy = 'random'):
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
        else:
            bandit = NonStationaryCombinatorialBandit(
                num_qubits=self.qNum,
                num_select=self.localSize,
                epsilon= 0.1,   # 25% 概率全局瞎逛防止卡死，75% 概率追着热点打
                alpha=0.9,      # 50% 的遗忘率，因为局部优化完一次，这块区域的潜力瞬间归零
                initial_q=30.0  # 假设初期一次优化能降 30，鼓励算法先把所有 qubit 摸一遍
            )

            for i in range(iterNum):
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
        #print(wList)
        return new_weight
            
if __name__ == "__main__":

    qNum = 20
    H = HoppingHam(qNum-1, qNum)
    #mapper = JordanWignerMapper()
    mapper = BravyiKitaevMapper()
    mapped_op = mapper.map(H)
    print(len(mapped_op.paulis))
    for j in range(10):
        numpy.random.seed(j)

        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))

        SEARCH_DEPTH = 3
        gs = globalSolver(ps, 6, SEARCH_DEPTH)
        w_b = gs(strategy = 'b')


        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]))

        SEARCH_DEPTH = 3
        gs = globalSolver(ps, 6, SEARCH_DEPTH)
        w_s = gs(strategy = 'random')
        print(w_b,w_s)
