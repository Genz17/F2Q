from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper, ParityMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import *
from local_solver import localSolver, worker_task
from HamiltonianGen import PauliSymplectic
from ttMapper import TernaryTreeMapper
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from joblib import Parallel, delayed
from itertools import combinations
import numpy
import copy
import pickle
import xgboost as xgb

def extract_features(H, selected_qubits):
    active_matrix = (H.Xrep | H.Zrep).astype(numpy.int32)
    activities = active_matrix[:, selected_qubits].sum(axis=0)
    activities = numpy.sort(activities)[::-1]

    row_sums = active_matrix[:, selected_qubits].sum(axis=1)
    strong_couplings = numpy.sum(row_sums >= 2)
    current_weight = active_matrix.sum()

    return numpy.array([
        activities[0], activities[1], activities[2], activities[3],
        strong_couplings, current_weight
    ])

class globalSolver():
    def __init__(self, H, localSize, SEARCH_DEPTH):
        self.H = H
        self.qNum = self.H.qNum
        self.localSize = localSize
        self.sd = SEARCH_DEPTH

    def __call__(self, iterNum = 15000, strategy = 'random', nTerm = 1):
        wList = []
        failed_vectors = set()
        if strategy == 'random':
            for i in range(iterNum):
                random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False)
                old_weight = self.H.weight()
                print(random_ints, i, old_weight, old_weight/nTerm)
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints].copy(), self.H.Zrep[:, random_ints].copy(), self.localSize) 
                old_sub_weight = ps.weight()

                num_branches = 2*self.localSize + (self.localSize * (self.localSize - 1))
                results = Parallel(n_jobs=-1)(
                    delayed(worker_task)(ps.Xrep.copy(), ps.Zrep.copy(), self.localSize, self.sd, idx)
                    for idx in range(num_branches)
                )
                new_sub_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
                
                ls = localSolver(ps, searchDepth=self.sd)
                if new_sub_weight < old_sub_weight:
                    for act_name, args in best_seq:
                        func = getattr(ls.H, act_name)
                        func(*args)
                    self.H.Xrep[:,random_ints] = ls.H.Xrep
                    self.H.Zrep[:,random_ints] = ls.H.Zrep
        if strategy == 'rmax':
            active_matrix = self.H.Xrep | self.H.Zrep
            qubit_weights = active_matrix.sum(axis=0)
            p_q = qubit_weights
            old_weight = self.H.weight()
            for i in range(iterNum):
                p = (p_q + 1e-3) / (p_q + 1e-3).sum()
                for _ in range(2000):
                    random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False, p=p)
                    key = tuple(sorted(int(x) for x in random_ints))
                    if key not in failed_vectors:
                        break
                    else:
                        #print("repeated sampling")
                        pass
                else:
                    print("No new subset found after many retries.")
                    break

                print(random_ints, i, old_weight, old_weight/nTerm,end='\t')
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints].copy(), self.H.Zrep[:, random_ints].copy(),self.localSize) # take space
                old_sub_weight = ps.weight()

                num_branches = 2*self.localSize+ (self.localSize* (self.localSize- 1))
                results = Parallel(n_jobs=-1)(
                    delayed(worker_task)(ps.Xrep.copy(), ps.Zrep.copy(), self.localSize, self.sd, idx)
                    for idx in range(num_branches)
                )
                new_sub_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
                print(best_seq_str, new_sub_weight, old_sub_weight)
                ls = localSolver(ps, searchDepth=self.sd)
                if new_sub_weight < old_sub_weight:
                    for act_name, args in best_seq:
                        func = getattr(ls.H, act_name)
                        func(*args)
                    self.H.Xrep[:,random_ints] = ls.H.Xrep
                    self.H.Zrep[:,random_ints] = ls.H.Zrep
                    local_active = ls.H.Xrep | ls.H.Zrep
                    p_q[random_ints] = local_active.sum(axis = 0)
                    old_weight = old_weight - (old_sub_weight - new_sub_weight)
                if new_sub_weight >= old_sub_weight:
                    failed_vectors.add(key)
                else:
                    failed_vectors.clear()
        if strategy == 'qdrift':
            active_matrix = self.H.Xrep | self.H.Zrep
            p_q = numpy.dot(self.H.coeffs, active_matrix)
            old_weight = self.H.qdriftWeight()
            for i in range(iterNum):
                p = (p_q + 1e-3) / (p_q + 1e-3).sum()
                for _ in range(2000):
                    random_ints = numpy.random.choice(self.qNum, self.localSize, replace=False, p=p)
                    key = tuple(sorted(int(x) for x in random_ints))
                    if key not in failed_vectors:
                        break
                else:
                    print("No new subset found after many retries.")
                    break
                print(random_ints, i, old_weight,end='\t')
                wList.append(old_weight)

                ps = PauliSymplectic(self.H.Xrep[:,random_ints].copy(), self.H.Zrep[:, random_ints].copy(),self.localSize, self.H.coeffs) # take space
                old_sub_weight = ps.qdriftWeight()

                num_branches = 2*self.localSize+ (self.localSize* (self.localSize- 1))
                results = Parallel(n_jobs=-1)(
                    delayed(worker_task)(ps.Xrep.copy(), ps.Zrep.copy(), self.localSize, self.sd, idx, method='qdrift', coeffs = self.H.coeffs)
                    for idx in range(num_branches)
                )
                new_sub_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
                print(best_seq_str, new_sub_weight, old_sub_weight)
                ls = localSolver(ps, searchDepth=self.sd, method='qdrift')
                if new_sub_weight < old_sub_weight:
                    for act_name, args in best_seq:
                        func = getattr(ls.H, act_name)
                        func(*args)
                    self.H.Xrep[:,random_ints] = ls.H.Xrep
                    self.H.Zrep[:,random_ints] = ls.H.Zrep

                    local_active = ls.H.Xrep | ls.H.Zrep
                    p_q[random_ints] = numpy.dot(ls.H.coeffs, local_active)
                    old_weight = old_weight - (old_sub_weight - new_sub_weight)

                if new_sub_weight >= old_sub_weight:
                    failed_vectors.add(key)
                else:
                    failed_vectors.clear()
        if strategy == 'xgboost':
            print("Loading XGBoost surrogate model...")
            model = xgb.XGBRegressor()
            try:
                model.load_model("light_surrogate.json")
            except Exception as e:
                print(f"Error loading model: {e}. Please ensure 'light_surrogate.json' exists.")
                return 
            
            candidate_pool_size = 200

            for i in range(iterNum):
                old_weight = self.H.weight()
                wList.append(old_weight)

                candidates = [numpy.random.choice(self.qNum, self.localSize, replace=False) for _ in range(candidate_pool_size)]
                
                X_infer = numpy.array([extract_features(self.H, c) for c in candidates])
                
                predicted_drops = model.predict(X_infer)
                
                best_idx = numpy.argmax(predicted_drops)
                random_ints = candidates[best_idx]
                print(random_ints, i, old_weight, old_weight/nTerm)
                ps = PauliSymplectic(self.H.Xrep[:,random_ints].copy(), self.H.Zrep[:, random_ints].copy(), self.localSize) 
                old_sub_weight = ps.weight()

                num_branches = 2*self.localSize + (self.localSize * (self.localSize - 1))
                results = Parallel(n_jobs=-1)(
                    delayed(worker_task)(ps.Xrep.copy(), ps.Zrep.copy(), self.localSize, self.sd, idx)
                    for idx in range(num_branches)
                )
                new_sub_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
                
                ls = localSolver(ps, searchDepth=self.sd)
                if new_sub_weight < old_sub_weight:
                    for act_name, args in best_seq:
                        func = getattr(ls.H, act_name)
                        func(*args)
                    self.H.Xrep[:,random_ints] = ls.H.Xrep
                    self.H.Zrep[:,random_ints] = ls.H.Zrep
                # -----------------------
        return numpy.asarray(wList)
            
if __name__ == "__main__":

    jw = []
    bk = []
    tt = []
    op = []
    molecules = (
            ("H_2", "H 0 0 0; H 0 0 0.735"),
            #("LiH", "H 0 0 0; Li 0 0 1.6"),
            #("H_2O", "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"),
            #("CH_4", "6 2.5369 0 0; 1 3.0739 0.31 0; 1 2 -0.31 0; 1 2.2269 0.5369 0; 1 2.8469 -0.5369 0; "),
            #("O_2", "O 0.616 0.0 0.0; O -0.616 0.0 0.0"),
            #("NaF", "11 3 0 0; 9 2 0 0;"),
            #("CO_2", "8 3.732 0.5 0; 8 2 -0.5 0; 6 2.866 0 0;"),
    )

    #for j in range(1,20):
    for m in molecules:
        print("==================")
        numpy.random.seed(1234)
        #H = HoppingHam(j, 20)
        #H = HubbardHam2D(j,j)
        #H = H2D(j,j)
        
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
        #basis="6-31g",
        #charge=0,
        #spin=0,
        #unit=DistanceUnit.ANGSTROM,
        #).run().hamiltonian.second_q_op()

        #with open("%d_%d_hopping.pkl" % (j-1,j), "wb") as f:
        #    pickle.dump(H, f)
        #print("%s" % m[0])
        with open("./../data/mols/%s_sto3g.pkl" % m[0], "rb") as f:
            H = pickle.load(f)


        mapper = JordanWignerMapper()
        mapped_op = mapper.map(H)
        print(mapped_op)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]), mapped_op.coeffs)
        print('jw',ps.weight()/len(mapped_op.paulis), ps.qdriftWeight())
        jw.append(ps.qdriftWeight())

        mapper = BravyiKitaevMapper()
        mapped_op = mapper.map(H)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]), mapped_op.coeffs)
        print('bk',ps.weight()/len(mapped_op.paulis), ps.qdriftWeight())
        bk.append(ps.qdriftWeight())


        mapper = TernaryTreeMapper()
        mapped_op = mapper.map(H)
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)
        ps = PauliSymplectic(x_mat,
                             z_mat,
                             len(mapped_op.paulis[0]), mapped_op.coeffs)
        print('tt', ps.weight()/len(mapped_op.paulis), ps.qdriftWeight())
        tt.append(ps.qdriftWeight())


        mapper = JordanWignerMapper()
        #mapper = BravyiKitaevMapper()
        #mapper = TernaryTreeMapper()
        mapped_op = mapper.map(H)
        print(len(mapped_op.paulis))
        x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
        z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)

        ps = PauliSymplectic(x_mat.copy(),
                             z_mat.copy(),
                             len(mapped_op.paulis[0]), mapped_op.coeffs)

        SEARCH_DEPTH = 6
        gs = globalSolver(ps, 4, SEARCH_DEPTH)
        wrmax = gs(strategy = 'qdrift', iterNum = 2000, nTerm = len(mapped_op.paulis))
        op.append(wrmax[-1])

        print('jw', jw)
        print('bk', bk)
        print('tt', tt)
        print('op', op)

