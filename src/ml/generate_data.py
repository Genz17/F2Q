import sys
sys.path.append("./../")
from joblib import Parallel, delayed
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import HoppingHam, PauliSymplectic, HubbardHam2D
from local_solver import localSolver, worker_task 
import numpy as np
import copy
import time

def extract_features(H, selected_qubits, active_matrix):
    activities = active_matrix[:, selected_qubits].sum(axis=0)
    activities = np.sort(activities)[::-1]
    
    row_sums = active_matrix[:, selected_qubits].sum(axis=1)
    strong_couplings = np.sum(row_sums >= 2)
    current_weight = active_matrix.sum()
    
    return np.array([
        activities[0], activities[1], activities[2], activities[3],
        strong_couplings, current_weight
    ])

def generate_training_data(qNum=30, localSize=4, searchDepth=4, num_restarts=100, steps_per_run=100):
    X_data = []
    y_data = []
    
    start_time = time.perf_counter()

    for run in range(num_restarts):
        print(run, num_restarts)
        rr = np.random.randint(21, qNum-1, 1)[0]
        H_initial_op = HoppingHam(rr, qNum)
        mapper = JordanWignerMapper()
        mapped_op = mapper.map(H_initial_op)
        
        initial_x_mat = np.fliplr(PauliList(mapped_op.paulis).x).astype(np.int32)
        initial_z_mat = np.fliplr(PauliList(mapped_op.paulis).z).astype(np.int32)
        num_paulis = len(mapped_op.paulis[0])

        ps_global = PauliSymplectic(initial_x_mat.copy(), initial_z_mat.copy(), num_paulis)
        
        for step in range(steps_per_run):
            print(step, steps_per_run, end='\t')
            active_matrix = ps_global.Xrep | ps_global.Zrep
            random_ints = np.random.choice(qNum, localSize, replace=False)
            
            features = extract_features(ps_global, random_ints, active_matrix)
            old_w = ps_global.weight()
            

            ps_local = PauliSymplectic(ps_global.Xrep[:, random_ints], ps_global.Zrep[:, random_ints], localSize)

            num_branches = 2*localSize+ (localSize* (localSize- 1))
            results = Parallel(n_jobs=-1)(
                delayed(worker_task)(ps_local.Xrep.copy(), ps_local.Zrep.copy(), localSize, searchDepth, idx)
                for idx in range(num_branches)
            )
            new_sub_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
            ls = localSolver(ps_local, searchDepth=searchDepth)

            for act_name, args in best_seq:
                func = getattr(ls.H, act_name)
                func(*args)
            
            ps_global.Xrep[:, random_ints] = ls.H.Xrep
            ps_global.Zrep[:, random_ints] = ls.H.Zrep
            
            new_w = ps_global.weight()
            drop_amount = max(0, old_w - new_w) 
            
            X_data.append(features)
            y_data.append(drop_amount)
            
            if drop_amount > 0:
                print("drop")

    X_array = np.array(X_data)
    y_array = np.array(y_data)
    
    np.savez('xgboost_training_data.npz', X=X_array, y=y_array)
    
    end_time = time.perf_counter()
    print(f"\nCompleted. Time used:{end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    generate_training_data()
