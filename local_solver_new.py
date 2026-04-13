from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import PauliList
from HamiltonianGen import *
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from joblib import Parallel, delayed
import numpy
import copy

class localSolver():
    def __init__(self, H_symp, searchDepth, method='weight'):
        self.H = H_symp
        self.qNum = self.H.qNum
        self.searchDepth = searchDepth

        # Group actions as (function, args_tuple) for clean unpacking
        self.actions = [(self.H.h, (i,)) for i in range(self.qNum)] + \
                       [(self.H.s, (i,)) for i in range(self.qNum)] + \
                       [(self.H.cnot, (i, j)) for i in range(self.qNum) for j in range(self.qNum) if i != j]

        # Assign a strict ordering index to each action for commutation pruning
        self.action_idx = {act: idx for idx, act in enumerate(self.actions)}
        self.method = method

        if self.method == 'weight':
            self.best_weight = self.H.weight()
        else:
            self.best_weight = self.H.qdriftWeight()
        self.best_seq = []

    def _commutes(self, act1, act2):
        """Evaluates if two actions mathematically commute."""
        name1, args1 = act1[0].__name__, act1[1]
        name2, args2 = act2[0].__name__, act2[1]

        # 1. Disjoint qubits always commute
        if not set(args1).intersection(set(args2)):
            return True

        # 2. CNOTs commute if they share ONLY the same control or ONLY the same target
        if name1 == 'cnot' and name2 == 'cnot':
            return (args1[0] == args2[0]) or (args1[1] == args2[1])

        # 3. S commutes with CNOT ONLY if S is on the CNOT's control qubit
        if name1 == 's' and name2 == 'cnot':
            return args1[0] == args2[0]
        if name1 == 'cnot' and name2 == 's':
            return args1[0] == args2[0]

        return False

    def _search(self, depth, current_seq):
        if self.method == 'weight':
            current_weight = self.H.weight()
            if current_weight < self.best_weight:
                self.best_weight = current_weight
                self.best_seq = current_seq.copy()

            if depth == 0:
                return

            for act in self.actions:
                if current_seq:
                    prev_act = current_seq[-1]
                    
                    # Pruning 1: All operations are self-inverse in boolean symplectic rep.
                    # Applying the exact same move twice does nothing.
                    if act == prev_act:
                        continue
                    
                    # Pruning 2: Lexicographical Ordering for Commutation.
                    # If two moves commute, only allow them in one specific index order.
                    if self._commutes(prev_act, act) and self.action_idx[act] < self.action_idx[prev_act]:
                        continue

                # Apply
                func, args = act
                func(*args)
                current_seq.append(act)

                # Recurse
                self._search(depth - 1, current_seq)

                # Backtrack: Neatly undo by reapplying the self-inverse action
                current_seq.pop()
                func(*args)
        else:
            current_weight = self.H.qdriftWeight()
            if current_weight < self.best_weight:
                self.best_weight = current_weight
                self.best_seq = current_seq.copy()

            if depth == 0:
                return

            for act in self.actions:
                if current_seq:
                    prev_act = current_seq[-1]
                    
                    # Pruning 1: All operations are self-inverse in boolean symplectic rep.
                    # Applying the exact same move twice does nothing.
                    if act == prev_act:
                        continue
                    
                    # Pruning 2: Lexicographical Ordering for Commutation.
                    # If two moves commute, only allow them in one specific index order.
                    if self._commutes(prev_act, act) and self.action_idx[act] < self.action_idx[prev_act]:
                        continue

                # Apply
                func, args = act
                func(*args)
                current_seq.append(act)

                # Recurse
                self._search(depth - 1, current_seq)

                # Backtrack: Neatly undo by reapplying the self-inverse action
                current_seq.pop()
                func(*args)

    def __call__(self, show=False):
        if show:
            print(f"Starting Search... Initial weight: {self.best_weight}")
        
        self._search(self.searchDepth, [])
        
        if show:
            print(f"Search Complete. Best weight found: {self.best_weight}")
            print("Optimal Sequence:")
            
        # Re-apply the optimal sequence to leave self.H in the minimized state
        for act in self.best_seq:
            func, args = act
            func(*args)
            if show:
                args_str = ",".join(map(str, args))
                print(f" -> {func.__name__}({args_str})")

        return self.best_weight, self.best_seq

def worker_task(initial_Xrep, initial_Zrep, qNum, search_depth, first_action_idx, method='weight', coeffs=None):
    """
    This function is sent to an independent CPU core.
    It applies ONE specific first move, then runs the DFS tree from there.
    """
    # 1. Reconstruct the local state for this specific CPU core
    ps = PauliSymplectic(initial_Xrep.copy(), initial_Zrep.copy(), qNum, coeffs)
    solver = localSolver(ps, searchDepth=search_depth, method=method)

    # 2. Identify the assigned first action
    first_act = solver.actions[first_action_idx]
    func, args = first_act

    # 3. Apply the first action
    func(*args)

    # 4. Search the rest of the tree (starting from depth - 1)
    solver._search(search_depth - 1, [first_act])

    # Return the best sequence and weight found in this specific branch
    # We must format the sequence into strings here because function pointers
    # cannot be sent back safely through the multiprocessing pipeline.
    string_seq = [f"{f.__name__}({','.join(map(str, a))})" for f, *a in solver.best_seq]
    action_instructions = []
    for func, args in solver.best_seq:
        action_instructions.append((func.__name__, args))

    return solver.best_weight, string_seq, action_instructions



if __name__ == "__main__":
    import time
    qNum = 8

    H = HoppingHam(8,9)

    mapper = TernaryTreeMapper()
    mapped_op = mapper.map(H)

    x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
    z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)

    ps = PauliSymplectic(x_mat.copy(),
                         z_mat.copy(),
                         len(mapped_op.paulis[0]))
    print(len(mapped_op.paulis[0]))


    SEARCH_DEPTH = 5

    #s = time.perf_counter()
    #solver = localSolver(ps, searchDepth=SEARCH_DEPTH)
    #solver(show=True)
    #e = time.perf_counter()
    #print('time: ', e-s)
    #print(solver.best_seq)



    num_branches = 2*qNum + (qNum * (qNum - 1))
    s = time.perf_counter()
    results = Parallel(n_jobs=-1)(
        delayed(worker_task)(x_mat.copy(), z_mat.copy(), qNum, SEARCH_DEPTH, idx)
        for idx in range(num_branches)
    )
    e = time.perf_counter()
    print('time: ', e-s)

    best_overall_weight, best_seq_str, best_seq = min(results, key=lambda x: x[0])
    print(f"\nSearch Complete. Best weight found: {best_overall_weight}")
    print(best_seq_str)

