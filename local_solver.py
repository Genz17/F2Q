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
    def __init__(self, H, searchDepth):
        self.H = H
        self.qNum = self.H.qNum
        self.searchDepth = searchDepth

        # Fixed the strings to target `self.H` since the `h` and `cnot` methods live there
        self.actions = [(self.H.h, i) for i in range(self.qNum)] + \
                       [(self.H.cnot, i, j) for i in range(self.qNum) for j in range(self.qNum) if i != j] + \
                       [(self.H.s, i) for i in range(self.qNum)]

        self.best_weight = self.H.weight()
        self.best_seq = []

    def _search(self, depth, current_seq):
        # Check if the current state is the best we've seen
        current_weight = self.H.weight()
        if current_weight < self.best_weight:
            self.best_weight = current_weight
            self.best_seq = current_seq.copy()

        # Base case for DFS
        if depth == 0:
            return

        for act in self.actions:
            # Optimization: H and CNOT are self-inverse.
            # Doing the same move twice in a row does nothing, so we skip it to prune the tree.
            if current_seq and current_seq[-1] == act:
                continue

            # Apply action
            func, *args = act
            func(*args)
            current_seq.append(act)

            # Recurse deeper
            self._search(depth - 1, current_seq)

            # Backtrack: pop the action and apply it again to reverse the state
            # (Requires no slow memory deepcopies!)
            current_seq.pop()
            func(*args)
    def _format_action(self, act):
        func, *args = act
        func_name = func.__name__
        args_str = ",".join(map(str, args))
        return f"{func_name}({args_str})"

    def _apply(self, seq):
        for act in seq:
            func, *args = act
            func(*args)

        return

    def __call__(self, show = False):
        if show:
            print(f"Starting Search... Initial weight: {self.best_weight}")
            self._search(self.searchDepth, [])
            print(f"Search Complete. Best weight found: {self.best_weight}")
            print("Optimal Sequence:")
            for step in self.best_seq:
                print(f" -> {self._format_action(step)}")
            for act in self.best_seq:
                func, *args = act
                func(*args)
        else:
            iw = self.best_weight
            self._search(self.searchDepth, [])
            fw = self.best_weight
            if fw < iw:
                for act in self.best_seq:
                    func, *args = act
                    func(*args)

        return self.best_weight, self.best_seq

def worker_task(initial_Xrep, initial_Zrep, qNum, search_depth, first_action_idx):
    """
    This function is sent to an independent CPU core.
    It applies ONE specific first move, then runs the DFS tree from there.
    """
    # 1. Reconstruct the local state for this specific CPU core
    ps = PauliSymplectic(initial_Xrep.copy(), initial_Zrep.copy(), qNum)
    solver = localSolver(ps, searchDepth=search_depth)

    # 2. Identify the assigned first action
    first_act = solver.actions[first_action_idx]
    func, *args = first_act

    # 3. Apply the first action
    func(*args)

    # 4. Search the rest of the tree (starting from depth - 1)
    solver._search(search_depth - 1, [first_act])

    # Return the best sequence and weight found in this specific branch
    # We must format the sequence into strings here because function pointers
    # cannot be sent back safely through the multiprocessing pipeline.
    string_seq = [f"{f.__name__}({','.join(map(str, a))})" for f, *a in solver.best_seq]
    action_instructions = []
    for func, *args in solver.best_seq:
        action_instructions.append((func.__name__, args))

    return solver.best_weight, string_seq, action_instructions


if __name__ == "__main__":
    import time
    qNum = 8

    H = HoppingHam(3,8)

    mapper = TernaryTreeMapper()
    mapped_op = mapper.map(H)

    x_mat = numpy.fliplr(PauliList(mapped_op.paulis).x).astype(numpy.int32)
    z_mat = numpy.fliplr(PauliList(mapped_op.paulis).z).astype(numpy.int32)

    ps = PauliSymplectic(x_mat.copy(),
                         z_mat.copy(),
                         len(mapped_op.paulis[0]))
    print(len(mapped_op.paulis[0]))


    SEARCH_DEPTH = 4

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
