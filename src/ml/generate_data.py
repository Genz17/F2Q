"""Generate local-subsystem examples for the learned sampler."""

from pathlib import Path
import sys

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from global_solver import extract_features
from HamiltonianGen import HoppingHam, PauliSymplectic
from local_solver import get_local_search_engine, solve_local_subproblem_detailed


def generate_training_data(
    qNum=30,
    localSize=4,
    searchDepth=4,
    num_restarts=100,
    steps_per_run=100,
    *,
    seed=0,
    n_jobs=1,
    output="xgboost_training_data.npz",
):
    features, labels = [], []
    rng = np.random.default_rng(seed)
    mapper = JordanWignerMapper()
    engine = get_local_search_engine(localSize, searchDepth, n_jobs)

    for _ in range(num_restarts):
        hopping_range = int(rng.integers(1, qNum))
        state = PauliSymplectic.from_sparse_pauli_op(
            mapper.map(HoppingHam(hopping_range, qNum))
        )
        for _ in range(steps_per_run):
            selected = np.sort(rng.choice(state.qNum, localSize, replace=False))
            features.append(extract_features(state, selected))
            result = solve_local_subproblem_detailed(
                state,
                searchDepth,
                method="weight",
                n_jobs=n_jobs,
                qubits=selected,
                engine=engine,
            )
            drop = max(0, result.initial_score - result.best_score)
            labels.append(float(drop))
            if drop:
                state.apply_actions(result.actions, qubit_map=selected)

    x_array, y_array = np.asarray(features), np.asarray(labels)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, X=x_array, y=y_array)
    return x_array, y_array


if __name__ == "__main__":
    generate_training_data()
