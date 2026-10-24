"""Randomized Subsystem Descent."""

from itertools import combinations
from math import comb
import numpy
from local_solver import (get_local_search_engine, solve_local_subproblem_detailed)


def extract_features(
    H,
    selected_qubits,
    *,
    active_matrix=None,
    current_weight=None,
):
    if active_matrix is None:
        active_matrix = H.Xrep | H.Zrep
    selected_activity = active_matrix[:, selected_qubits]
    activities = selected_activity.sum(axis=0)
    activities = numpy.sort(activities)[::-1]
    activities = numpy.pad(activities, (0, max(0, 4 - len(activities))))

    row_sums = selected_activity.sum(axis=1)
    strong_couplings = numpy.sum(row_sums >= 2)
    if current_weight is None:
        current_weight = active_matrix.sum()

    return numpy.array([
        activities[0], activities[1], activities[2], activities[3],
        strong_couplings, current_weight
    ], dtype=float)


class OptimizationResult:
    """Objective history and accepted Clifford circuit."""

    def __init__(self, history_iterations, cost_history, iterations_requested, iterations_completed, accepted_moves, circuit):
        self.history_iterations = history_iterations
        self.cost_history = cost_history
        self.iterations_requested = iterations_requested
        self.iterations_completed = iterations_completed
        self.accepted_moves = accepted_moves
        self.circuit = circuit

    @property
    def initial_cost(self):
        return float(self.cost_history[0])

    @property
    def final_cost(self):
        return float(self.cost_history[-1])


class globalSolver:
    """Optimize H in place and conjugate tracked observables by accepted moves."""

    def __init__(
        self,
        H,
        localSize,
        SEARCH_DEPTH,
        tracked_observables = None,
        n_jobs = 1,
        seed = None,
    ):
        self.H = H
        self.qNum = self.H.qNum
        self.localSize = int(localSize)
        self.sd = int(SEARCH_DEPTH)
        if not 1 <= self.localSize <= self.qNum:
            raise ValueError("localSize must lie between 1 and the number of qubits")
        self.tracked_observables = dict(tracked_observables or {})
        for name, observable in self.tracked_observables.items():
            if observable.qNum != self.qNum:
                raise ValueError(
                    f"tracked observable {name!r} has {observable.qNum} qubits; "
                    f"expected {self.qNum}"
                )
        self.n_jobs = int(n_jobs)
        self.rng = (
            seed if isinstance(seed, numpy.random.Generator)
            else numpy.random.default_rng(seed)
        )
        self.circuit = []
        self.last_result = None
        self.local_engine = None
        self.num_subsystems = comb(self.qNum, self.localSize)
        self._enumerated_subsystems = None

    @staticmethod
    def _normalize_strategy(strategy):
        aliases = {
            "random": "uniform",
            "rmax": "hamming",
            "qdrift": "weighted_hamming",
            "xgboost": "ml",
        }
        return aliases.get(strategy, strategy)

    @staticmethod
    def _objective(operator, method):
        if method == "weight":
            return operator.weight()
        if method == "qdrift":
            return operator.qdriftWeight()
        raise ValueError(f"unknown objective method: {method}")

    def _qubit_scores(self, method):
        active = self.H.Xrep | self.H.Zrep
        if method == "weight":
            return active.sum(axis=0, dtype=numpy.float64)
        if self.H.coeffs is None:
            raise ValueError("weighted optimization requires Hamiltonian coefficients")
        return numpy.dot(self.H.coeffs, active)

    @staticmethod
    def _successive_subset_probabilities(subsets, probabilities):
        """Sum weighted sampling probabilities over all orders of each subset."""

        subset_array = numpy.asarray(subsets, dtype=int)
        if subset_array.ndim == 1:
            subset_array = subset_array.reshape(1, -1)
        weights = probabilities[subset_array]
        batch_size, subset_size = weights.shape
        weight_sums = numpy.zeros(
            (batch_size, 1 << subset_size),
            dtype=float,
        )
        path_probabilities = numpy.zeros_like(weight_sums)
        path_probabilities[:, 0] = 1.0
        for mask in range(1, 1 << subset_size):
            least_bit = mask & -mask
            bit_index = least_bit.bit_length() - 1
            weight_sums[:, mask] = (
                weight_sums[:, mask ^ least_bit] + weights[:, bit_index]
            )
            probability = numpy.zeros(batch_size, dtype=float)
            remaining_bits = mask
            while remaining_bits:
                final_bit = remaining_bits & -remaining_bits
                final_index = final_bit.bit_length() - 1
                prefix = mask ^ final_bit
                probability += (
                    path_probabilities[:, prefix]
                    * weights[:, final_index]
                    / (1.0 - weight_sums[:, prefix])
                )
                remaining_bits ^= final_bit
            path_probabilities[:, mask] = probability
        return path_probabilities[:, -1]

    def _sample_subset(
        self,
        strategy,
        failed_vectors,
        probabilities,
        model,
        candidate_pool_size,
        max_sampling_attempts,
    ):
        if len(failed_vectors) >= self.num_subsystems:
            return None, None
        if strategy == "ml":
            remaining_attempts = max_sampling_attempts
            active_matrix = self.H.Xrep | self.H.Zrep
            current_weight = active_matrix.sum()
            while remaining_attempts > 0:
                pool_size = min(candidate_pool_size, remaining_attempts)
                candidates = [
                    numpy.sort(
                        self.rng.choice(
                            self.qNum,
                            self.localSize,
                            replace=False,
                        )
                    )
                    for _ in range(pool_size)
                ]
                features = numpy.asarray(
                    [
                        extract_features(
                            self.H,
                            candidate,
                            active_matrix=active_matrix,
                            current_weight=current_weight,
                        )
                        for candidate in candidates
                    ]
                )
                predictions = model.predict(features)
                for candidate_index in numpy.argsort(predictions)[::-1]:
                    selected = candidates[int(candidate_index)]
                    key = tuple(int(index) for index in selected)
                    if key not in failed_vectors:
                        return selected, key
                remaining_attempts -= pool_size
            return None, None

        if strategy not in {"uniform", "hamming", "weighted_hamming"}:
            raise ValueError(f"unknown sampling strategy: {strategy}")

        # Enumerating remaining subsets avoids repeated failed draws near convergence.
        enumerable = self.num_subsystems <= 100_000
        retry_budget = (
            min(max_sampling_attempts, 64)
            if enumerable
            else max_sampling_attempts
        )
        for _ in range(retry_budget):
            if probabilities is None:
                selected = self.rng.choice(
                    self.qNum,
                    self.localSize,
                    replace=False,
                )
            else:
                # Exponential clocks give weighted sampling without replacement.
                clocks = -numpy.log(
                    numpy.maximum(
                        self.rng.random(self.qNum),
                        numpy.finfo(float).tiny,
                    )
                ) / probabilities
                selected = numpy.argpartition(
                    clocks,
                    self.localSize - 1,
                )[: self.localSize]
            selected.sort()
            key = tuple(int(index) for index in selected)
            if key not in failed_vectors:
                return selected, key

        if enumerable:
            if self._enumerated_subsystems is None:
                self._enumerated_subsystems = tuple(
                    combinations(range(self.qNum), self.localSize)
                )
            remaining = [
                subset
                for subset in self._enumerated_subsystems
                if subset not in failed_vectors
            ]
            if not remaining:
                return None, None
            if probabilities is None:
                chosen_index = int(self.rng.integers(len(remaining)))
            else:
                probability_chunks = (
                    self._successive_subset_probabilities(
                        remaining[start : start + 8192],
                        probabilities,
                    )
                    for start in range(0, len(remaining), 8192)
                )
                remaining_weights = numpy.concatenate(tuple(probability_chunks))
                remaining_weights /= remaining_weights.sum()
                chosen_index = int(
                    self.rng.choice(len(remaining), p=remaining_weights)
                )
            key = remaining[chosen_index]
            return numpy.asarray(key, dtype=int), key
        return None, None

    def run(
        self,
        iterNum=15000,
        strategy="hamming",
        nTerm=1,
        *,
        model=None,
        candidate_pool_size=200,
        record_every=1,
        max_sampling_attempts=2000,
        verbose=False,
    ):
        """Optimize in place and return the initial/final costs and accepted gates."""

        del nTerm  # retained only for backward-compatible call signatures
        if record_every < 1:
            raise ValueError("record_every must be positive")
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")

        strategy = self._normalize_strategy(strategy)
        method = "qdrift" if strategy == "weighted_hamming" else "weight"
        if strategy == "ml" and model is None:
            raise ValueError("the ML sampling strategy requires a fitted model")

        if self.local_engine is None:
            self.local_engine = get_local_search_engine(
                self.localSize,
                self.sd,
                self.n_jobs,
            )
        failed_vectors = set()
        qubit_scores = self._qubit_scores(method)
        probabilities = None
        if strategy in {"hamming", "weighted_hamming"}:
            probabilities = qubit_scores + 1e-3
            probabilities /= probabilities.sum()
        current_cost = self._objective(self.H, method)
        history_iterations = [0]
        cost_history = [current_cost]
        accepted_moves = 0
        iterations_completed = 0
        for iteration in range(1, iterNum + 1):
            selected, key = self._sample_subset(
                strategy,
                failed_vectors,
                probabilities,
                model,
                candidate_pool_size,
                max_sampling_attempts,
            )
            if selected is None:
                if verbose:
                    print("No unfailed subsystem found within the retry budget.")
                break

            local_result = solve_local_subproblem_detailed(
                self.H,
                self.sd,
                method=method,
                n_jobs=self.n_jobs,
                qubits=selected,
                engine=self.local_engine,
            )
            old_sub_cost = local_result.initial_score
            new_sub_cost = local_result.best_score
            best_sequence = local_result.actions
            improved = new_sub_cost < old_sub_cost

            if improved:
                self.H.apply_actions(best_sequence, qubit_map=selected)

                mapped_sequence = []
                for action_name, args in best_sequence:
                    global_args = tuple(int(selected[index]) for index in args)
                    mapped_sequence.append((action_name, global_args))
                    for observable in self.tracked_observables.values():
                        getattr(observable, action_name)(*global_args)
                self.circuit.extend(mapped_sequence)

                current_cost += new_sub_cost - old_sub_cost
                accepted_moves += 1
                failed_vectors.clear()
                qubit_scores = self._qubit_scores(method)
                if probabilities is not None:
                    probabilities = qubit_scores + 1e-3
                    probabilities /= probabilities.sum()
            else:
                failed_vectors.add(key)

            iterations_completed = iteration
            if iteration % record_every == 0 or iteration == iterNum:
                history_iterations.append(iteration)
                cost_history.append(current_cost)

            if verbose:
                print(
                    f"iteration={iteration} subset={selected.tolist()} "
                    f"cost={current_cost} accepted={improved} "
                    f"sequence={local_result.sequence_strings}"
                )

        current_cost = self._objective(self.H, method)

        if history_iterations[-1] != iterations_completed:
            history_iterations.append(iterations_completed)
            cost_history.append(current_cost)
        else:
            cost_history[-1] = current_cost

        result = OptimizationResult(
            history_iterations=numpy.asarray(history_iterations, dtype=int),
            cost_history=numpy.asarray(cost_history),
            iterations_requested=int(iterNum),
            iterations_completed=int(iterations_completed),
            accepted_moves=accepted_moves,
            circuit=self.circuit.copy(),
        )
        self.last_result = result
        return result

    def __call__(self, iterNum=15000, strategy="hamming", nTerm=1, **kwargs):
        return self.run(
            iterNum=iterNum,
            strategy=strategy,
            nTerm=nTerm,
            **kwargs,
        ).cost_history

