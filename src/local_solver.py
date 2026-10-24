from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import numpy

def _action_catalog(num_qubits):
    return tuple(
        [("h", (index,)) for index in range(num_qubits)]
        + [("s", (index,)) for index in range(num_qubits)]
        + [
            ("cnot", (control, target))
            for control in range(num_qubits)
            for target in range(num_qubits)
            if control != target
        ]
    )


def _identity_transformation_key(num_qubits):
    """Pack the columns of the identity 2k-by-2k binary matrix into an int."""

    bits = 2 * num_qubits
    key = 0
    for column in range(bits):
        key |= (1 << column) << (column * bits)
    return key


def _apply_action_to_transformation_key(
    key,
    action,
    num_qubits,
):
    """Compose one phase-free Clifford action with a packed transformation."""

    bits = 2 * num_qubits
    segment_mask = (1 << bits) - 1
    name, arguments = action

    if name == "h":
        qubit = arguments[0]
        x_shift = qubit * bits
        z_shift = (num_qubits + qubit) * bits
        x_column = (key >> x_shift) & segment_mask
        z_column = (key >> z_shift) & segment_mask
        difference = x_column ^ z_column
        return key ^ (difference << x_shift) ^ (difference << z_shift)

    if name == "s":
        qubit = arguments[0]
        x_column = (key >> (qubit * bits)) & segment_mask
        return key ^ (x_column << ((num_qubits + qubit) * bits))

    if name == "cnot":
        control, target = arguments
        x_control = (key >> (control * bits)) & segment_mask
        z_target = (
            key >> ((num_qubits + target) * bits)
        ) & segment_mask
        return (
            key
            ^ (x_control << (target * bits))
            ^ (z_target << ((num_qubits + control) * bits))
        )

    raise ValueError(f"unknown Clifford action: {name}")


def _fwht(values):
    """Return the unnormalized Walsh--Hadamard transform of a 1D array."""

    transformed = numpy.asarray(values).copy()
    block_size = 1
    while block_size < transformed.size:
        blocks = transformed.reshape(-1, 2 * block_size)
        left = blocks[:, :block_size].copy()
        right = blocks[:, block_size:].copy()
        blocks[:, :block_size] = left + right
        blocks[:, block_size:] = left - right
        block_size *= 2
    return transformed


class CliffordSearchSpace:

    def __init__(self, num_qubits, search_depth, actions, parents, action_ids, columns):
        self.num_qubits = num_qubits
        self.search_depth = search_depth
        self.actions = actions
        self.parents = parents
        self.action_ids = action_ids
        self.columns = columns

    @property
    def num_transformations(self):
        return int(self.parents.size)

    def sequence(self, node_index):
        sequence = []
        while node_index:
            sequence.append(self.actions[int(self.action_ids[node_index])])
            node_index = int(self.parents[node_index])
        sequence.reverse()
        return sequence


def _build_search_space(num_qubits, search_depth):
    if not 1 <= num_qubits <= 16:
        raise ValueError("local width must be between 1 and 16 for uint32 masks")
    if search_depth < 0:
        raise ValueError("search_depth must be non-negative")

    actions = _action_catalog(num_qubits)
    cnot_start = 2 * num_qubits
    root_key = _identity_transformation_key(num_qubits)
    keys = [root_key]
    parents = [-1]
    action_ids = [-1]
    seen = {root_key: 0}
    frontier = [0]
    level_ranges = [(0, 1)]

    for depth in range(1, search_depth + 1):
        level_start = len(keys)
        next_frontier = []
        candidate_action_ids = (
            range(len(actions))
            if depth < search_depth
            else range(cnot_start, len(actions))
        )
        for parent_index in frontier:
            parent_key = keys[parent_index]
            for action_id in candidate_action_ids:
                child_key = _apply_action_to_transformation_key(
                    parent_key,
                    actions[action_id],
                    num_qubits,
                )
                if child_key in seen:
                    continue
                child_index = len(keys)
                seen[child_key] = child_index
                keys.append(child_key)
                parents.append(parent_index)
                action_ids.append(action_id)
                next_frontier.append(child_index)
        frontier = next_frontier
        level_ranges.append((level_start, len(keys)))

    parent_array = numpy.asarray(parents, dtype=numpy.int32)
    action_array = numpy.asarray(action_ids, dtype=numpy.int16)
    num_columns = 2 * num_qubits
    column_dtype = numpy.uint16 if num_columns <= 16 else numpy.uint32
    columns = numpy.empty((len(keys), num_columns), dtype=column_dtype)
    columns[0] = numpy.left_shift(
        numpy.asarray(1, dtype=column_dtype),
        numpy.arange(num_columns, dtype=column_dtype),
    )

    for level_start, level_end in level_ranges[1:]:
        level_slice = slice(level_start, level_end)
        columns[level_slice] = columns[parent_array[level_slice]]
        level_actions = action_array[level_slice]
        for action_id in numpy.unique(level_actions):
            relative_rows = numpy.flatnonzero(level_actions == action_id)
            rows = relative_rows + level_start
            name, arguments = actions[int(action_id)]
            if name == "h":
                qubit = arguments[0]
                temporary = columns[rows, qubit].copy()
                columns[rows, qubit] = columns[rows, num_qubits + qubit]
                columns[rows, num_qubits + qubit] = temporary
            elif name == "s":
                qubit = arguments[0]
                columns[rows, num_qubits + qubit] ^= columns[rows, qubit]
            else:
                control, target = arguments
                columns[rows, target] ^= columns[rows, control]
                columns[rows, num_qubits + control] ^= columns[
                    rows, num_qubits + target
                ]

    del seen
    del keys

    parent_array.setflags(write=False)
    action_array.setflags(write=False)
    columns.setflags(write=False)
    return CliffordSearchSpace(
        num_qubits=num_qubits,
        search_depth=search_depth,
        actions=actions,
        parents=parent_array,
        action_ids=action_array,
        columns=columns,
    )


@lru_cache(maxsize=12)
def get_clifford_search_space(
    num_qubits,
    search_depth,
):
    """Return a process-local cached exact Clifford search space."""

    return _build_search_space(int(num_qubits), int(search_depth))


class LocalSolveResult:
    def __init__(self, initial_score, best_score, sequence_strings, actions):
        self.initial_score = initial_score
        self.best_score = best_score
        self.sequence_strings = sequence_strings
        self.actions = actions


def _local_histogram(
    operator,
    qubits,
    method,
):
    if qubits is None:
        qubit_indices = numpy.arange(operator.qNum, dtype=int)
    else:
        qubit_indices = numpy.asarray(qubits, dtype=int)
    num_qubits = int(qubit_indices.size)

    codes = numpy.zeros(operator.term_count, dtype=numpy.int64)
    for local_index, global_index in enumerate(qubit_indices):
        codes |= operator.Xrep[:, global_index].astype(numpy.int64) << local_index
        codes |= (
            operator.Zrep[:, global_index].astype(numpy.int64)
            << (num_qubits + local_index)
        )

    histogram_size = 1 << (2 * num_qubits)
    if method == "weight":
        histogram = numpy.bincount(codes, minlength=histogram_size).astype(
            numpy.int64,
            copy=False,
        )
    elif method == "qdrift":
        if operator.coeffs is None:
            raise ValueError("weighted Pauli weight requires coefficients")
        histogram = numpy.bincount(
            codes,
            weights=operator.coeffs,
            minlength=histogram_size,
        ).astype(numpy.float64, copy=False)
    else:
        raise ValueError(f"unknown objective method: {method}")

    active_indices = numpy.flatnonzero(histogram)
    cache_key = (
        method,
        active_indices.tobytes(),
        numpy.ascontiguousarray(histogram[active_indices]).tobytes(),
    )
    return histogram, cache_key


class LocalSearchEngine:
    """Reusable exact local solver for one width/depth configuration."""

    def __init__(
        self,
        num_qubits,
        search_depth,
        n_jobs = 1,
        cache_size = None,
    ):
        self.space = get_clifford_search_space(num_qubits, search_depth)
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive or -1")
        self.n_jobs = int(n_jobs)
        if cache_size is None:
            maximum_key_bytes = 16 * (1 << (2 * self.num_qubits))
            cache_size = max(
                4,
                min(1024, (64 * 1024**2) // maximum_key_bytes),
            )
        self.cache_size = int(cache_size)
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self._result_cache = OrderedDict()
        self._executor = (
            ThreadPoolExecutor(max_workers=self.n_jobs)
            if self.n_jobs > 1
            else None
        )

    @property
    def num_qubits(self):
        return self.space.num_qubits

    @property
    def search_depth(self):
        return self.space.search_depth

    def _score_chunk(
        self,
        walsh,
        total_mass,
        start,
        stop,
    ):
        columns = self.space.columns
        num_qubits = self.num_qubits
        accumulated = numpy.zeros(stop - start, dtype=walsh.dtype)
        for qubit in range(num_qubits):
            x_column = columns[start:stop, qubit]
            z_column = columns[start:stop, num_qubits + qubit]
            accumulated += walsh[x_column]
            accumulated += walsh[z_column]
            accumulated += walsh[numpy.bitwise_xor(x_column, z_column)]
        # 1[x or z] = (3 - (-1)^x - (-1)^z - (-1)^(x xor z)) / 4.
        numerator = 3 * num_qubits * total_mass - accumulated
        if numpy.issubdtype(walsh.dtype, numpy.integer):
            scores = numerator // 4
        else:
            scores = numerator * 0.25
        relative_index = int(numpy.argmin(scores))
        return scores[relative_index].item(), start + relative_index

    def _best_node(self, histogram):
        walsh = _fwht(histogram)
        total_mass = histogram.sum().item()
        count = self.space.num_transformations

        parallel = self._executor is not None and count >= 65_536
        if parallel:
            if count < 200_000:
                desired_workers = 2
            elif count < 1_000_000:
                desired_workers = 4
            else:
                desired_workers = 8
            useful_workers = min(self.n_jobs, desired_workers)
            chunk_size = min(
                131_072,
                (count + useful_workers - 1) // useful_workers,
            )
        else:
            chunk_size = 131_072
        ranges = [
            (start, min(start + chunk_size, count))
            for start in range(0, count, chunk_size)
        ]
        if not parallel or len(ranges) == 1:
            candidates = [
                self._score_chunk(walsh, total_mass, start, stop)
                for start, stop in ranges
            ]
        else:
            futures = [
                self._executor.submit(
                    self._score_chunk,
                    walsh,
                    total_mass,
                    start,
                    stop,
                )
                for start, stop in ranges
            ]
            candidates = [future.result() for future in futures]
        return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))

    def _score_node_direct(self, histogram, node_index):

        active_indices = numpy.flatnonzero(histogram)
        transformed_weights = numpy.zeros(active_indices.size, dtype=numpy.uint8)
        columns = self.space.columns[node_index]
        parity = _parity_lookup(2 * self.num_qubits)
        for qubit in range(self.num_qubits):
            x_active = parity[
                numpy.bitwise_and(active_indices, columns[qubit])
            ]
            z_active = parity[
                numpy.bitwise_and(
                    active_indices,
                    columns[self.num_qubits + qubit],
                )
            ]
            transformed_weights += numpy.bitwise_or(x_active, z_active)
        return numpy.dot(
            histogram[active_indices],
            transformed_weights,
        ).item()

    def solve(
        self,
        operator,
        *,
        qubits=None,
        method = "weight",
    ):
        histogram, cache_key = _local_histogram(operator, qubits, method)
        pauli_weights = _pauli_weight_lookup(self.num_qubits)
        initial_score = numpy.dot(histogram, pauli_weights).item()

        cached = self._result_cache.get(cache_key)
        if cached is None:
            best_score, best_node = self._best_node(histogram)
            if method == "qdrift":
                # Recheck the winner directly to avoid accepting roundoff-only gains.
                best_score = self._score_node_direct(histogram, best_node)
                tolerance = (
                    128
                    * numpy.finfo(float).eps
                    * max(1.0, abs(initial_score), abs(best_score))
                )
                if best_score >= initial_score - tolerance:
                    best_score = initial_score
                    best_node = 0
            if self.cache_size:
                self._result_cache[cache_key] = (best_score, best_node)
                self._result_cache.move_to_end(cache_key)
                if len(self._result_cache) > self.cache_size:
                    self._result_cache.popitem(last=False)
        else:
            best_score, best_node = cached
            self._result_cache.move_to_end(cache_key)

        actions = self.space.sequence(best_node)
        sequence_strings = [
            f"{name}({','.join(map(str, arguments))})"
            for name, arguments in actions
        ]
        if method == "weight":
            initial_score = int(initial_score)
            best_score = int(round(best_score))
        else:
            initial_score = float(initial_score)
            best_score = float(best_score)
        return LocalSolveResult(
            initial_score=initial_score,
            best_score=best_score,
            sequence_strings=sequence_strings,
            actions=actions,
        )


@lru_cache(maxsize=16)
def _pauli_weight_lookup(num_qubits):
    values = numpy.arange(1 << (2 * num_qubits), dtype=numpy.uint64)
    active = (values & ((1 << num_qubits) - 1)) | (values >> num_qubits)
    byte_view = active.view(numpy.uint8).reshape(active.size, active.itemsize)
    weights = numpy.unpackbits(byte_view, axis=1).sum(axis=1, dtype=numpy.uint8)
    weights.setflags(write=False)
    return weights


@lru_cache(maxsize=16)
def _parity_lookup(num_bits):
    values = numpy.arange(1 << num_bits, dtype=numpy.uint64)
    byte_view = values.view(numpy.uint8).reshape(values.size, values.itemsize)
    parity = (
        numpy.unpackbits(byte_view, axis=1).sum(axis=1, dtype=numpy.uint8)
        & 1
    )
    parity.setflags(write=False)
    return parity


@lru_cache(maxsize=16)
def get_local_search_engine(
    num_qubits,
    search_depth,
    n_jobs = 1,
):
    return LocalSearchEngine(num_qubits, search_depth, n_jobs=n_jobs)


def solve_local_subproblem_detailed(
    H_symp,
    search_depth,
    method="weight",
    n_jobs=1,
    *,
    qubits=None,
    engine=None,
):
    if qubits is None:
        num_qubits = H_symp.qNum
    else:
        qubit_indices = numpy.asarray(qubits, dtype=int)
        num_qubits = int(qubit_indices.size)
        qubits = qubit_indices

    if engine is None:
        engine = get_local_search_engine(num_qubits, search_depth, n_jobs)
    elif engine.search_depth != search_depth:
        raise ValueError("the supplied local-search engine has a different depth")
    return engine.solve(H_symp, qubits=qubits, method=method)


def solve_local_subproblem(
    H_symp,
    search_depth,
    method="weight",
    n_jobs=1,
    *,
    qubits=None,
    engine=None,
):

    result = solve_local_subproblem_detailed(
        H_symp,
        search_depth,
        method=method,
        n_jobs=n_jobs,
        qubits=qubits,
        engine=engine,
    )
    return result.best_score, result.sequence_strings, result.actions
