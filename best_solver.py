import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    candidates = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness))

    if not candidates:
        return []

    # Try ILP approach first
    result = _solve_ilp(candidates)
    if result is not None:
        return result

    # Fallback to greedy
    return _solve_greedy(candidates)


def _calibrated_cost(score, willingness, num_tasks):
    return willingness * score + (1.0 - willingness) * 100.0 * num_tasks


def _solve_greedy(candidates):
    # Sort by calibrated cost per task ascending, prefer bundles at same cost-per-task
    indexed_cands = []
    for i, (score, task_id_list_str, courier_id, willingness) in enumerate(candidates):
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        n = len(task_ids)
        cc = _calibrated_cost(score, willingness, n)
        indexed_cands.append((cc / n, cc, -n, i, task_id_list_str, courier_id, task_ids))

    indexed_cands.sort()

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for _, cc, neg_n, i, task_id_list_str, courier_id, task_ids in indexed_cands:
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))

    return result


def _solve_ilp(candidates):
    try:
        import pulp
    except ImportError:
        return None

    if not candidates:
        return []

    cand_data = []
    all_tasks = set()
    all_couriers = set()
    task_to_cands = defaultdict(list)
    courier_to_cands = defaultdict(list)

    for i, (score, task_id_list_str, courier_id, willingness) in enumerate(candidates):
        task_ids = tuple(t.strip() for t in task_id_list_str.split(","))
        num_tasks = len(task_ids)
        cal_cost = _calibrated_cost(score, willingness, num_tasks)
        cand_data.append((score, task_id_list_str, courier_id, willingness, task_ids, num_tasks, cal_cost))
        all_couriers.add(courier_id)
        for t in task_ids:
            all_tasks.add(t)
            task_to_cands[t].append(i)
        courier_to_cands[courier_id].append(i)

    task_list = sorted(all_tasks)
    courier_list = sorted(all_couriers)
    n_tasks = len(task_list)
    n_couriers = len(courier_list)
    n_cands = len(cand_data)

    # Single-phase ILP:
    # Decision: which candidates to select
    # Each candidate i has calibrated cost cc_i and covers n_i tasks
    # If we DON'T cover a task, we pay 100
    # Total cost = sum_i(cc_i * x_i) + 100 * (n_tasks - sum_j y_j)
    # where y_j = 1 if task j is covered
    # = sum_i(cc_i * x_i) - 100 * sum_j(y_j) + const
    # Minimize: sum_i(cc_i * x_i) - 100 * sum_j(y_j)

    prob = pulp.LpProblem("MinCalCost", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_cands)]
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_tasks)]

    # Objective
    prob += pulp.lpSum(x[i] * cand_data[i][6] for i in range(n_cands)) - 100.0 * pulp.lpSum(
        y[j] for j in range(n_tasks)
    )

    # Each courier used at most once
    for c in courier_list:
        cands = courier_to_cands[c]
        if len(cands) > 1:
            prob += pulp.lpSum(x[i] for i in cands) <= 1

    # Task coverage linking
    for j, t in enumerate(task_list):
        cands = task_to_cands[t]
        prob += y[j] <= pulp.lpSum(x[i] for i in cands)

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=8, threads=0)
    prob.solve(solver)

    if prob.status in (pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved):
        result = []
        for i in range(n_cands):
            val = pulp.value(x[i])
            if val is not None and val > 0.5:
                _, task_id_list_str, courier_id, _, _, _, _ = cand_data[i]
                result.append((task_id_list_str, [courier_id]))
        if result:
            return result

    return None