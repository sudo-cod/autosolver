# ALGORITHM: ILP two-phase (max coverage, then min score) with bundle-priority greedy fallback

import sys
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
        task_id_list_str = parts[0].strip()
        courier_id = parts[1].strip()
        try:
            score = float(parts[2])
            willingness = float(parts[3])
        except ValueError:
            continue
        task_ids = tuple(t.strip() for t in task_id_list_str.split(","))
        candidates.append((score, task_id_list_str, courier_id, willingness, task_ids))

    if not candidates:
        return []

    all_tasks = set()
    for _, _, _, _, tids in candidates:
        all_tasks.update(tids)

    all_couriers = set(c[2] for c in candidates)

    # Try ILP first
    result = _try_ilp(candidates, all_tasks, all_couriers)
    if result is not None:
        return result

    # Fallback: bundle-aware greedy
    return _bundle_greedy(candidates)


def _try_ilp(candidates, all_tasks, all_couriers):
    try:
        import pulp
    except ImportError:
        return None

    try:
        task_list = sorted(all_tasks)
        courier_list = sorted(all_couriers)
        task_idx = {t: i for i, t in enumerate(task_list)}
        courier_idx = {c: i for i, c in enumerate(courier_list)}

        n_tasks = len(task_list)
        n_couriers = len(courier_list)
        n_cand = len(candidates)

        # Big-M for score minimization phase
        M = 1000000.0

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n_cand)]

        # Each task at most once
        for t in task_list:
            covering = [i for i, c in enumerate(candidates) if t in c[4]]
            if covering:
                prob1 += pulp.lpSum(x[i] for i in covering) <= 1

        # Each courier at most once
        for c in courier_list:
            using = [i for i, cand in enumerate(candidates) if cand[2] == c]
            if using:
                prob1 += pulp.lpSum(x[i] for i in using) <= 1

        # Maximize coverage
        prob1 += pulp.lpSum(
            x[i] * len(candidates[i][4]) for i in range(n_cand)
        )

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5, threads=0)
        prob1.solve(solver)

        if prob1.status != pulp.constants.LpStatusOptimal and prob1.status != pulp.constants.LpStatusNotSolved:
            return None

        max_coverage = sum(
            pulp.value(x[i]) * len(candidates[i][4])
            for i in range(n_cand)
            if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        )

        # Phase 2: Minimize score with coverage constraint
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_cand)]

        for t in task_list:
            covering = [i for i, c in enumerate(candidates) if t in c[4]]
            if covering:
                prob2 += pulp.lpSum(x2[i] for i in covering) <= 1

        for c in courier_list:
            using = [i for i, cand in enumerate(candidates) if cand[2] == c]
            if using:
                prob2 += pulp.lpSum(x2[i] for i in using) <= 1

        # Must achieve max coverage
        prob2 += pulp.lpSum(
            x2[i] * len(candidates[i][4]) for i in range(n_cand)
        ) >= max_coverage - 0.01

        # Minimize total score
        prob2 += pulp.lpSum(
            x2[i] * candidates[i][0] for i in range(n_cand)
        )

        prob2.solve(solver)

        result = []
        for i in range(n_cand):
            if pulp.value(x2[i]) is not None and pulp.value(x2[i]) > 0.5:
                result.append((candidates[i][1], [candidates[i][2]]))

        return result

    except Exception:
        return None


def _bundle_greedy(candidates):
    """
    Greedy that prioritizes 2-task bundles when couriers are scarce,
    otherwise picks by best score-per-task ratio.
    """
    # Count unique couriers and tasks
    all_tasks = set()
    all_couriers = set()
    for _, _, cid, _, tids in candidates:
        all_couriers.add(cid)
        all_tasks.update(tids)

    n_tasks = len(all_tasks)
    n_couriers = len(all_couriers)

    # If couriers are scarce relative to tasks, prioritize bundles
    courier_scarcity = n_tasks / max(n_couriers, 1)

    # Sort: prioritize bundles when scarce, then by score efficiency
    def sort_key(cand):
        score, _, _, _, tids = cand
        n_tids = len(tids)
        if courier_scarcity > 1.5:
            # Prioritize bundles (more tasks per courier), then lower score
            return (-n_tids, score)
        else:
            # Prioritize lower score-per-task
            return (score / n_tids, 0)

    sorted_cands = sorted(candidates, key=sort_key)

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, task_id_list_str, courier_id, willingness, task_ids in sorted_cands:
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))

    return result