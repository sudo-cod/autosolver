# ALGORITHM: regret-aware greedy with coverage-first priority, then ILP refinement

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

    # Try ILP first
    result = _try_ilp(candidates)
    if result is not None:
        return result

    # Fallback: regret-aware greedy
    return _regret_greedy(candidates)


def _try_ilp(candidates):
    try:
        import pulp
    except ImportError:
        return None

    if not candidates:
        return []

    # Collect all tasks and couriers
    all_tasks = set()
    for _, _, _, _, task_ids in candidates:
        for t in task_ids:
            all_tasks.add(t)

    # Build index: candidate index -> (score, task_ids, courier_id)
    n = len(candidates)

    # Phase 1: Maximize coverage
    prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat='Binary') for i in range(n)]

    # Objective: maximize number of tasks covered
    task_list = sorted(all_tasks)
    task_idx = {t: i for i, t in enumerate(task_list)}
    nt = len(task_list)

    # y[j] = 1 if task j is covered
    y = [pulp.LpVariable(f"y{j}", cat='Binary') for j in range(nt)]

    # Maximize coverage
    prob1 += pulp.lpSum(y)

    # Each courier at most once
    courier_cands = defaultdict(list)
    for i, (_, _, cid, _, _) in enumerate(candidates):
        courier_cands[cid].append(i)

    for cid, cands in courier_cands.items():
        prob1 += pulp.lpSum(x[i] for i in cands) <= 1

    # Task coverage linking
    for j, t in enumerate(task_list):
        covering = []
        for i, (_, _, _, _, tids) in enumerate(candidates):
            if t in tids:
                covering.append(i)
        if covering:
            prob1 += y[j] <= pulp.lpSum(x[i] for i in covering)
        else:
            prob1 += y[j] == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=4)
    prob1.solve(solver)

    if prob1.status != pulp.constants.LpStatusOptimal and prob1.status != pulp.constants.LpStatusNotSolved:
        return None

    max_coverage = sum(pulp.value(y[j]) or 0 for j in range(nt))

    # Phase 2: Minimize score with coverage constraint
    prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
    x2 = [pulp.LpVariable(f"x2_{i}", cat='Binary') for i in range(n)]
    y2 = [pulp.LpVariable(f"y2_{j}", cat='Binary') for j in range(nt)]

    # Minimize total score
    prob2 += pulp.lpSum(candidates[i][0] * x2[i] for i in range(n))

    # Must achieve max coverage
    prob2 += pulp.lpSum(y2) >= max_coverage - 0.5

    # Each courier at most once
    for cid, cands in courier_cands.items():
        prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

    # Task coverage linking
    for j, t in enumerate(task_list):
        covering = []
        for i, (_, _, _, _, tids) in enumerate(candidates):
            if t in tids:
                covering.append(i)
        if covering:
            prob2 += y2[j] <= pulp.lpSum(x2[i] for i in covering)
        else:
            prob2 += y2[j] == 0

    # Each task at most once
    for j, t in enumerate(task_list):
        covering = []
        for i, (_, _, _, _, tids) in enumerate(candidates):
            if t in tids:
                covering.append(i)
        if covering:
            prob2 += pulp.lpSum(x2[i] for i in covering) <= 1

    solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4)
    prob2.solve(solver2)

    if prob2.status not in (pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved):
        return None

    result = []
    for i in range(n):
        val = pulp.value(x2[i])
        if val is not None and val > 0.5:
            _, task_id_list_str, cid, _, _ = candidates[i]
            result.append((task_id_list_str, [cid]))

    return result


def _regret_greedy(candidates):
    """Regret-aware greedy: prioritize bundles and high-regret assignments."""
    if not candidates:
        return []

    # Group by courier
    courier_cands = defaultdict(list)
    for i, (score, tstr, cid, w, tids) in enumerate(candidates):
        courier_cands[cid].append(i)

    # For each task, find which candidates cover it
    task_cands = defaultdict(list)
    for i, (_, _, _, _, tids) in enumerate(candidates):
        for t in tids:
            task_cands[t].append(i)

    all_tasks = set(task_cands.keys())

    # Compute "regret" for each candidate = opportunity cost
    # A candidate with high regret should be prioritized
    # Regret = (best alternative for its tasks) - (this candidate's score)
    # But more importantly, we want to maximize coverage first

    # Strategy: score-per-task with bundle bonus
    # For each candidate, compute effective cost = score / (num_tasks ^ alpha)
    # where alpha > 1 to favor bundles

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    # Sort candidates by a composite score that favors bundles and low cost
    def candidate_priority(idx):
        score, _, _, _, tids = candidates[idx]
        nt = len(tids)
        # Favor bundles: divide by nt^1.5 so 2-task bundles get significant advantage
        return score / (nt ** 1.5)

    remaining = set(range(len(candidates)))

    while remaining:
        # Find best candidate
        best = None
        best_pri = float('inf')
        for i in remaining:
            score, _, cid, _, tids = candidates[i]
            if cid in assigned_couriers:
                continue
            if any(t in assigned_tasks for t in tids):
                continue
            pri = candidate_priority(i)
            if pri < best_pri:
                best_pri = pri
                best = i

        if best is None:
            break

        score, tstr, cid, w, tids = candidates[best]
        assigned_couriers.add(cid)
        for t in tids:
            assigned_tasks.add(t)
        result.append((tstr, [cid]))
        remaining.discard(best)

    return result