# ALGORITHM: ILP two-phase (max coverage then min score) with robust fallback greedy

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

    # Try ILP first
    result = _try_ilp(candidates)
    if result is not None:
        return result

    # Fallback: greedy
    return _greedy_fallback(candidates)


def _try_ilp(candidates):
    try:
        import pulp
    except ImportError:
        return None

    try:
        all_tasks = set()
        for _, _, _, _, task_ids in candidates:
            for t in task_ids:
                all_tasks.add(t)

        n = len(candidates)
        task_list = sorted(all_tasks)
        nt = len(task_list)

        courier_cands = defaultdict(list)
        for i, (_, _, cid, _, _) in enumerate(candidates):
            courier_cands[cid].append(i)

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{i}", cat='Binary') for i in range(n)]
        y = [pulp.LpVariable(f"y{j}", cat='Binary') for j in range(nt)]

        prob1 += pulp.lpSum(y)

        for cid, cands in courier_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        for j, t in enumerate(task_list):
            covering = []
            for i, (_, _, _, _, tids) in enumerate(candidates):
                if t in tids:
                    covering.append(i)
            if covering:
                prob1 += y[j] <= pulp.lpSum(x[i] for i in covering)
            else:
                prob1 += y[j] == 0

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=3, threads=1)
        prob1.solve(solver)

        status1 = pulp.LpStatus[prob1.status]
        if status1 not in ('Optimal', 'Not Solved', 'Feasible'):
            return None

        max_coverage = sum(pulp.value(y[j]) or 0 for j in range(nt))

        # Phase 2: Minimize score with coverage constraint
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"x2_{i}", cat='Binary') for i in range(n)]
        y2 = [pulp.LpVariable(f"y2_{j}", cat='Binary') for j in range(nt)]

        prob2 += pulp.lpSum(candidates[i][0] * x2[i] for i in range(n))
        prob2 += pulp.lpSum(y2) >= max_coverage - 0.01

        for cid, cands in courier_cands.items():
            prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

        for j, t in enumerate(task_list):
            covering = []
            for i, (_, _, _, _, tids) in enumerate(candidates):
                if t in tids:
                    covering.append(i)
            if covering:
                prob2 += y2[j] <= pulp.lpSum(x2[i] for i in covering)
                prob2 += pulp.lpSum(x2[i] for i in covering) <= 1
            else:
                prob2 += y2[j] == 0

        solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=3, threads=1)
        prob2.solve(solver2)

        status2 = pulp.LpStatus[prob2.status]
        if status2 not in ('Optimal', 'Not Solved', 'Feasible'):
            return None

        result = []
        for i in range(n):
            val = pulp.value(x2[i])
            if val is not None and val > 0.5:
                _, task_id_list_str, cid, _, _ = candidates[i]
                result.append((task_id_list_str, [cid]))

        return result

    except Exception:
        return None


def _greedy_fallback(candidates):
    """Simple greedy by score ascending - the proven baseline."""
    candidates_sorted = sorted(candidates, key=lambda x: x[0])

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, task_id_list_str, courier_id, willingness, task_ids in candidates_sorted:
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))

    return result