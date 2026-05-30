import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
    """
    Two-phase ILP: first maximize coverage, then minimize total_score.
    Fallback: greedy with score-per-task normalization and bundle priority.
    """
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    # Parse candidates
    candidates = []
    all_tasks = set()
    all_couriers = set()
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
        task_ids = tuple(t.strip() for t in task_id_list_str.split(","))
        for t in task_ids:
            all_tasks.add(t)
        all_couriers.add(courier_id.strip())
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness, task_ids))

    if not candidates:
        return []

    # Build index structures
    task_to_cands = defaultdict(list)
    courier_to_cands = defaultdict(list)
    for i, (score, tils, cid, will, tids) in enumerate(candidates):
        for t in tids:
            task_to_cands[t].append(i)
        courier_to_cands[cid].append(i)

    # Try ILP via PuLP
    try:
        import pulp

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(candidates))]

        # Objective: maximize coverage (weight bundles slightly higher to break ties)
        prob1 += pulp.lpSum(
            x[i] * len(candidates[i][4]) for i in range(len(candidates))
        )

        # Each courier at most once
        for cid, cands in courier_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        # Each task at most once
        for t, cands in task_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=8)
        prob1.solve(solver)

        if prob1.status == 1:  # Optimal
            max_coverage = int(round(pulp.value(prob1.objective)))

            # Phase 2: Minimize score with coverage constraint
            prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
            x2 = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(len(candidates))]

            prob2 += pulp.lpSum(x2[i] * candidates[i][0] for i in range(len(candidates)))

            prob2 += (
                pulp.lpSum(x2[i] * len(candidates[i][4]) for i in range(len(candidates)))
                >= max_coverage
            )

            for cid, cands in courier_to_cands.items():
                prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

            for t, cands in task_to_cands.items():
                prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

            prob2.solve(solver)

            if prob2.status in (1, -1):  # Optimal or feasible
                result = []
                for i in range(len(candidates)):
                    if pulp.value(x2[i]) and pulp.value(x2[i]) > 0.5:
                        result.append((candidates[i][1], [candidates[i][2]]))
                if result:
                    return result
    except Exception:
        pass

    # Fallback: Greedy with score-per-task normalization and bundle priority
    # Sort by score per task (ascending), with bundle bonus
    def cand_key(item):
        score, tils, cid, will, tids = item
        n_tasks = len(tids)
        # Prefer bundles (more tasks per courier), then lower score per task
        # Also factor in willingness (higher is better)
        score_per_task = score / n_tasks if n_tasks > 0 else float('inf')
        # Bundle bonus: prefer 2-task bundles slightly
        bundle_bonus = -0.001 * n_tasks
        # Willingness bonus
        will_bonus = -0.0001 * will
        return (score_per_task + bundle_bonus + will_bonus, score)

    candidates_sorted = sorted(candidates, key=cand_key)

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, tils, cid, will, tids in candidates_sorted:
        if cid in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in tids):
            continue
        assigned_couriers.add(cid)
        for t in tids:
            assigned_tasks.add(t)
        result.append((tils, [cid]))

    # If coverage is low, try a second pass with relaxed scoring
    # Check if many tasks are uncovered
    uncovered = all_tasks - assigned_tasks
    if len(uncovered) > 0:
        # Try to fill gaps with remaining candidates
        remaining = []
        for score, tils, cid, will, tids in candidates:
            if cid in assigned_couriers:
                continue
            # Only include if at least one task is uncovered
            uncovered_tasks = [t for t in tids if t not in assigned_tasks]
            if uncovered_tasks:
                remaining.append((score, tils, cid, will, tids, uncovered_tasks))

        # Sort remaining by score per uncovered task
        remaining.sort(key=lambda x: x[0] / len(x[5]) if len(x[5]) > 0 else float('inf'))

        for score, tils, cid, will, tids, unc_tasks in remaining:
            if cid in assigned_couriers:
                continue
            # Check if any task is already assigned (skip those)
            if any(t in assigned_tasks for t in tids):
                continue
            assigned_couriers.add(cid)
            for t in tids:
                assigned_tasks.add(t)
            result.append((tils, [cid]))

    return result