# ALGORITHM: ILP via PuLP for optimal coverage-first, then min-score; fallback to greedy with bundle-aware scoring

import sys
from collections import defaultdict


def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    # Parse candidates
    candidates = []  # (score, task_id_list_str, courier_id, willingness, task_ids_list)
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
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        candidates.append((score, task_id_list_str.strip(), courier_id.strip(), willingness, task_ids))
        all_tasks.update(task_ids)
        all_couriers.add(courier_id.strip())

    if not candidates:
        return []

    # Try ILP approach first
    try:
        import pulp

        # Build ILP: maximize coverage, then minimize score
        # Two-phase approach:
        # Phase 1: Maximize coverage
        # Phase 2: Minimize score with coverage fixed at optimal

        n = len(candidates)
        task_list = sorted(all_tasks)
        courier_list = sorted(all_couriers)
        task_idx = {t: i for i, t in enumerate(task_list)}
        courier_idx = {c: i for i, c in enumerate(courier_list)}
        num_tasks = len(task_list)
        num_couriers = len(courier_list)

        # For each candidate, which tasks and courier
        cand_tasks = []
        cand_courier = []
        cand_score = []
        for score, tstr, cid, will, tids in candidates:
            cand_tasks.append([task_idx[t] for t in tids])
            cand_courier.append(courier_idx[cid])
            cand_score.append(score)

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

        # Objective: maximize number of tasks covered (each task counts once)
        # Use a large weight for coverage
        BIG = 1000000
        prob1 += pulp.lpSum(BIG * x[i] for i in range(n) for _ in cand_tasks[i])  # approximate

        # Actually, we need to count each task at most once
        # Better: maximize sum of covered tasks
        y = [pulp.LpVariable(f"y_{j}", cat='Binary') for j in range(num_tasks)]
        prob1 += pulp.lpSum(y[j] for j in range(num_tasks))

        # y_j <= sum of x_i for candidates covering task j
        task_cands = defaultdict(list)
        for i in range(n):
            for t in cand_tasks[i]:
                task_cands[t].append(i)

        for j in range(num_tasks):
            if task_cands[j]:
                prob1 += y[j] <= pulp.lpSum(x[i] for i in task_cands[j])

        # Courier constraint: each courier at most once
        courier_cands = defaultdict(list)
        for i in range(n):
            courier_cands[cand_courier[i]].append(i)

        for c in range(num_couriers):
            if courier_cands[c]:
                prob1 += pulp.lpSum(x[i] for i in courier_cands[c]) <= 1

        # Solve phase 1
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
        prob1.solve(solver)

        if prob1.status == 1:  # Optimal
            max_coverage = int(round(pulp.value(prob1.objective)))
        else:
            max_coverage = -1

        if max_coverage >= 0:
            # Phase 2: Minimize score with coverage >= max_coverage
            prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
            x2 = [pulp.LpVariable(f"x2_{i}", cat='Binary') for i in range(n)]
            y2 = [pulp.LpVariable(f"y2_{j}", cat='Binary') for j in range(num_tasks)]

            # Minimize total score
            prob2 += pulp.lpSum(cand_score[i] * x2[i] for i in range(n))

            # Coverage constraint
            prob2 += pulp.lpSum(y2[j] for j in range(num_tasks)) >= max_coverage

            # y_j <= sum of x_i for candidates covering task j
            for j in range(num_tasks):
                if task_cands[j]:
                    prob2 += y2[j] <= pulp.lpSum(x2[i] for i in task_cands[j])

            # Courier constraint
            for c in range(num_couriers):
                if courier_cands[c]:
                    prob2 += pulp.lpSum(x2[i] for i in courier_cands[c]) <= 1

            solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
            prob2.solve(solver2)

            if prob2.status in (1, -1):  # Optimal or feasible
                result = []
                for i in range(n):
                    val = pulp.value(x2[i])
                    if val is not None and val > 0.5:
                        _, tstr, cid, _, _ = candidates[i]
                        result.append((tstr, [cid]))
                if result:
                    return result

    except Exception:
        pass

    # Fallback: improved greedy with better scoring
    # Score candidates by: prefer bundles (more tasks per courier), then lower score per task
    # Use a priority: maximize tasks covered, minimize score
    
    # Sort by score per task ascending, with tiebreaker for bundles (prefer more tasks)
    scored_cands = []
    for i, (score, tstr, cid, will, tids) in enumerate(candidates):
        ntasks = len(tids)
        # Prefer more tasks per courier, lower score per task
        score_per_task = score / ntasks if ntasks > 0 else float('inf')
        scored_cands.append((score_per_task, -ntasks, score, i))
    
    scored_cands.sort()

    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for _, _, _, i in scored_cands:
        score, tstr, cid, will, tids = candidates[i]
        if cid in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in tids):
            continue
        assigned_couriers.add(cid)
        for t in tids:
            assigned_tasks.add(t)
        result.append((tstr, [cid]))

    return result