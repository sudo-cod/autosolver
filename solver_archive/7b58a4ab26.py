# ALGORITHM: ILP two-phase (OR-Tools CP-SAT + PuLP/CBC) with improved greedy fallback

import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
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

    n = len(candidates)
    
    # Build index structures
    task_to_cands = defaultdict(list)
    courier_to_cands = defaultdict(list)
    for i, (score, tils, cid, will, tids) in enumerate(candidates):
        for t in tids:
            task_to_cands[t].append(i)
        courier_to_cands[cid].append(i)

    # Try OR-Tools CP-SAT first (usually faster/more reliable than CBC)
    try:
        from ortools.sat.python import cp_model

        # Phase 1: Maximize coverage
        model1 = cp_model.CpModel()
        x = [model1.NewBoolVar(f'x{i}') for i in range(n)]

        # Maximize coverage
        model1.Maximize(sum(x[i] * len(candidates[i][4]) for i in range(n)))

        # Each courier at most once
        for cid, cands in courier_to_cands.items():
            model1.Add(sum(x[i] for i in cands) <= 1)

        # Each task at most once
        for t, cands in task_to_cands.items():
            model1.Add(sum(x[i] for i in cands) <= 1)

        solver1 = cp_model.CpSolver()
        solver1.parameters.max_time_in_seconds = 5.0
        status1 = solver1.Solve(model1)

        if status1 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            max_coverage = int(solver1.ObjectiveValue())

            # Phase 2: Minimize score with coverage constraint
            model2 = cp_model.CpModel()
            y = [model2.NewBoolVar(f'y{i}') for i in range(n)]

            # Coverage constraint
            model2.Add(sum(y[i] * len(candidates[i][4]) for i in range(n)) >= max_coverage)

            # Each courier at most once
            for cid, cands in courier_to_cands.items():
                model2.Add(sum(y[i] for i in cands) <= 1)

            # Each task at most once
            for t, cands in task_to_cands.items():
                model2.Add(sum(y[i] for i in cands) <= 1)

            # Minimize total score (scale to integer)
            SCALE = 1000
            model2.Minimize(sum(y[i] * int(candidates[i][0] * SCALE) for i in range(n)))

            solver2 = cp_model.CpSolver()
            solver2.parameters.max_time_in_seconds = 4.0
            status2 = solver2.Solve(model2)

            if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                result = []
                for i in range(n):
                    if solver2.Value(y[i]) == 1:
                        result.append((candidates[i][1], [candidates[i][2]]))
                if result:
                    return result
    except Exception:
        pass

    # Try PuLP/CBC as backup
    try:
        import pulp

        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        prob1 += pulp.lpSum(x[i] * len(candidates[i][4]) for i in range(n))

        for cid, cands in courier_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1
        for t, cands in task_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=7)
        prob1.solve(solver)

        if prob1.status == 1:
            max_coverage = int(round(pulp.value(prob1.objective)))

            prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
            x2 = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]
            prob2 += pulp.lpSum(x2[i] * candidates[i][0] for i in range(n))
            prob2 += pulp.lpSum(x2[i] * len(candidates[i][4]) for i in range(n)) >= max_coverage

            for cid, cands in courier_to_cands.items():
                prob2 += pulp.lpSum(x2[i] for i in cands) <= 1
            for t, cands in task_to_cands.items():
                prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

            prob2.solve(solver)

            if prob2.status in (1, -1):
                result = []
                for i in range(n):
                    if pulp.value(x2[i]) and pulp.value(x2[i]) > 0.5:
                        result.append((candidates[i][1], [candidates[i][2]]))
                if result:
                    return result
    except Exception:
        pass

    # Improved greedy fallback: score-per-task with willingness factor
    # Key insight: we want to minimize total_score while maximizing coverage
    # Use score per task as primary metric, with small willingness bonus
    
    def greedy_score(item):
        score, tils, cid, will, tids = item
        n_tasks = len(tids)
        if n_tasks == 0:
            return (float('inf'), 0)
        # Primary: score per task (lower is better)
        # Secondary: willingness (higher is better, so negate)
        # Tertiary: prefer bundles (more tasks = better coverage per courier)
        return (score / n_tasks, -will, -n_tasks)
    
    candidates_sorted = sorted(candidates, key=greedy_score)
    
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

    # Second pass: try to fill any remaining gaps with remaining candidates
    uncovered = all_tasks - assigned_tasks
    if uncovered:
        remaining = []
        for i, (score, tils, cid, will, tids) in enumerate(candidates):
            if cid in assigned_couriers:
                continue
            uncovered_in_cand = [t for t in tids if t not in assigned_tasks]
            if uncovered_in_cand:
                remaining.append((score / len(uncovered_in_cand), score, tils, cid, tids, i))
        
        remaining.sort()
        
        for _, score, tils, cid, tids, i in remaining:
            if cid in assigned_couriers:
                continue
            if any(t in assigned_tasks for t in tids):
                continue
            assigned_couriers.add(cid)
            for t in tids:
                assigned_tasks.add(t)
            result.append((tils, [cid]))

    return result