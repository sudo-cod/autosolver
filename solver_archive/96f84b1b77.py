# ALGORITHM: ILP two-phase (max coverage, then min score) with OR-Tools CP-SAT fallback and bundle-aware greedy

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

    # Try PuLP ILP first
    result = _try_pulp_ilp(candidates, all_tasks, all_couriers)
    if result is not None:
        return result

    # Try OR-Tools CP-SAT
    result = _try_ortools_cp(candidates, all_tasks, all_couriers)
    if result is not None:
        return result

    # Fallback: bundle-aware greedy
    return _bundle_greedy(candidates)


def _try_pulp_ilp(candidates, all_tasks, all_couriers):
    try:
        import pulp
    except ImportError:
        return None

    try:
        task_list = sorted(all_tasks)
        courier_list = sorted(all_couriers)
        n_cand = len(candidates)

        # Build index structures
        task_to_cands = defaultdict(list)
        courier_to_cands = defaultdict(list)
        for i, c in enumerate(candidates):
            for t in c[4]:
                task_to_cands[t].append(i)
            courier_to_cands[c[2]].append(i)

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n_cand)]

        for t in task_list:
            covering = task_to_cands[t]
            if covering:
                prob1 += pulp.lpSum(x[i] for i in covering) <= 1

        for c in courier_list:
            using = courier_to_cands[c]
            if using:
                prob1 += pulp.lpSum(x[i] for i in using) <= 1

        prob1 += pulp.lpSum(x[i] * len(candidates[i][4]) for i in range(n_cand))

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
        prob1.solve(solver)

        if prob1.status not in (pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved):
            return None

        max_coverage = sum(
            (pulp.value(x[i]) or 0) * len(candidates[i][4])
            for i in range(n_cand)
        )
        max_coverage = int(round(max_coverage))

        # Phase 2: Minimize score with coverage constraint
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_cand)]

        for t in task_list:
            covering = task_to_cands[t]
            if covering:
                prob2 += pulp.lpSum(x2[i] for i in covering) <= 1

        for c in courier_list:
            using = courier_to_cands[c]
            if using:
                prob2 += pulp.lpSum(x2[i] for i in using) <= 1

        prob2 += pulp.lpSum(x2[i] * len(candidates[i][4]) for i in range(n_cand)) >= max_coverage

        prob2 += pulp.lpSum(x2[i] * candidates[i][0] for i in range(n_cand))

        solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
        prob2.solve(solver2)

        result = []
        for i in range(n_cand):
            val = pulp.value(x2[i])
            if val is not None and val > 0.5:
                result.append((candidates[i][1], [candidates[i][2]]))

        return result if result else None

    except Exception:
        return None


def _try_ortools_cp(candidates, all_tasks, all_couriers):
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    try:
        task_list = sorted(all_tasks)
        courier_list = sorted(all_couriers)
        n_cand = len(candidates)

        task_to_cands = defaultdict(list)
        courier_to_cands = defaultdict(list)
        for i, c in enumerate(candidates):
            for t in c[4]:
                task_to_cands[t].append(i)
            courier_to_cands[c[2]].append(i)

        # Scale scores to integers for CP-SAT
        max_score = max(c[0] for c in candidates) if candidates else 1
        scale = 1000
        int_scores = [int(round(c[0] * scale)) for c in candidates]
        max_total = sum(int_scores) + 1

        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n_cand)]

        # Task constraints
        for t in task_list:
            covering = task_to_cands[t]
            if covering:
                model.Add(sum(x[i] for i in covering) <= 1)

        # Courier constraints
        for c in courier_list:
            using = courier_to_cands[c]
            if using:
                model.Add(sum(x[i] for i in using) <= 1)

        # Coverage variable
        coverage_terms = [x[i] * len(candidates[i][4]) for i in range(n_cand)]
        total_coverage = sum(coverage_terms)

        # Score variable
        score_terms = [x[i] * int_scores[i] for i in range(n_cand)]
        total_score = sum(score_terms)

        # Lexicographic: maximize coverage, then minimize score
        # Use weighted objective: coverage * max_total - score
        # This prioritizes coverage first, then minimizes score
        objective = total_coverage * max_total - total_score
        model.Maximize(objective)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 6.0
        solver.parameters.num_workers = 0

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = []
            for i in range(n_cand):
                if solver.Value(x[i]) == 1:
                    result.append((candidates[i][1], [candidates[i][2]]))
            return result if result else None

        return None

    except Exception:
        return None


def _bundle_greedy(candidates):
    """
    Greedy that prioritizes 2-task bundles when couriers are scarce,
    otherwise picks by best score-per-task ratio.
    """
    all_tasks = set()
    all_couriers = set()
    for _, _, cid, _, tids in candidates:
        all_couriers.add(cid)
        all_tasks.update(tids)

    n_tasks = len(all_tasks)
    n_couriers = len(all_couriers)

    courier_scarcity = n_tasks / max(n_couriers, 1)

    def sort_key(cand):
        score, _, _, _, tids = cand
        n_tids = len(tids)
        if courier_scarcity > 1.5:
            return (-n_tids, score)
        else:
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