# ALGORITHM: ILP two-phase (max coverage then min score) with OR-Tools CP-SAT fallback and greedy baseline

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

    # Try ILP with PuLP
    result = _try_pulp_ilp(candidates)
    if result is not None:
        return result

    # Try OR-Tools CP-SAT
    result = _try_ortools_ilp(candidates)
    if result is not None:
        return result

    # Fallback: greedy
    return _greedy_fallback(candidates)


def _try_pulp_ilp(candidates):
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

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=1)
        prob1.solve(solver)

        status1 = pulp.LpStatus[prob1.status]
        if status1 not in ('Optimal', 'Not Solved', 'Feasible'):
            return None

        max_coverage = int(round(sum(pulp.value(y[j]) or 0 for j in range(nt))))

        # Phase 2: Minimize score with coverage constraint
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"x2_{i}", cat='Binary') for i in range(n)]
        y2 = [pulp.LpVariable(f"y2_{j}", cat='Binary') for j in range(nt)]

        prob2 += pulp.lpSum(candidates[i][0] * x2[i] for i in range(n))
        prob2 += pulp.lpSum(y2) >= max_coverage

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

        solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=1)
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


def _try_ortools_ilp(candidates):
    try:
        from ortools.sat.python import cp_model
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

        # Scale scores to integers for CP-SAT
        scale = 1000
        int_scores = [int(round(c[0] * scale)) for c in candidates]

        # Phase 1: Maximize coverage
        model1 = cp_model.CpModel()
        x = [model1.NewBoolVar(f'x{i}') for i in range(n)]
        y = [model1.NewBoolVar(f'y{j}') for j in range(nt)]

        for cid, cands in courier_cands.items():
            model1.Add(sum(x[i] for i in cands) <= 1)

        for j, t in enumerate(task_list):
            covering = []
            for i, (_, _, _, _, tids) in enumerate(candidates):
                if t in tids:
                    covering.append(i)
            if covering:
                model1.Add(y[j] <= sum(x[i] for i in covering))
            else:
                model1.Add(y[j] == 0)

        model1.Maximize(sum(y))

        solver1 = cp_model.CpSolver()
        solver1.parameters.max_time_in_seconds = 4.0
        status1 = solver1.Solve(model1)

        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        max_coverage = sum(solver1.Value(y[j]) for j in range(nt))

        # Phase 2: Minimize score with coverage constraint
        model2 = cp_model.CpModel()
        x2 = [model2.NewBoolVar(f'x2_{i}') for i in range(n)]
        y2 = [model2.NewBoolVar(f'y2_{j}') for j in range(nt)]

        for cid, cands in courier_cands.items():
            model2.Add(sum(x2[i] for i in cands) <= 1)

        for j, t in enumerate(task_list):
            covering = []
            for i, (_, _, _, _, tids) in enumerate(candidates):
                if t in tids:
                    covering.append(i)
            if covering:
                model2.Add(y2[j] <= sum(x2[i] for i in covering))
                model2.Add(sum(x2[i] for i in covering) <= 1)
            else:
                model2.Add(y2[j] == 0)

        model2.Add(sum(y2) >= max_coverage)

        model2.Minimize(sum(int_scores[i] * x2[i] for i in range(n)))

        solver2 = cp_model.CpSolver()
        solver2.parameters.max_time_in_seconds = 4.0
        status2 = solver2.Solve(model2)

        if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        result = []
        for i in range(n):
            if solver2.Value(x2[i]) == 1:
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