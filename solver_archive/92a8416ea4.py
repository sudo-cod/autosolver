# ALGORITHM: ILP two-phase (max coverage then min score) via PuLP/CBC with smart greedy fallback

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
        except (ValueError, IndexError):
            continue
        task_ids = tuple(t.strip() for t in task_id_list_str.split(","))
        if not task_ids or not all(task_ids):
            continue
        candidates.append((score, task_id_list_str, courier_id, willingness, task_ids))

    if not candidates:
        return []

    # Collect all tasks and couriers
    all_tasks = set()
    for _, _, _, _, tids in candidates:
        all_tasks.update(tids)

    # Try ILP approach first
    try:
        import pulp
        result = _solve_ilp(candidates, all_tasks)
        if result is not None:
            return result
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: smart greedy with multiple strategies, pick best
    best_result = None
    best_coverage = -1
    best_score = float('inf')

    strategies = [
        _greedy_by_score_asc,
        _greedy_by_score_per_task,
        _greedy_by_willingness_desc,
        _greedy_by_score_per_task_bundle_first,
        _greedy_by_coverage_efficiency,
    ]

    for strategy in strategies:
        result = strategy(candidates)
        cov, sc = _evaluate(result, all_tasks)
        if cov > best_coverage or (cov == best_coverage and sc < best_score):
            best_coverage = cov
            best_score = sc
            best_result = result

    return best_result if best_result else []


def _evaluate(result, all_tasks):
    covered = set()
    total_score = 0.0
    # We don't have scores in result directly, but we can count coverage
    for task_id_list_str, _ in result:
        for t in task_id_list_str.split(","):
            covered.add(t.strip())
    return len(covered), 0.0


def _greedy_by_score_asc(candidates):
    sorted_cands = sorted(candidates, key=lambda x: x[0])
    return _greedy_pick(sorted_cands)


def _greedy_by_score_per_task(candidates):
    sorted_cands = sorted(candidates, key=lambda x: x[0] / max(len(x[4]), 1))
    return _greedy_pick(sorted_cands)


def _greedy_by_willingness_desc(candidates):
    sorted_cands = sorted(candidates, key=lambda x: -x[3])
    return _greedy_pick(sorted_cands)


def _greedy_by_score_per_task_bundle_first(candidates):
    # Prefer bundles (2+ tasks) with good score-per-task, then single tasks
    bundles = [c for c in candidates if len(c[4]) >= 2]
    singles = [c for c in candidates if len(c[4]) < 2]
    bundles.sort(key=lambda x: x[0] / len(x[4]))
    singles.sort(key=lambda x: x[0])
    return _greedy_pick(bundles + singles)


def _greedy_by_coverage_efficiency(candidates):
    # Score per task, but with a slight preference for bundles
    sorted_cands = sorted(candidates, key=lambda x: (x[0] / max(len(x[4]), 1)) - 0.001 * len(x[4]))
    return _greedy_pick(sorted_cands)


def _greedy_pick(sorted_candidates):
    assigned_couriers = set()
    assigned_tasks = set()
    result = []
    for score, task_id_list_str, courier_id, willingness, task_ids in sorted_candidates:
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))
    return result


def _solve_ilp(candidates, all_tasks):
    import pulp

    task_list = sorted(all_tasks)
    task_idx = {t: i for i, t in enumerate(task_list)}
    n_tasks = len(task_list)
    n_cands = len(candidates)

    # Build courier -> candidate indices mapping
    courier_cands = defaultdict(list)
    for i, (score, tstr, cid, will, tids) in enumerate(candidates):
        courier_cands[cid].append(i)

    # Phase 1: Maximize coverage
    prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n_cands)]
    y = [pulp.LpVariable(f"y_{j}", cat='Binary') for j in range(n_tasks)]

    # Objective: maximize coverage
    prob1 += pulp.lpSum(y[j] for j in range(n_tasks))

    # Each courier used at most once
    for cid, indices in courier_cands.items():
        prob1 += pulp.lpSum(x[i] for i in indices) <= 1

    # Task coverage linking
    for i, (score, tstr, cid, will, tids) in enumerate(candidates):
        for t in tids:
            j = task_idx[t]
            prob1 += y[j] <= pulp.lpSum(x[ii] for ii in range(n_cands) if t in candidates[ii][4])

    # Each task at most once
    for j in range(n_tasks):
        prob1 += pulp.lpSum(x[i] for i in range(n_cands) if task_list[j] in candidates[i][4]) <= 1

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
    prob1.solve(solver)

    if prob1.status != pulp.constants.LpStatusOptimal and prob1.status != pulp.constants.LpStatusNotSolved:
        return None

    max_coverage = sum(pulp.value(y[j]) for j in range(n_tasks))
    if max_coverage == 0:
        return []

    # Phase 2: Minimize score with coverage constraint
    prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
    x2 = [pulp.LpVariable(f"x2_{i}", cat='Binary') for i in range(n_cands)]

    # Objective: minimize total score
    prob2 += pulp.lpSum(candidates[i][0] * x2[i] for i in range(n_cands))

    # Must achieve max coverage
    # Re-add task variables for coverage constraint
    y2 = [pulp.LpVariable(f"y2_{j}", cat='Binary') for j in range(n_tasks)]
    prob2 += pulp.lpSum(y2[j] for j in range(n_tasks)) >= max_coverage - 0.5

    for cid, indices in courier_cands.items():
        prob2 += pulp.lpSum(x2[i] for i in indices) <= 1

    for i, (score, tstr, cid, will, tids) in enumerate(candidates):
        for t in tids:
            j = task_idx[t]
            prob2 += y2[j] <= pulp.lpSum(x2[ii] for ii in range(n_cands) if t in candidates[ii][4])

    for j in range(n_tasks):
        prob2 += pulp.lpSum(x2[i] for i in range(n_cands) if task_list[j] in candidates[i][4]) <= 1

    solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
    prob2.solve(solver2)

    if prob2.status == pulp.constants.LpStatusOptimal or prob2.status == pulp.constants.LpStatusNotSolved:
        result = []
        for i in range(n_cands):
            val = pulp.value(x2[i])
            if val is not None and val > 0.5:
                _, task_id_list_str, courier_id, _, _ = candidates[i]
                result.append((task_id_list_str, [courier_id]))
        if result:
            return result

    return None