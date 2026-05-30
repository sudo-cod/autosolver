# ALGORITHM: ILP via PuLP (CBC) for optimal coverage-first, then min-score; fallback to greedy

import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    # Parse candidates
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

    n = len(candidates)
    task_sets = []
    courier_ids = []
    scores = []
    task_id_strs = []

    for i, (score, task_id_list_str, cid, w) in enumerate(candidates):
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        task_sets.append(frozenset(task_ids))
        courier_ids.append(cid)
        scores.append(score)
        task_id_strs.append(task_id_list_str)

    all_tasks = set()
    for ts in task_sets:
        all_tasks.update(ts)

    courier_to_cands = defaultdict(list)
    for i, cid in enumerate(courier_ids):
        courier_to_cands[cid].append(i)

    task_to_cands = defaultdict(list)
    for i, ts in enumerate(task_sets):
        for t in ts:
            task_to_cands[t].append(i)

    # Try ILP approach
    try:
        import pulp

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        task_cover = {}
        for t in all_tasks:
            task_cover[t] = pulp.LpVariable(f"cover_{t}", cat="Binary")

        prob1 += pulp.lpSum(task_cover[t] for t in all_tasks)

        for cid, cands in courier_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        for t in all_tasks:
            prob1 += task_cover[t] <= pulp.lpSum(x[i] for i in task_to_cands[t])

        solver1 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
        prob1.solve(solver1)

        max_coverage = sum(pulp.value(task_cover[t]) or 0 for t in all_tasks)
        max_coverage = int(round(max_coverage))

        # Phase 2: Minimize score with coverage = max_coverage
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"x2_{i}", cat="Binary") for i in range(n)]

        prob2 += pulp.lpSum(scores[i] * x2[i] for i in range(n))

        task_cover2 = {}
        for t in all_tasks:
            task_cover2[t] = pulp.LpVariable(f"tc2_{t}", cat="Binary")
        prob2 += pulp.lpSum(task_cover2[t] for t in all_tasks) == max_coverage

        for cid, cands in courier_to_cands.items():
            prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

        for t in all_tasks:
            prob2 += task_cover2[t] <= pulp.lpSum(x2[i] for i in task_to_cands[t])

        solver2 = pulp.PULP_CBC_CMD(msg=False, timeLimit=4, threads=0)
        prob2.solve(solver2)

        result = []
        for i in range(n):
            val = pulp.value(x2[i])
            if val is not None and val > 0.5:
                result.append((task_id_strs[i], [courier_ids[i]]))
        return result

    except Exception:
        pass

    # Fallback: greedy by score ascending
    candidates.sort(key=lambda x: x[0])
    assigned_couriers = set()
    assigned_tasks = set()
    result = []

    for score, task_id_list_str, courier_id, willingness in candidates:
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        if courier_id in assigned_couriers:
            continue
        if any(t in assigned_tasks for t in task_ids):
            continue
        assigned_couriers.add(courier_id)
        for t in task_ids:
            assigned_tasks.add(t)
        result.append((task_id_list_str, [courier_id]))

    return result
```