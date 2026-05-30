# ALGORITHM: ILP via PuLP (CBC) for optimal coverage-first, then min-score; fallback to greedy

```python
# ALGORITHM: ILP via PuLP (CBC) for optimal coverage-first, then min-score; fallback to greedy

import heapq
from collections import defaultdict


def solve(input_text: str) -> list:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0

    # Parse candidates
    candidates = []  # (score, task_id_list_str, courier_id, willingness)
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

    # Build index structures
    # Each candidate is an option: covers some tasks, uses one courier, has a score
    n = len(candidates)
    task_sets = []
    courier_ids = []
    scores = []
    task_id_strs = []

    for i, (score, task_id_list_str, cid, w) in enumerate(candidates):
        task_ids = [t.strip() for t in task_id_list_str.split(",")]
        task_sets.append(set(task_ids))
        courier_ids.append(cid)
        scores.append(score)
        task_id_strs.append(task_id_list_str)

    # Collect all tasks and couriers
    all_tasks = set()
    for ts in task_sets:
        all_tasks.update(ts)
    all_couriers = list(set(courier_ids))

    # Try ILP approach
    try:
        import pulp

        # Phase 1: Maximize coverage
        prob1 = pulp.LpProblem("MaxCoverage", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        # Objective: maximize number of covered tasks (each task counted once)
        # For each task, create a variable indicating if it's covered
        task_cover = {}
        for t in all_tasks:
            task_cover[t] = pulp.LpVariable(f"cover_{t}", cat="Binary")

        # Objective: maximize sum of task_cover
        prob1 += pulp.lpSum(task_cover[t] for t in all_tasks)

        # Constraints:
        # Each courier used at most once
        courier_to_cands = defaultdict(list)
        for i, cid in enumerate(courier_ids):
            courier_to_cands[cid].append(i)
        for cid, cands in courier_to_cands.items():
            prob1 += pulp.lpSum(x[i] for i in cands) <= 1

        # Task cover constraints: task_cover[t] <= sum of x[i] for candidates covering t
        task_to_cands = defaultdict(list)
        for i, ts in enumerate(task_sets):
            for t in ts:
                task_to_cands[t].append(i)
        for t in all_tasks:
            prob1 += task_cover[t] <= pulp.lpSum(x[i] for i in task_to_cands[t])

        # Solve phase 1
        prob1.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))

        max_coverage = sum(pulp.value(task_cover[t]) for t in all_tasks)
        max_coverage = int(round(max_coverage))

        # Phase 2: Minimize score with coverage = max_coverage
        prob2 = pulp.LpProblem("MinScore", pulp.LpMinimize)
        x2 = [pulp.LpVariable(f"x2_{i}", cat="Binary") for i in range(n)]

        # Objective: minimize total score
        prob2 += pulp.lpSum(scores[i] * x2[i] for i in range(n))

        # Constraint: coverage must equal max_coverage
        task_cover2 = {}
        for t in all_tasks:
            task_cover2[t] = pulp.LpVariable(f"tc2_{t}", cat="Binary")
        prob2 += pulp.lpSum(task_cover2[t] for t in all_tasks) == max_coverage

        # Courier constraints
        for cid, cands in courier_to_cands.items():
            prob2 += pulp.lpSum(x2[i] for i in cands) <= 1

        # Task cover constraints
        for t in all_tasks:
            prob2 += task_cover2[t] <= pulp.lpSum(x2[i] for i in task_to_cands[t])

        prob2.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))

        result = []
        for i in range(n):
            if pulp.value(x2[i]) and pulp.value(x2[i]) > 0.5:
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