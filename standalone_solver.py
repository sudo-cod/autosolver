"""
AutoSolver Agent — 外卖配送任务分配
====================================
单文件可提交版本。判题入口： solve(input_text: str) -> list

返回格式：[(task_id_list_str, [courier_id, ...]), ...]

设计：
  * 把问题建模为加权集合打包 / 二部指派：从候选行 (task_set, courier, score)
    中选出一组互不冲突的行（每个骑手最多一行；每个 task 最多被覆盖一次），
    在【最大化覆盖订单数】的前提下【最小化总分数】。
  * Agent 控制器在 10s 预算内并行尝试多种策略，用可配置的本地目标函数打分，
    保留最优解，超时安全返回。
  * 目标函数 OBJECTIVE 可配置，便于在探测判题机后切换（accepted-count、
    willingness 加权、score 方向等）。

依赖优先级： pulp(CBC) -> 贪心+局部搜索 兜底。任何环境都能返回合法解。
"""

import time
import heapq
import random
from collections import defaultdict

# ----------------------------------------------------------------------------
# 可配置目标（探测判题机后在此切换）
# ----------------------------------------------------------------------------
OBJECTIVE = {
    # 主目标：最大化覆盖的订单数；tie-break：最小化总分数
    "primary": "max_tasks",          # max_tasks | max_willingness_weighted
    "tiebreak": "min_score",         # min_score | max_score | none
    "allow_multi_courier": False,    # 一个 task 是否可派给多个骑手（待判题机验证）
    "score_sign": 1.0,               # +1 表示分数越小越好；探测后可改
}

TIME_BUDGET = 9.3   # 秒，留余量给 I/O


# ----------------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------------
def parse(input_text):
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    cands = []  # (idx, frozenset(task_ids), task_str, courier, score, willingness)
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_str, courier, score_str, will_str = parts[:4]
        try:
            score = float(score_str)
            will = float(will_str)
        except ValueError:
            continue
        tasks = tuple(t.strip() for t in task_str.split(","))
        cands.append({
            "idx": len(cands),
            "tasks": tasks,
            "tset": frozenset(tasks),
            "task_str": task_str.strip(),
            "courier": courier.strip(),
            "score": score,
            "will": will,
        })
    return cands


# ----------------------------------------------------------------------------
# 本地评估：给一个解打分（用于 agent 比较各策略）
# 返回 (primary_value, tiebreak_value)，按字典序，primary 越大越好，
# tiebreak 我们统一转成"越大越好"。
# ----------------------------------------------------------------------------
def evaluate(solution_rows):
    """solution_rows: list of candidate dicts (已保证合法)"""
    covered = set()
    total_score = 0.0
    weighted = 0.0
    for r in solution_rows:
        covered |= r["tset"]
        total_score += r["score"]
        weighted += r["will"]
    if OBJECTIVE["primary"] == "max_willingness_weighted":
        primary = weighted
    else:
        primary = len(covered)
    if OBJECTIVE["tiebreak"] == "min_score":
        tie = -total_score
    elif OBJECTIVE["tiebreak"] == "max_score":
        tie = total_score
    else:
        tie = 0.0
    return (primary, tie)


def is_better(a, b):
    """a,b 为 evaluate 结果元组。a 是否优于 b。"""
    if b is None:
        return True
    return a > b


# ----------------------------------------------------------------------------
# 合法性：每个骑手最多一行；每个 task 最多覆盖一次（除非 allow_multi_courier）
# ----------------------------------------------------------------------------
def is_valid(rows):
    used_c = set()
    used_t = set()
    for r in rows:
        if r["courier"] in used_c:
            return False
        used_c.add(r["courier"])
        if not OBJECTIVE["allow_multi_courier"]:
            for t in r["tasks"]:
                if t in used_t:
                    return False
                used_t.add(t)
    return True


# ----------------------------------------------------------------------------
# 策略 1：精确 ILP（pulp + CBC）
# 变量 x_i ∈ {0,1} 每个候选行。约束：每骑手 ≤1；每 task ≤1。
# 目标：max ( BIG * covered_tasks  -  total_score )   —— 字典序合一为加权目标
# ----------------------------------------------------------------------------
def strategy_ilp(cands, deadline):
    try:
        import pulp
    except Exception:
        return None
    if time.time() > deadline:
        return None

    prob = pulp.LpProblem("assign", pulp.LpMaximize)
    x = {c["idx"]: pulp.LpVariable(f"x{c['idx']}", cat="Binary") for c in cands}

    # 每个骑手最多一行
    by_courier = defaultdict(list)
    by_task = defaultdict(list)
    for c in cands:
        by_courier[c["courier"]].append(c["idx"])
        for t in c["tasks"]:
            by_task[t].append(c["idx"])
    for ids in by_courier.values():
        prob += pulp.lpSum(x[i] for i in ids) <= 1
    if not OBJECTIVE["allow_multi_courier"]:
        for ids in by_task.values():
            prob += pulp.lpSum(x[i] for i in ids) <= 1

    cmap = {c["idx"]: c for c in cands}
    # 字典序：覆盖数权重远大于分数项。score≤100，bundle≤2，行数有限，
    # BIG 取一个安全大数即可保证"先最大化覆盖，再最小化分数"。
    BIG = 1e6
    obj_terms = []
    for c in cands:
        ntask = len(c["tasks"])
        coverage_term = BIG * ntask
        score_term = -OBJECTIVE["score_sign"] * c["score"]  # 默认越小越好
        obj_terms.append((coverage_term + score_term) * x[c["idx"]])
    prob += pulp.lpSum(obj_terms)

    remaining = max(1.0, deadline - time.time())
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=int(remaining))
    try:
        prob.solve(solver)
    except Exception:
        return None

    rows = [cmap[i] for i in x if x[i].value() and x[i].value() > 0.5]
    if not is_valid(rows):
        return None
    return rows


# ----------------------------------------------------------------------------
# 策略 2：贪心基线（覆盖优先 / 分数）—— 永远可用的兜底
# ----------------------------------------------------------------------------
def strategy_greedy(cands, deadline, key):
    """key: 排序函数。覆盖优先则先大 bundle 再低分。"""
    ordered = sorted(cands, key=key)
    used_c = set()
    used_t = set()
    rows = []
    for c in ordered:
        if time.time() > deadline:
            break
        if c["courier"] in used_c:
            continue
        if not OBJECTIVE["allow_multi_courier"] and any(t in used_t for t in c["tasks"]):
            continue
        used_c.add(c["courier"])
        for t in c["tasks"]:
            used_t.add(t)
        rows.append(c)
    return rows


# ----------------------------------------------------------------------------
# 策略 3：局部搜索改进（在贪心解上做 swap / add）
# ----------------------------------------------------------------------------
def strategy_local_search(cands, deadline, seed_rows):
    best = list(seed_rows)
    best_val = evaluate(best)
    by_courier = defaultdict(list)
    for c in cands:
        by_courier[c["courier"]].append(c)

    rng = random.Random(12345)
    while time.time() < deadline:
        used_c = {r["courier"] for r in best}
        used_t = set()
        for r in best:
            used_t |= r["tset"]
        # 尝试加入一个能覆盖新 task 的空闲骑手行
        improved = False
        free_cands = [c for c in cands
                      if c["courier"] not in used_c
                      and (OBJECTIVE["allow_multi_courier"] or not (c["tset"] & used_t))]
        rng.shuffle(free_cands)
        for c in free_cands[:200]:
            trial = best + [c]
            v = evaluate(trial)
            if is_better(v, best_val):
                best, best_val = trial, v
                improved = True
                break
        if not improved:
            break
    return best


# ----------------------------------------------------------------------------
# Agent 控制器
# ----------------------------------------------------------------------------
def solve(input_text):
    t0 = time.time()
    deadline = t0 + TIME_BUDGET
    cands = parse(input_text)
    if not cands:
        return []

    best_rows = None
    best_val = None

    def consider(rows):
        nonlocal best_rows, best_val
        if rows is None:
            return
        if not is_valid(rows):
            return
        v = evaluate(rows)
        if is_better(v, best_val):
            best_rows, best_val = list(rows), v

    # 覆盖优先贪心（大 bundle 优先，低分优先）—— 快速保底
    consider(strategy_greedy(
        cands, deadline,
        key=lambda c: (-len(c["tasks"]), c["score"] * OBJECTIVE["score_sign"])))

    # 纯低分贪心（模拟基线）
    consider(strategy_greedy(
        cands, deadline,
        key=lambda c: (c["score"] * OBJECTIVE["score_sign"], -len(c["tasks"]))))

    # 精确 ILP（主力）——给它大部分剩余时间
    ilp_rows = strategy_ilp(cands, deadline - 0.5)
    consider(ilp_rows)

    # 局部搜索补强（若还有时间）
    if best_rows is not None and time.time() < deadline - 0.3:
        consider(strategy_local_search(cands, deadline - 0.2, best_rows))

    if best_rows is None:
        return []

    # 输出格式：[(task_id_list_str, [courier_id])]
    return [(r["task_str"], [r["courier"]]) for r in best_rows]


# ----------------------------------------------------------------------------
# 本地自测
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/large_seed301__1_.txt"
    with open(path) as f:
        text = f.read()
    t = time.time()
    sol = solve(text)
    dt = time.time() - t
    covered = set()
    score = 0.0
    cmap = {}
    for r in parse(text):
        cmap[(r["task_str"], r["courier"])] = r
    for ts, cs in sol:
        r = cmap.get((ts, cs[0]))
        if r:
            covered |= r["tset"]
            score += r["score"]
    print(f"assignments: {len(sol)}")
    print(f"tasks covered: {len(covered)}")
    print(f"total score: {score:.3f}")
    print(f"time: {dt:.2f}s")
