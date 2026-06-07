# ALGORITHM: greedy primary (cost) + bundle uncovered tasks + spare-courier backups on riskiest
from collections import defaultdict


def solve(input_text: str) -> list:
    lines = input_text.strip().split("\n")
    singles = defaultdict(list)          # task -> [(cost, c, s, w)]
    bundles = defaultdict(list)          # "t1,t2" -> [(cost, c, s, w)]
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) < 4:
            continue
        ts, c = p[0].strip(), p[1].strip()
        try:
            s = float(p[2]); w = float(p[3])
        except ValueError:
            continue
        nt = ts.count(",") + 1
        cost = w * s + (1.0 - w) * 100.0 * nt
        (singles if nt == 1 else bundles)[ts].append((cost, c, s, w))
    for d in (singles, bundles):
        for k in d:
            d[k].sort()

    used = set()                          # couriers used
    chosen = {}                           # task_str -> [courier,...] (primary first)
    all_tasks = set(singles.keys())
    n_couriers = len({c for lst in singles.values() for _, c, _, _ in lst})

    # Build, per uncovered task, its cheapest available single; and a quick way
    # to find a cheap bundle covering two uncovered tasks.
    # bundles_by_pair: frozenset({a,b}) -> sorted [(cost,c,..)]
    bundle_index = {}
    for ts, lst in bundles.items():
        a, b = ts.split(",")
        bundle_index[ts] = (a, b, lst)

    # Phase 1+2: COVERAGE FIRST. While tasks remain uncovered and couriers are
    # free, cover them — preferring 2-for-1 bundles when couriers are scarce
    # relative to the remaining uncovered tasks.
    uncov = set(all_tasks)
    avail = lambda lst: next((row for row in lst if row[1] not in used), None)
    while uncov:
        free = n_couriers - len(used)
        if free <= 0:
            break
        use_bundle = free < len(uncov)    # scarce -> cover 2 per courier
        placed = False
        if use_bundle:
            best = None
            for ts, (a, b, lst) in bundle_index.items():
                if a in uncov and b in uncov:
                    row = avail(lst)
                    if row and (best is None or row[0] < best[0]):
                        best = (row[0], ts, a, b, row[1])
            if best:
                _, ts, a, b, c = best
                used.add(c); chosen[ts] = [c]; uncov.discard(a); uncov.discard(b)
                placed = True
        if not placed:                    # cover one task with its cheapest single
            # pick the uncovered task whose cheapest available single is cheapest
            bestt = None
            for t in uncov:
                row = avail(singles[t])
                if row and (bestt is None or row[0] < bestt[0]):
                    bestt = (row[0], t, row[1])
            if bestt is None:
                break
            _, t, c = bestt
            used.add(c); chosen[t] = [c]; uncov.discard(t)

    # Phase 3: spend spare couriers as BACKUPS on the riskiest covered single
    # tasks (lowest current p_complete), highest-willingness courier first.
    def p_of(task):
        pf = 1.0
        for c in chosen[task]:
            for _, cc, s, w in singles[task]:
                if cc == c:
                    pf *= (1.0 - w); break
        return 1.0 - pf

    single_chosen = [t for t in chosen if "," not in t]
    # iterate: repeatedly add the best backup to the currently riskiest task
    improved = True
    while improved:
        improved = False
        single_chosen.sort(key=p_of)      # lowest p first
        for t in single_chosen:
            if p_of(t) > 0.995:           # already near-certain; skip
                continue
            for cost, c, s, w in singles[t]:
                if c not in used and c not in chosen[t]:
                    used.add(c); chosen[t].append(c); improved = True
                    break
            if improved:
                break

    return [(t, cs) for t, cs in chosen.items()]
