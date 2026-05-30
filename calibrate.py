"""
calibrate.py — reverse-engineer the judge's cost() formula (probe → fit → validate)
===================================================================================
The judge scores each case as:  case_score = sum(per-assignment cost) + penalty * unassigned_tasks
Every assignment's `cost`, `p_complete`, `expected_score` is returned in
`case_results[].detail`. We MINE that detail to learn `cost(total_score, willingness)`,
then VALIDATE by comparing predicted vs actual per-case scores.

Data sources:
  - cost points: only cases with a LOCAL input file (large_seed301.txt) — join
    each detail entry back to its input row to recover (total_score, willingness).
  - case aggregates: ALL cases (detail already carries cost; no input needed) —
    used to learn the unassigned penalty and validate the aggregation.

No third-party deps: least-squares via normal equations (Gaussian elimination).
"""

import os
import json
import time

import judge_adapter  # for _parse_candidates (input row -> total_score, willingness)
import archive        # for load_history

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "cost_model.json")
COST_TOL = 1e-3       # per-assignment cost match tolerance (exact-form / "calibrated")
CASE_TOL = 1e-2       # per-case predicted-vs-actual tolerance


# ---------------------------------------------------------------------------
# Local input files we can join detail back to (to recover input features).
# ---------------------------------------------------------------------------
def _local_input_map():
    """case_file basename (without .txt) -> parsed candidate map, for every
    case whose input file exists locally."""
    out = {}
    for fname in os.listdir(HERE):
        if fname.endswith(".txt") and fname not in ("example_solution.txt",):
            path = os.path.join(HERE, fname)
            try:
                with open(path) as f:
                    cmap, _ = judge_adapter._parse_candidates(f.read())
                if cmap:
                    out[fname] = cmap          # keyed by exact filename, e.g. large_seed301.txt
                    out[fname[:-4]] = cmap      # and without .txt
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Build the calibration dataset from logged submissions carrying detail.
# ---------------------------------------------------------------------------
def build_dataset(records=None):
    if records is None:
        records = archive.load_history()
    inputs = _local_input_map()

    points = []       # (total_score, willingness, cost)
    aggregates = []   # dicts per (record, case) with detail

    for r in records:
        if not (r.get("submitted") and r.get("ok")):
            continue
        for c in (r.get("case_results") or []):
            detail = c.get("detail")
            if not detail:
                continue
            name = c.get("case_file", "")
            reported = c.get("score", c.get("total_score"))
            assigned = c.get("assigned", c.get("assigned_count"))
            total = c.get("total_tasks")
            sum_cost = 0.0
            cmap = inputs.get(name) or inputs.get(name[:-4] if name.endswith(".txt") else name)
            for d in detail:
                cost = d.get("cost")
                if cost is None:
                    continue
                sum_cost += cost
                if cmap is not None:
                    ts = d.get("task_id_list")
                    couriers = d.get("couriers") or []
                    if not couriers:
                        continue
                    row = cmap.get((ts, str(couriers[0])))
                    if row is not None:
                        _, total_score, willingness = row
                        points.append((total_score, willingness, cost))
            unassigned = None
            if assigned is not None and total is not None:
                unassigned = total - assigned
            aggregates.append({
                "case": name, "reported": reported, "sum_cost": sum_cost,
                "assigned": assigned, "total": total, "unassigned": unassigned,
            })
    return points, aggregates


# ---------------------------------------------------------------------------
# Cost model: a closed form OR a linear combination over a feature basis.
# ---------------------------------------------------------------------------
# Each basis term is (name, fn(total_score, willingness)). Kept linearly
# INDEPENDENT (no ts*w / ts*(1-w) pair, which sum to ts and make A^T A singular).
# Spans the discovered form cost = ts - K*w + K  via {ts, w, const}.
_BASIS = [
    ("ts",          lambda s, w: s),
    ("w",           lambda s, w: w),
    ("ts/w",        lambda s, w: s / w if w else 0.0),
    ("1/w",         lambda s, w: 1.0 / w if w else 0.0),
    ("const",       lambda s, w: 1.0),
]

# Clean single-term closed forms to test for an EXACT match first.
_EXACT_FORMS = [
    ("ts",            lambda s, w: s),
    ("ts*(1-w)",      lambda s, w: s * (1.0 - w)),
    ("ts/w",          lambda s, w: s / w if w else None),
    ("ts*(1-w)/w",    lambda s, w: s * (1.0 - w) / w if w else None),
    ("ts+90*(1-w)",   lambda s, w: s + 90.0 * (1.0 - w)),  # discovered form
    ("ts*w",          lambda s, w: s * w),
]


class CostModel:
    def __init__(self, form, params, penalty_per_task=0.0,
                 max_resid=None, calibrated=False, n_points=0):
        self.form = form              # "exact:<name>" or "linear"
        self.params = params          # exact: {} ; linear: {term: coef}
        self.penalty_per_task = penalty_per_task
        self.max_resid = max_resid
        self.calibrated = calibrated
        self.n_points = n_points

    def predict_cost(self, total_score, willingness):
        if self.form.startswith("exact:"):
            name = self.form.split(":", 1)[1]
            fn = dict(_EXACT_FORMS)[name]
            v = fn(total_score, willingness)
            return v if v is not None else 0.0
        # linear
        basis = dict(_BASIS)
        return sum(coef * basis[term](total_score, willingness)
                   for term, coef in self.params.items())

    def to_dict(self):
        return {"cost_form": self.form, "cost_params": self.params,
                "penalty_per_task": self.penalty_per_task,
                "max_resid": self.max_resid, "calibrated": self.calibrated,
                "n_points": self.n_points, "ts": time.time()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["cost_form"], d.get("cost_params", {}),
                   d.get("penalty_per_task", 0.0), d.get("max_resid"),
                   d.get("calibrated", False), d.get("n_points", 0))


def _solve_normal_equations(A, b, ridge=1e-9):
    """Least squares: solve (A^T A + ridge*I) x = A^T b via Gaussian elimination.
    A tiny ridge term guarantees invertibility against numerical collinearity."""
    n = len(A[0])
    AT = [[A[r][c] for r in range(len(A))] for c in range(n)]
    ATA = [[sum(AT[i][k] * AT[j][k] for k in range(len(A))) for j in range(n)]
           for i in range(n)]
    for i in range(n):
        ATA[i][i] += ridge
    ATb = [sum(AT[i][k] * b[k] for k in range(len(A))) for i in range(n)]
    # augment + eliminate
    M = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        M[col] = [v / pivval for v in M[col]]
        for r in range(n):
            if r != col and abs(M[r][col]) > 1e-15:
                factor = M[r][col]
                M[r] = [a - factor * b_ for a, b_ in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def fit_cost_model(points):
    """Try exact closed forms first; fall back to linear least-squares."""
    if not points:
        return None

    # 1) exact single-term forms
    for name, fn in _EXACT_FORMS:
        ok = True
        max_err = 0.0
        for s, w, cost in points:
            pred = fn(s, w)
            if pred is None:
                ok = False
                break
            max_err = max(max_err, abs(pred - cost))
            if max_err > COST_TOL:
                ok = False
                break
        if ok:
            return CostModel(f"exact:{name}", {}, max_resid=max_err,
                             calibrated=True, n_points=len(points))

    # 2) linear least-squares over the basis
    basis = _BASIS
    A = [[fn(s, w) for _, fn in basis] for s, w, _ in points]
    b = [cost for _, _, cost in points]
    coef = _solve_normal_equations(A, b)
    if coef is None:
        return None
    params = {name: coef[i] for i, (name, _) in enumerate(basis)}
    max_err = max(
        abs(sum(coef[i] * fn(s, w) for i, (_, fn) in enumerate(basis)) - cost)
        for s, w, cost in points)
    return CostModel("linear", params, max_resid=max_err,
                     calibrated=(max_err <= COST_TOL * 100),  # looser for regression
                     n_points=len(points))


def fit_penalty(aggregates):
    """penalty_per_task = median of (reported - sum_cost)/unassigned over cases
    with unassigned > 0. Cases with unassigned==0 should already match sum_cost."""
    per_task = []
    for a in aggregates:
        u = a.get("unassigned")
        rep, sc = a.get("reported"), a.get("sum_cost")
        if u and u > 0 and rep is not None:
            per_task.append((rep - sc) / u)
    if not per_task:
        return 0.0
    per_task.sort()
    mid = len(per_task) // 2
    return per_task[mid] if len(per_task) % 2 else \
        0.5 * (per_task[mid - 1] + per_task[mid])


def validate(model, aggregates, points):
    """Compare predicted vs actual. Returns a report dict."""
    # per-assignment residual (cost points)
    cost_max_err = max((abs(model.predict_cost(s, w) - cost)
                        for s, w, cost in points), default=0.0)
    # per-case predicted vs reported
    case_rows = []
    case_max_err = 0.0
    for a in aggregates:
        rep, sc, u = a.get("reported"), a.get("sum_cost"), a.get("unassigned") or 0
        if rep is None:
            continue
        predicted = sc + model.penalty_per_task * u
        err = abs(predicted - rep)
        case_max_err = max(case_max_err, err)
        case_rows.append({"case": a["case"], "reported": rep,
                          "predicted": predicted, "err": err,
                          "unassigned": u})
    calibrated = (cost_max_err <= COST_TOL and case_max_err <= CASE_TOL) or \
                 (model.form == "linear" and case_max_err <= CASE_TOL)
    return {"cost_max_err": cost_max_err, "case_max_err": case_max_err,
            "calibrated": calibrated, "cases": case_rows,
            "n_points": len(points)}


# ---------------------------------------------------------------------------
# Persistence + the top-level recalibrate() the agent calls.
# ---------------------------------------------------------------------------
def save_model(model: CostModel):
    with open(MODEL_PATH, "w") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH) as f:
        return CostModel.from_dict(json.load(f))


def recalibrate(records=None, verbose=True):
    """Mine logged detail, fit cost() + penalty, validate, persist. Returns report."""
    points, aggregates = build_dataset(records)
    if not aggregates:
        if verbose:
            print("  [calibrate] no submission detail in log yet — run a probe first.")
        return None
    model = fit_cost_model(points)
    if model is None:
        if verbose:
            if not points:
                print("  [calibrate] no cost points yet — need large_seed301 "
                      "detail (submit a solver that covers it).")
            else:
                print(f"  [calibrate] have {len(points)} points but the fit "
                      "failed (singular system).")
        return None
    model.penalty_per_task = fit_penalty(aggregates)
    report = validate(model, aggregates, points)
    model.calibrated = report["calibrated"]
    model.max_resid = report["cost_max_err"]
    save_model(model)
    if verbose:
        _print_report(model, report)
    return report


def _print_report(model, report):
    print(f"  [calibrate] cost form: {model.form}  "
          f"penalty/task={model.penalty_per_task:.4f}  "
          f"points={report['n_points']}")
    print(f"  [calibrate] per-assignment max err={report['cost_max_err']:.4g}  "
          f"per-case max err={report['case_max_err']:.4g}  "
          f"=> calibrated={report['calibrated']}")
    for row in sorted(report["cases"], key=lambda r: r["err"], reverse=True)[:10]:
        flag = "" if row["err"] <= CASE_TOL else "  <-- MISMATCH"
        print(f"      {row['case']}: predicted={row['predicted']:.2f} "
              f"actual={row['reported']:.2f} err={row['err']:.3g}{flag}")


if __name__ == "__main__":
    rep = recalibrate()
    if rep is None:
        print("Nothing to calibrate.")
