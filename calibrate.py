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
    case whose input file exists locally. Scans project root + dataset/ + data/."""
    out = {}
    exclude = {"example_solution.txt", "README.md"}
    search_dirs = [HERE]
    for subdir in ("dataset", "data"):
        subpath = os.path.join(HERE, subdir)
        if os.path.isdir(subpath):
            search_dirs.append(subpath)
    for sdir in search_dirs:
        for fname in os.listdir(sdir):
            if fname.endswith(".txt") and fname not in exclude:
                path = os.path.join(sdir, fname)
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
def build_dataset(records=None, only_best=False, max_cost=float("inf")):
    """Mine cost points and aggregates from logged submissions.

    only_best: if True, only use the lowest-scoring (best) submission.
    max_cost: optional upper bound to drop cost points (default: off). The
              exact formula cost = w*ts + (1-w)*100*num_tasks fits ALL points,
              including failed 2-task bundles near cost≈200, so no filtering is
              needed; the bound is kept only as an escape hatch.
    """
    if records is None:
        records = archive.load_history()
    inputs = _local_input_map()

    # Filter to submitted+ok records
    ok_records = [r for r in records
                  if r.get("submitted") and r.get("ok") and r.get("score") is not None]
    if not ok_records:
        # Fall back to all submitted records (may have None scores from errors)
        ok_records = [r for r in records if r.get("submitted") and r.get("ok")]

    if only_best:
        ok_records.sort(key=lambda r: r.get("score", float("inf")))
        if ok_records:
            ok_records = [ok_records[0]]

    points = []       # (total_score, willingness, num_tasks, cost)
    aggregates = []   # dicts per (record, case) with detail

    for r in ok_records:
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
            # Per-assignment input features for this case, when we have its input
            # file — lets validate() recompute the case score FROM THE MODEL
            # (a genuine test) rather than from the judge's own costs.
            model_feats = [] if cmap is not None else None
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
                        tasks, total_score, willingness = row
                        model_feats.append((total_score, willingness, len(tasks)))
                        if cost <= max_cost:
                            points.append((total_score, willingness, len(tasks), cost))
            unassigned = None
            if assigned is not None and total is not None:
                unassigned = total - assigned
            aggregates.append({
                "case": name, "reported": reported, "sum_cost": sum_cost,
                "assigned": assigned, "total": total, "unassigned": unassigned,
                "model_feats": model_feats,
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

# Fallback single-term closed forms (the primary model is the expected-value
# form fit in _fit_expected_value; these rarely trigger). NOTE: the TRUE judge
# formula is cost = w*ts + (1-w)*100*num_tasks — see _fit_expected_value.
_EXACT_FORMS = [
    ("ts",            lambda s, w: s),
    ("ts*(1-w)",      lambda s, w: s * (1.0 - w)),
    ("ts/w",          lambda s, w: s / w if w else None),
    ("ts*(1-w)/w",    lambda s, w: s * (1.0 - w) / w if w else None),
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

    def predict_cost(self, total_score, willingness, num_tasks=1):
        if self.form == "expected_value":
            # cost = w*ts + (1-w)*P*num_tasks  (exact judge formula; convex
            # combination, so it never extrapolates to negative/garbage values).
            P = self.params["P"]
            return (willingness * total_score
                    + (1.0 - willingness) * P * num_tasks)
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


def _fit_expected_value(points):
    """Fit cost = w*ts + (1-w)*P*num_tasks for the single parameter P (the
    per-task failure penalty). Least squares:
        P = Σ (1-w)*nt*(cost - w*ts) / Σ ((1-w)*nt)^2
    Robust by construction: a convex combination that never extrapolates to
    negative/garbage values for valid (ts, w∈[0,1], nt≥1)."""
    num = den = 0.0
    for ts, w, nt, cost in points:
        base = (1.0 - w) * nt
        num += base * (cost - w * ts)
        den += base * base
    if den <= 0:
        return None
    P = num / den
    model = CostModel("expected_value", {"P": P}, n_points=len(points))
    max_err = max(abs(model.predict_cost(ts, w, nt) - cost)
                  for ts, w, nt, cost in points)
    model.max_resid = max_err
    return model


def fit_cost_model(points):
    """Fit the exact expected-value form first; fall back to closed forms /
    linear least-squares only if it does not match."""
    if not points:
        return None

    # 1) PRIMARY: exact expected-value form cost = w*ts + (1-w)*P*num_tasks
    ev = _fit_expected_value(points)
    if ev is not None and ev.max_resid <= COST_TOL * 100:  # ≤0.1 abs error
        ev.calibrated = True
        return ev

    # 2) exact single-term forms (no num_tasks dependence)
    for name, fn in _EXACT_FORMS:
        ok = True
        max_err = 0.0
        for s, w, nt, cost in points:
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

    # 3) last resort: linear least-squares over the basis (NOT robust to
    #    extrapolation — only used if the exact forms somehow fail)
    basis = _BASIS
    A = [[fn(s, w) for _, fn in basis] for s, w, _, _ in points]
    b = [cost for _, _, _, cost in points]
    coef = _solve_normal_equations(A, b)
    if coef is None:
        return ev  # return the expected-value fit even if slightly off
    params = {name: coef[i] for i, (name, _) in enumerate(basis)}
    max_err = max(
        abs(sum(coef[i] * fn(s, w) for i, (_, fn) in enumerate(basis)) - cost)
        for s, w, _, cost in points)
    linear = CostModel("linear", params, max_resid=max_err,
                       calibrated=(max_err <= COST_TOL * 100),
                       n_points=len(points))
    # Prefer whichever fits better.
    if ev is not None and ev.max_resid <= max_err:
        ev.calibrated = ev.max_resid <= COST_TOL * 100
        return ev
    return linear


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
    """Compare predicted vs actual. Returns a report dict.

    Per-case validation is GENUINE only for cases with a local input file
    (model_feats present): there we recompute each assignment's cost from input
    features via the model and rebuild the case score — so the error reflects
    real model accuracy. For cases without local input we can only sum the
    judge's own costs (an identity → ~0), flagged genuine=False.
    """
    # per-assignment residual (cost points) — always a genuine model test
    cost_max_err = max((abs(model.predict_cost(s, w, nt) - cost)
                        for s, w, nt, cost in points), default=0.0)

    case_rows = []
    genuine_max_err = 0.0          # max err over genuine (model-based) cases
    agg_max_err = 0.0             # max err over aggregation-only cases
    have_genuine = False
    for a in aggregates:
        rep, sc, u = a.get("reported"), a.get("sum_cost"), a.get("unassigned") or 0
        if rep is None:
            continue
        feats = a.get("model_feats")
        if feats is not None:
            predicted = sum(model.predict_cost(ts, w, nt) for ts, w, nt in feats) \
                + model.penalty_per_task * u
            genuine = True
            have_genuine = True
        else:
            predicted = sc + model.penalty_per_task * u   # tautological identity
            genuine = False
        err = abs(predicted - rep)
        if genuine:
            genuine_max_err = max(genuine_max_err, err)
        else:
            agg_max_err = max(agg_max_err, err)
        case_rows.append({"case": a["case"], "reported": rep,
                          "predicted": predicted, "err": err,
                          "unassigned": u, "genuine": genuine})

    # The headline per-case error is the GENUINE one when available.
    case_max_err = genuine_max_err if have_genuine else agg_max_err
    calibrated = cost_max_err <= COST_TOL * 100 and case_max_err <= CASE_TOL
    return {"cost_max_err": cost_max_err, "case_max_err": case_max_err,
            "calibrated": calibrated, "cases": case_rows,
            "n_points": len(points), "have_genuine": have_genuine}


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


def recalibrate(records=None, verbose=True, only_best=True, max_cost=float("inf")):
    """Mine logged detail, fit cost() + penalty, validate, persist. Returns report.

    only_best: only use the best-scoring submission for calibration (default True
               to avoid distortion from old bad solvers).
    max_cost: optional cost upper bound (default off — the exact formula fits all
              points, so no outlier filtering is needed).
    """
    points, aggregates = build_dataset(records, only_best=only_best, max_cost=max_cost)
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
    kind = "model-based" if report.get("have_genuine") else "aggregation-only"
    print(f"  [calibrate] per-assignment max err={report['cost_max_err']:.4g}  "
          f"per-case max err={report['case_max_err']:.4g} ({kind})  "
          f"=> calibrated={report['calibrated']}")
    print(f"  [calibrate] per-case is a genuine model test only for cases with a "
          f"local input file; others are tagged (agg-only).")
    for row in sorted(report["cases"], key=lambda r: r["err"], reverse=True)[:10]:
        if not row.get("genuine"):
            tag = "  (agg-only, tautological)"
        elif row["err"] > CASE_TOL:
            tag = "  <-- MISMATCH"
        else:
            tag = ""
        print(f"      {row['case']}: predicted={row['predicted']:.2f} "
              f"actual={row['reported']:.2f} err={row['err']:.3g}{tag}")


if __name__ == "__main__":
    rep = recalibrate()
    if rep is None:
        print("Nothing to calibrate.")
