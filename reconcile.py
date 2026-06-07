"""
reconcile.py — learn the local-suite -> real-judge mapping.
===========================================================
The synthetic suite's average predicted score is a PROXY for the real 10-case
avg_score, but not equal to it (synthetic cases != hidden real cases). Each
submission gives a (suite_predicted, real_avg) pair; we fit real ~= a*pred + b
so the submission gate can judge candidates in REAL units and stop wasting
tries on solvers that only look better locally.

Pairs are obtained by RE-RUNNING each submitted solver on the CURRENT suite
(consistent pred values), cached by code hash so each solver is run once.
"""
import os
import json
import time

import archive
import judge_adapter as J

HERE = os.path.dirname(__file__)
RECON_PATH = os.path.join(HERE, "recon.json")
CACHE_PATH = os.path.join(HERE, "recon_cache.json")


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def gather_pairs(verbose=False):
    """(suite_predicted, real) for every submitted+ok solver with archived code.
    Cached by hash; only new solvers are actually re-run."""
    recs = archive.load_history()
    cache = _load_json(CACHE_PATH, {})
    pairs, seen, ran = [], set(), 0
    for r in recs:
        if not (r.get("submitted") and r.get("ok") and r.get("score") is not None):
            continue
        if r.get("manual"):           # exclude human-injected solvers from the fit
            continue
        h = r.get("hash")
        if not h or h in seen:
            continue
        seen.add(h)
        real = r["score"]
        c = cache.get(h)
        if c and abs(c.get("real", -1) - real) < 1e-6 and c.get("pred") is not None:
            pairs.append((c["pred"], real))
            continue
        code = archive.load_solver(h)
        if not code:
            continue
        # NOTE: do NOT wrap with the greedy fallback here. A pair is only
        # meaningful if the solver runs its REAL algorithm locally (same as the
        # judge). Solvers that crash locally (old buggy ILP on some suite case)
        # are skipped on purpose — wrapping them would make them fall back to
        # greedy, so suite_pred would not match their real ILP score (noise).
        try:
            ok, info = J.run_local_gate(code)
            pred = info.get("stats", {}).get("predicted_score")
        except Exception:
            pred = None
        ran += 1
        cache[h] = {"pred": pred, "real": real}
        if pred is not None:
            pairs.append((pred, real))
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    if verbose:
        print(f"  [reconcile] {len(pairs)} pairs ({ran} newly run)")
    return pairs


def fit_recon(pairs):
    """Least-squares real = a*pred + b, with rmse and R^2."""
    n = len(pairs)
    if n < 2:
        return None
    sx = sum(p for p, _ in pairs); sy = sum(r for _, r in pairs)
    sxx = sum(p * p for p, _ in pairs); sxy = sum(p * r for p, r in pairs)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9:
        return None
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    resid = [a * p + b - r for p, r in pairs]
    rmse = (sum(e * e for e in resid) / n) ** 0.5
    ybar = sy / n
    sstot = sum((r - ybar) ** 2 for _, r in pairs) or 1e-9
    r2 = 1.0 - sum(e * e for e in resid) / sstot
    return {"a": a, "b": b, "n": n, "rmse": rmse, "r2": r2, "ts": time.time()}


def _ranks(xs):
    """Average ranks (1-based), ties get the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def rank_fidelity(pairs=None):
    """Spearman rank-correlation between local suite_pred and real across the
    submitted solvers. This is the metric that matters for the gate: does local
    RANK solvers the same way the real judge does? Returns (rho, n)."""
    if pairs is None:
        pairs = gather_pairs(verbose=False)
    n = len(pairs)
    if n < 3:
        return None, n
    rp = _ranks([p for p, _ in pairs])
    rr = _ranks([r for _, r in pairs])
    mp = sum(rp) / n; mr = sum(rr) / n
    cov = sum((a - mp) * (b - mr) for a, b in zip(rp, rr))
    vp = sum((a - mp) ** 2 for a in rp) ** 0.5
    vr = sum((b - mr) ** 2 for b in rr) ** 0.5
    if vp == 0 or vr == 0:
        return None, n
    return cov / (vp * vr), n


def save_model(m):
    with open(RECON_PATH, "w") as f:
        json.dump(m, f, indent=2)


def load_model():
    return _load_json(RECON_PATH, None)


def estimate_real(suite_pred, model=None):
    """Map a suite-avg predicted score to estimated real avg_score."""
    model = model or load_model()
    if not model or suite_pred is None:
        return None
    return model["a"] * suite_pred + model["b"]


def is_reliable(model=None, min_n=4, min_r2=0.5):
    model = model or load_model()
    return bool(model and model["n"] >= min_n and model["r2"] >= min_r2)


def recompute(verbose=True):
    pairs = gather_pairs(verbose=verbose)
    model = fit_recon(pairs)
    if model:
        rho, rn = rank_fidelity(pairs)
        model["rank_rho"] = rho
        model["rank_n"] = rn
        save_model(model)
        if verbose:
            print(f"  [reconcile] real ~= {model['a']:.3f}*pred + {model['b']:.1f}  "
                  f"(n={model['n']} rmse={model['rmse']:.1f} R2={model['r2']:.2f} "
                  f"reliable={is_reliable(model)})")
            if rho is not None:
                print(f"  [reconcile] local<->real RANK fidelity: rho={rho:.3f} "
                      f"over {rn} solvers (1.0 = local ranks exactly like real)")
    elif verbose:
        print("  [reconcile] not enough pairs to fit yet.")
    return model


if __name__ == "__main__":
    recompute()
