"""
calibrate_synthetic.py — tune each synthetic case to match its REAL counterpart.
================================================================================
The synthetic suite is a proxy for the 10 hidden real cases. We make it a TIGHT
proxy: for each real case we know (from logged submissions) its task count,
courier count, and the champion solver's real per-case score. We fix the
synthetic case's task/courier counts to the real values and tune the WILLINGNESS
mean so the champion scores the SAME on the synthetic case as it did on the real
one (cost rises monotonically as willingness falls, so a 1-D bisection suffices).

Result: the local suite-avg tracks the real avg far better -> the suite->real
reconciliation rmse shrinks and the submission gate becomes accurate.
"""
import os
import tempfile

import make_synthetic as ms
import judge_adapter as J
import archive

HERE = os.path.dirname(__file__)
DATASET = os.path.join(HERE, "dataset")

# real case name -> synthetic filename stem
MAP = {
    "high_noise_seed601": "syn_high_noise",
    "large_seed302": "syn_large",
    "low_willingness_seed501": "syn_low_willing",
    "medium_seed201": "syn_medium_a",
    "medium_seed202": "syn_medium_b",
    "medium_seed203": "syn_medium_c",
    "scarce_couriers_seed401": "syn_scarce",
    "small_seed100": "syn_small",
    "tiny_seed42": "syn_tiny",
}
# stable per-archetype seed so regeneration is deterministic
SEED = {v: 1000 + i for i, v in enumerate(MAP.values())}


def real_targets():
    """From the best agent submission with detail: {real_case: (tasks, couriers,
    real_per_case_score)} and the champion code."""
    recs = archive.load_history()
    best = None
    for r in recs:
        if (r.get("submitted") and r.get("ok") and r.get("case_results")
                and not r.get("manual") and r.get("score") is not None):
            if best is None or r["score"] < best["score"]:
                best = r
    tg = {}
    for c in best["case_results"]:
        name = c["case_file"].replace(".txt", "")
        tg[name] = (c.get("total_tasks"), c.get("total_couriers"),
                    c.get("total_score", c.get("score")))
    return tg, archive.load_solver(best["hash"]) or open(
        os.path.join(HERE, "best_solver.py")).read()


def champ_score_on(case_text, champ_code):
    """Champion's predicted score on a single synthetic case."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(case_text)
        path = f.name
    try:
        ok, info = J.run_local_gate(champ_code, case_path=path)
        return info.get("stats", {}).get("predicted_score")
    finally:
        os.unlink(path)


def _bisect_w(n_tasks, n_couriers, target, champ_code, seed, s_mean, w_sd=0.18, steps=14):
    """1-D bisection on willingness mean (cost falls as willingness rises)."""
    def make(wm):
        return ms.gen_case(n_tasks=n_tasks, n_couriers=n_couriers, w_mean=wm,
                           w_sd=w_sd, s_mean=s_mean, s_sd=0.4 * s_mean,
                           bundle_mult=5, seed=seed)
    lo, hi, best = 0.01, 0.92, None
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        text = make(mid)
        sc = champ_score_on(text, champ_code)
        if sc is None:
            break
        if best is None or abs(sc - target) < abs(best[0] - target):
            best = (sc, mid, text, s_mean)
        if sc > target:
            lo = mid
        else:
            hi = mid
    return best


def calibrate_case(n_tasks, n_couriers, target, champ_code, seed):
    """2-parameter search: outer-adjust s_mean to set the cost LEVEL (the
    champion adds backups -> p~1 -> cost~expected_score~s_mean), inner-bisect
    willingness. If the willingness sweep can't reach the target, rescale s_mean
    by target/best and retry (handles very-expensive cases like low_willingness)."""
    s_mean = 29.6
    best = None
    for _ in range(4):
        cand = _bisect_w(n_tasks, n_couriers, target, champ_code, seed, s_mean)
        if cand is None:
            break
        if best is None or abs(cand[0] - target) < abs(best[0] - target):
            best = cand
        # within 3% -> good enough
        if abs(cand[0] - target) <= 0.03 * target:
            break
        # otherwise rescale the score level and retry
        s_mean = max(8.0, min(120.0, s_mean * (target / max(cand[0], 1e-6)) ** 0.7))
    return best              # (score, w_mean, case_text, s_mean)


def main():
    targets, champ_code = real_targets()
    print(f"{'case':24} {'tasks':>5} {'cour':>5} {'target':>8} {'before':>8} {'after':>8} {'w_mean':>6} {'s_mean':>6}")
    for real_name, stem in MAP.items():
        if real_name not in targets:
            continue
        nt, nc, target = targets[real_name]
        if not (nt and nc and target):
            continue
        path = os.path.join(DATASET, f"{stem}.txt")
        before = champ_score_on(open(path).read(), champ_code) if os.path.exists(path) else float("nan")
        sc, wm, text, sm = calibrate_case(nt, nc, target, champ_code, SEED[stem])
        with open(path, "w") as f:
            f.write(text)
        print(f"{stem:24} {nt:>5} {nc:>5} {target:>8.1f} {before:>8.1f} {sc:>8.1f} {wm:>6.3f} {sm:>6.1f}")


if __name__ == "__main__":
    main()
