"""
make_synthetic.py — generate synthetic local cases matching the 10 archetypes.
=============================================================================
Only `large_seed301` is available as a real local case, so the agent can only
optimize that single case offline. These synthetic cases mirror the OTHER
archetypes (sizes tiny->large, high_noise, low_willingness, scarce_couriers) so
the local gate evaluates a diverse 10-case suite and the agent optimizes the
true multi-case objective. Cases are scored locally by the exact cost formula
(no judge needed).

Files are written to dataset/ with a 'syn_' PREFIX so their names NEVER collide
with real judge case names. Calibration joins real judge `detail` by case_file
name, so distinct names keep the cost-model calibration uncontaminated.

Structure mirrors large_seed301: every task offered to every courier as a
single-courier row, plus many 2-task bundle rows. willingness mean ~0.30 (low),
single scores 10-50.
"""
import os
import random

HERE = os.path.dirname(__file__)
DATASET = os.path.join(HERE, "dataset")
HEADER = "task_id_list\tcourier_id\ttotal_score\twillingness"


def _w(rng, mean, sd):
    return min(0.95, max(0.01, rng.gauss(mean, sd)))


def _s(rng, mean, sd, lo=10.0, hi=50.0):
    return min(hi, max(lo, rng.gauss(mean, sd)))


def gen_case(n_tasks, n_couriers, w_mean=0.30, w_sd=0.216,
             s_mean=29.6, s_sd=11.6, bundle_mult=5, seed=0):
    rng = random.Random(seed)
    tasks = [f"T{i:04d}" for i in range(n_tasks)]
    couriers = [f"C{j:03d}" for j in range(n_couriers)]
    lines = [HEADER]

    # singles: every task to every courier (one row each)
    for t in tasks:
        for c in couriers:
            lines.append(f"{t}\t{c}\t{round(_s(rng, s_mean, s_sd),3)}\t"
                         f"{round(_w(rng, w_mean, w_sd),4)}")

    # bundles: random distinct (task-pair, courier) rows; combined score ~2x
    pairs = [(a, b) for a in range(n_tasks) for b in range(a + 1, n_tasks)]
    if pairs:
        target = min(bundle_mult * n_tasks * n_couriers, len(pairs) * n_couriers)
        seen = set()
        attempts = 0
        while len(seen) < target and attempts < target * 4:
            attempts += 1
            a, b = rng.choice(pairs)
            c = rng.choice(couriers)
            ts = f"{tasks[a]},{tasks[b]}"
            if (ts, c) in seen:
                continue
            seen.add((ts, c))
            s = _s(rng, s_mean, s_sd) + _s(rng, s_mean, s_sd)
            lines.append(f"{ts}\t{c}\t{round(s,3)}\t{round(_w(rng, w_mean, w_sd),4)}")
    return "\n".join(lines) + "\n"


# Archetype configs (large_seed301 is the real case; these are the other 9).
CASES = [
    ("syn_tiny",        dict(n_tasks=6,  n_couriers=14, seed=42)),
    ("syn_small",       dict(n_tasks=15, n_couriers=32, seed=100)),
    ("syn_medium_a",    dict(n_tasks=30, n_couriers=60, seed=201)),
    ("syn_medium_b",    dict(n_tasks=30, n_couriers=60, seed=202)),
    ("syn_medium_c",    dict(n_tasks=30, n_couriers=60, seed=203)),
    ("syn_large",       dict(n_tasks=40, n_couriers=80, seed=302)),
    ("syn_high_noise",  dict(n_tasks=30, n_couriers=60, s_sd=20.0, w_sd=0.30, seed=601)),
    ("syn_low_willing", dict(n_tasks=30, n_couriers=60, w_mean=0.15, w_sd=0.12, seed=501)),
    ("syn_scarce",      dict(n_tasks=40, n_couriers=22, seed=401)),
]


def main():
    os.makedirs(DATASET, exist_ok=True)
    for name, cfg in CASES:
        text = gen_case(**cfg)
        path = os.path.join(DATASET, f"{name}.txt")
        with open(path, "w") as f:
            f.write(text)
        rows = text.count("\n") - 1
        print(f"  wrote {name}.txt  tasks={cfg['n_tasks']} "
              f"couriers={cfg['n_couriers']} rows={rows}")


if __name__ == "__main__":
    main()
