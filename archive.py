"""
archive.py — persistent learning memory for the AutoSolver agent
================================================================
Cross-run memory so the agent BUILDS ON past attempts instead of restarting.

Canonical history is `run_log.jsonl` (append-only, one record per iteration).
Generated source is archived on disk as `solver_archive/{hash}.py`. These pure
functions read that history to produce a knowledge digest the agent injects
into its opening prompt each run.
"""

import os
import re
import json

HERE = os.path.dirname(__file__)
ARCHIVE_DIR = os.path.join(HERE, "solver_archive")
LOG_PATH = os.path.join(HERE, "run_log.jsonl")


def archive_solver(code_hash: str, code: str) -> str:
    """Persist a generated solver to solver_archive/{hash}.py (idempotent)."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    path = os.path.join(ARCHIVE_DIR, f"{code_hash}.py")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(code)
    return path


def load_solver(code_hash: str):
    """Read archived source for a hash, or None if missing."""
    path = os.path.join(ARCHIVE_DIR, f"{code_hash}.py")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def load_history(log_path: str = LOG_PATH) -> list:
    """All logged iteration records, tolerant of malformed/partial lines."""
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


_ALGO_RE = re.compile(r"^\s*#\s*ALGORITHM:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_algorithm(code: str) -> str:
    """Extract the '# ALGORITHM: ...' header, or 'unlabeled'."""
    m = _ALGO_RE.search(code or "")
    return m.group(1).strip() if m else "unlabeled"


def _submitted_ok(records: list) -> list:
    """Records that were really submitted and scored by the judge."""
    return [r for r in records
            if r.get("submitted") and r.get("ok") and r.get("score") is not None]


def per_case_analysis(records: list) -> list:
    """For each case_file ever scored, the best (min) per-case score and the
    assigned/total at that best. Returns rows sorted WORST→best so the weakest
    cases surface first. Each row: dict(case, best_score, assigned, total)."""
    best = {}  # case_file -> (score, assigned, total)
    for r in _submitted_ok(records):
        for c in (r.get("case_results") or []):
            name = c.get("case_file")
            if name is None or c.get("status") != "ok" or c.get("validity") is False:
                continue
            score = c.get("score", c.get("total_score"))
            if score is None:
                continue
            assigned = c.get("assigned", c.get("assigned_count"))
            total = c.get("total_tasks")
            if name not in best or score < best[name][0]:
                best[name] = (score, assigned, total)
    rows = [{"case": n, "best_score": s, "assigned": a, "total": t}
            for n, (s, a, t) in best.items()]
    rows.sort(key=lambda x: x["best_score"], reverse=True)
    return rows


def best_real_solver(records: list):
    """(hash, score, code) of the lowest real avg_score submission, or None."""
    best = None
    for r in _submitted_ok(records):
        if best is None or r["score"] < best["score"]:
            best = r
    if best is None:
        return None
    return best.get("hash"), best["score"], load_solver(best.get("hash"))


def _approaches_tried(records: list) -> list:
    """Distinct algorithm tags with their best real score (or 'local/invalid').
    Returns list of (algorithm, best_real_score_or_None, outcome) sorted by score."""
    agg = {}  # algorithm -> {"score": min real score or None, "valid_local": bool, "invalid": bool}
    for r in records:
        algo = r.get("algorithm") or "unlabeled"
        a = agg.setdefault(algo, {"score": None, "valid_local": False, "invalid": False})
        if r.get("submitted") and r.get("ok") and r.get("score") is not None:
            if a["score"] is None or r["score"] < a["score"]:
                a["score"] = r["score"]
        elif r.get("gate_ok"):
            a["valid_local"] = True
        else:
            a["invalid"] = True
    out = []
    for algo, a in agg.items():
        if a["score"] is not None:
            out.append((algo, a["score"], f"real={a['score']:.2f}"))
        elif a["valid_local"]:
            out.append((algo, float("inf"), "valid locally, not submitted"))
        else:
            out.append((algo, float("inf"), "INVALID locally"))
    out.sort(key=lambda x: x[1])
    return out


def build_knowledge_digest(records: list) -> str:
    """Formatted accumulated-knowledge block for the agent's opening prompt.
    Empty string when there is no usable history yet."""
    if not records:
        return ""

    lines = ["\n\n=== ACCUMULATED KNOWLEDGE "
             f"(from {len(records)} past attempts) ==="]

    seed = best_real_solver(records)
    cases = per_case_analysis(records)

    if seed:
        _, score, _ = seed
        lines.append(f"BEST REAL avg_score so far: {score:.4f} (LOWER IS BETTER).")
    else:
        lines.append("No submission has been scored by the real judge yet.")

    if cases:
        lines.append("\nPer-case best scores (WEAK SPOTS first — attack these):")
        for row in cases:
            a, t = row["assigned"], row["total"]
            flag = ""
            if a is not None and t is not None and a < t:
                flag = f"  <-- only {a}/{t} assigned, {t - a} tasks UNCOVERED"
            lines.append(f"  {row['case']}: {row['best_score']:.2f}"
                         f"  assigned={a}/{t}{flag}")

    tried = _approaches_tried(records)
    if tried:
        lines.append("\nApproaches already tried (don't just repeat these — "
                     "improve or try something new):")
        for algo, _, outcome in tried[:12]:
            lines.append(f"  - \"{algo}\": {outcome}")

    lines.append("\nKEY LESSONS:")
    lines.append("  - Output each chosen row's task_id_list string VERBATIM "
                 "(no re-sorting) or the case is scored INVALID.")
    lines.append("  - The judge's cost per assignment is NOT raw total_score. "
                 "Optimize for the calibrated cost model provided in the prompt.")
    return "\n".join(lines)
