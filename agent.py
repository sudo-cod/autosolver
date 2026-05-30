"""
agent.py — AutoSolver Agent
===========================
An agentic loop that WRITES solver code, SUBMITS it for a score, REFLECTS on
the result, and REWRITES the code to drive the score down — autonomously.

  generate  ->  run+submit  ->  score  ->  reflect  ->  generate(better)  -> ...

The "brain" is the Longcat API (LongCat-2.0-Preview by default). The feedback
signal is whatever judge_adapter.submit_and_score returns (real judge once
you wire it; local scorer for offline development).

USAGE:
    # LONGCAT_KEY must be set in .env
    export JUDGE_MODE=local          # or http / browser / manual
    export LOCAL_CASE=/path/to/large_seed301.txt   # for local mode
    python3 agent.py --iterations 12 --model LongCat-2.0-Preview

Outputs:
    best_solver.py     the best-scoring solver found
    run_log.jsonl      full history of every iteration (code hash, score, notes)
"""

import os
import sys
import json
import time
import argparse
import hashlib
import urllib.request

from judge_adapter import submit_and_score, run_local_gate, ScoreResult
import archive
import calibrate

API_URL = "https://api.longcat.chat/anthropic/v1/messages"
STATE_FILE = os.path.join(os.path.dirname(__file__), "agent_state.json")
DAILY_LIMIT = 20  # judge allows 20 submissions/day per team


def _today():
    """Date string in Beijing time (judge resets at Beijing midnight)."""
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600))


def load_state():
    """Persisted submission budget. Resets counters when the date rolls over."""
    state = {"date": _today(), "submissions_used": 0,
             "daily_remaining": DAILY_LIMIT, "best_real_score": None}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            if saved.get("date") == state["date"]:
                state.update(saved)
        except Exception:
            pass
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Problem description handed to the model. This is the agent's understanding
# of the task; keep it faithful to the competition prompt.
# ---------------------------------------------------------------------------
PROBLEM_BRIEF = """\
You are writing a Python function `solve(input_text: str) -> list` for a
food-delivery task-assignment optimization problem.

INPUT: tab-separated text with a header row:
    task_id_list \\t courier_id \\t total_score \\t willingness
Each row is a CANDIDATE assignment: a set of one or more task ids (comma-
separated, e.g. "T0037,T0039") could be given to courier_id, with a
precomputed total_score (float) and willingness (accept probability, float).

OUTPUT: a list of (task_id_list_str, [courier_id, ...]) tuples — the chosen
assignments. Format example:
    [("T0037,T0039", ["C028"]), ("T0012", ["C073"])]

CRITICAL: the task_id_list_str you output for a chosen row MUST be the EXACT
string from that input row (same task ids, same order, same commas). Do NOT
re-sort, dedupe, or reformat the task ids — the judge matches your string
verbatim against the input candidate rows, and any mismatch is counted INVALID.

OBJECTIVE (lexicographic):
  1. MAXIMIZE the number of distinct orders (tasks) that get covered.
  2. Among solutions with equal coverage, MINIMIZE the total_score summed
     over chosen assignments.

CONSTRAINTS:
  - Each courier may appear in at most ONE chosen assignment.
  - Each task should be covered at most once (one assignment).
  - Only pairs that appear as rows in the input are valid choices.
  - Bundling: a single courier can take a 2-task bundle if such a row exists,
    which can cover two tasks with one courier at one combined score.

HARD RUNTIME LIMIT: your solve() must finish within 10 seconds per case.

You MAY use: standard library, and these if available: pulp (CBC),
ortools, scipy, networkx, numpy. If you use an optional library, GUARD the
import with try/except and provide a working fallback, because it may not be
installed in the judge environment.

Return ONLY the complete Python module text defining solve(). No markdown
fences, no commentary, no explanation. Begin the module with a single comment
line naming your approach, e.g.:
    # ALGORITHM: greedy by total_score ascending with 2-task bundles
then the imports and code.
"""

SYSTEM_PROMPT = """\
You are an expert in combinatorial optimization and competitive programming.
You write correct, fast, self-contained Python solvers. You think carefully
about the objective and constraints, exploit problem structure (this is a
weighted bipartite matching / set-packing problem), and you ALWAYS return a
valid solution within the time limit. When given feedback (a score and notes
on a previous attempt), you diagnose WHY the score is what it is and make a
concrete, targeted change to improve it. Output only runnable Python code."""


def call_claude(model, messages, api_key, max_tokens=8000):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    # concatenate text blocks
    return "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")


_CODE_START = ("import ", "from ", "def ", "class ", "#!", "# ")


def strip_fences(code: str) -> str:
    """Extract runnable Python from an LLM reply that may wrap it in markdown
    fences and/or surround it with prose."""
    code = code.strip()

    # 1) If there are ``` fences anywhere, take the LARGEST fenced block — that
    #    is the actual code, regardless of any prose before/after it.
    if "```" in code:
        blocks = []
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("```"):
                j = i + 1
                buf = []
                while j < len(lines) and not lines[j].lstrip().startswith("```"):
                    buf.append(lines[j])
                    j += 1
                blocks.append("\n".join(buf))
                i = j + 1
            else:
                i += 1
        if blocks:
            code = max(blocks, key=len).strip()

    # 2) Drop any remaining leading prose before the first Python-looking line.
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        if any(line.startswith(p) for p in _CODE_START):
            code = "\n".join(lines[idx:])
            break

    # 3) Drop a stray trailing fence if one slipped through.
    code = code.strip()
    if code.endswith("```"):
        code = code[:code.rfind("```")].rstrip()
    return code.strip()


def code_hash(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def build_reflection(history, last_code, last_result: ScoreResult,
                     submitted: bool, gate_info: dict):
    """Construct the feedback message that drives the next rewrite.

    `submitted` is True if this candidate was actually sent to the real judge
    (so last_result holds per-case feedback); False if it was only evaluated by
    the local gate (no submission spent)."""
    lines = []

    # --- Local gate outcome (always available) -----------------------------
    if gate_info is not None:
        if gate_info.get("error"):
            lines.append("LOCAL GATE: your solver crashed or timed out BEFORE "
                         "any submission was spent.")
            lines.append(f"Runtime error: {gate_info['error']}")
            lines.append("Fix this FIRST — make solve() run cleanly within the "
                         "time limit on large inputs.")
        elif gate_info.get("errors"):
            errs = gate_info["errors"]
            lines.append("LOCAL GATE: your solver ran but produced an INVALID "
                         "solution. No submission was spent.")
            for e in errs[:8]:
                lines.append(f"  - {e}")
            if len(errs) > 8:
                lines.append(f"  ...(+{len(errs)-8} more)")
            lines.append("Most common cause: the task_id_list string you output "
                         "does not EXACTLY match an input row. Preserve it verbatim.")
        else:
            st = gate_info.get("stats", {})
            lines.append(f"LOCAL GATE: solver is VALID locally — "
                         f"covered={st.get('covered')}/{st.get('total_tasks')}, "
                         f"total_score={st.get('total_score', 0):.2f}, "
                         f"ran in {gate_info.get('elapsed', 0):.1f}s.")

    # --- Real judge feedback (only when submitted) -------------------------
    if submitted:
        if last_result.ok:
            lines.append(f"\nREAL JUDGE avg penalty: {last_result.score:.4f} "
                         f"(LOWER IS BETTER). success={last_result.accepted_orders}/10 cases.")
        else:
            lines.append(f"\nREAL JUDGE rejected the submission: {last_result.message}")
        if last_result.case_results:
            lines.append("Per-case judge results:")
            for c in last_result.case_results:
                name = c.get("case_file", "?")
                status = c.get("status", "?")
                validity = c.get("validity")
                errs = c.get("errors") or []
                if status != "ok" or validity is False:
                    pen = c.get("penalty_score")
                    if pen is None and c.get("total_tasks") is not None:
                        pen = c["total_tasks"] * 100
                    tag = "INVALID" if validity is False else "ERROR"
                    e0 = f"  err: {errs[0]}" if errs else ""
                    lines.append(f"  [{tag}] {name}: penalty={pen} "
                                 f"assigned={c.get('assigned')}/{c.get('total_tasks')}{e0}")
                else:
                    lines.append(f"  [ok] {name}: score={c.get('score')} "
                                 f"assigned={c.get('assigned')}/{c.get('total_tasks')}")
    else:
        lines.append("\n(This candidate was NOT submitted to the real judge — "
                     "we only spend one of the limited daily submissions on a "
                     "solver that is locally valid and improved.)")

    # --- Trajectory --------------------------------------------------------
    if history:
        lines.append("\nHistory (iter: local-valid? / real-score-if-submitted):")
        for h in history[-8:]:
            tag = "valid" if h.get("gate_ok") else "INVALID"
            sub = f" submitted->{h['score']:.2f}" if h.get("submitted") and h.get("ok") else ""
            lines.append(f"  iter {h['iter']}: {tag}{sub}  ({h['note'][:60]})")

    lines.append("\nDiagnose the single biggest problem above, then make ONE "
                 "concrete change. Keep solve() correct and under 10s per case. "
                 "Output ONLY the full revised Python module, starting with imports.")
    lines.append("\nHere is your previous code:\n\n" + last_code)
    return "\n".join(lines)


def _local_score(stats):
    """Local objective (lower = better). When a calibrated cost model exists,
    this is the PREDICTED real per-case score (sum cost + penalty*uncovered) —
    directly comparable to the judge. Otherwise fall back to the coverage-first
    heuristic."""
    if "predicted_score" in stats:
        return stats["predicted_score"]
    return -1e6 * stats.get("covered", 0) + stats.get("total_score", 0.0)


def run_agent(model, iterations, api_key, case_name):
    is_http = os.environ.get("JUDGE_MODE", "local") == "http"
    state = load_state()
    print(f"Submission budget: {state['daily_remaining']}/{DAILY_LIMIT} left today "
          f"(used {state['submissions_used']}); "
          f"best real score so far: {state['best_real_score']}")

    _cm = calibrate.load_model()
    if _cm and _cm.calibrated:
        print(f"Local scoring: CALIBRATED (cost={_cm.form}, "
              f"penalty/task={_cm.penalty_per_task:.3f}) — local score predicts real.")
    else:
        print("Local scoring: heuristic (uncalibrated) — run --calibrate to fit the "
              "real cost formula.")

    history = []  # this run's iterations (run_log.jsonl holds the full cross-run history)

    # Seed the opening prompt from accumulated knowledge so each run BUILDS ON
    # past attempts instead of restarting from scratch.
    past = archive.load_history()
    digest = archive.build_knowledge_digest(past)
    seed = archive.best_real_solver(past)
    if digest:
        print(f"Loaded knowledge from {len(past)} past attempts.")
    opening = PROBLEM_BRIEF + digest
    if seed and seed[2]:
        _, seed_score, seed_code = seed
        opening += (f"\n\nHere is the BEST solver so far (real avg_score="
                    f"{seed_score:.4f}). Make ONE concrete improvement, "
                    f"especially on the weak cases above. Output the full module:\n\n"
                    + seed_code)
    else:
        opening += "\n\nWrite the first version now."
    messages = [{"role": "user", "content": opening}]

    best = {"score": float("inf"), "code": None, "iter": -1}  # best LOCAL solver
    last_submitted_local = float("inf")  # local score of the last solver we submitted

    for it in range(iterations):
        print(f"\n{'='*60}\nITERATION {it}\n{'='*60}")

        # 1) GENERATE / REWRITE code
        try:
            raw = call_claude(model, messages, api_key)
        except Exception as e:
            print(f"  API call failed: {e}")
            break
        code = strip_fences(raw)
        h = code_hash(code)
        algo = archive.parse_algorithm(raw)
        archive.archive_solver(h, code)
        print(f"  generated solver  [hash {h}]  ({len(code)} chars)  algo: {algo}")

        # 2) LOCAL GATE — run + validate WITHOUT spending a submission
        gate_ok, gate_info = run_local_gate(code)
        if gate_ok:
            local_score = _local_score(gate_info["stats"])
            st = gate_info["stats"]
            print(f"  LOCAL VALID  covered={st['covered']}/{st['total_tasks']} "
                  f"total_score={st['total_score']:.2f} "
                  f"local_est={local_score:.1f} ({gate_info['elapsed']:.1f}s)")
        else:
            local_score = float("inf")
            reason = gate_info.get("error") or "; ".join(gate_info.get("errors", [])[:3])
            print(f"  LOCAL INVALID — not submittable. {reason[:120]}")

        # 3) DECIDE: spend a real submission?
        submitted = False
        result = ScoreResult(ok=False, message="not submitted (local-only)")
        if is_http and gate_ok:
            first_ever = state["best_real_score"] is None
            improved = local_score < last_submitted_local
            if state["daily_remaining"] <= 0:
                print("  BUDGET EXHAUSTED — iterating locally only (no submission).")
            elif first_ever or improved:
                print(f"  SUBMITTING to real judge "
                      f"({state['daily_remaining']} left)...")
                result = submit_and_score(code, case_name)
                submitted = True
                state["submissions_used"] += 1
                state["daily_remaining"] = (
                    result.daily_remaining if result.daily_remaining is not None
                    else state["daily_remaining"] - 1)
                last_submitted_local = local_score
                if result.ok:
                    print(f"  REAL avg_penalty = {result.score:.4f} "
                          f"(success {result.accepted_orders}/10)")
                    if state["best_real_score"] is None or result.score < state["best_real_score"]:
                        state["best_real_score"] = result.score
                        with open("best_solver.py", "w") as f:
                            f.write(code)
                        print(f"  *** NEW REAL BEST: {result.score:.4f} "
                              f"(saved best_solver.py)")
                else:
                    print(f"  REAL JUDGE rejected: {result.message[:120]}")
                save_state(state)
            else:
                print("  SKIP submission — not better locally than last submitted.")

        # In local dev mode, take the gate result as the score signal.
        if not is_http and gate_ok:
            result = ScoreResult(ok=True, score=local_score,
                                 accepted_orders=gate_info["stats"]["covered"],
                                 message=gate_info_msg(gate_info))

        # 4) Track best LOCAL solver. Only write best_solver.py from local score
        # as a FALLBACK when no real judge score exists yet — real scores own the
        # file (saved in the submission block above).
        if gate_ok and local_score < best["score"]:
            best = {"score": local_score, "code": code, "iter": it}
            if state["best_real_score"] is None:
                with open("best_solver.py", "w") as f:
                    f.write(code)
                print(f"  *** NEW LOCAL BEST: {local_score:.1f} (saved best_solver.py)")
            else:
                print(f"  new local best {local_score:.1f} "
                      f"(best_solver.py kept = real-scored solver)")

        # 5) LOG
        history.append({"iter": it, "hash": h, "gate_ok": gate_ok,
                        "submitted": submitted, "ok": result.ok,
                        "score": result.score if result.ok else float("inf"),
                        "note": result.message})
        with open("run_log.jsonl", "a") as f:
            f.write(json.dumps({
                "iter": it, "hash": h, "algorithm": algo, "gate_ok": gate_ok,
                "gate_errors": gate_info.get("errors"),
                "gate_runtime_error": gate_info.get("error"),
                "submitted": submitted,
                "ok": result.ok,
                "score": result.score if result.ok else None,
                "covered": result.accepted_orders,
                "case_results": result.case_results,
                "note": result.message,
                "daily_remaining": state["daily_remaining"],
                "ts": time.time(),
            }) + "\n")

        # 5b) AUTO-RECALIBRATE — if we just captured fresh per-assignment detail,
        # refit cost()+penalty and re-check predicted-vs-actual (no submission).
        if submitted and result.ok and result.case_results:
            calibrate.recalibrate()

        # 6) REFLECT -> next message (skip on last iteration)
        if it < iterations - 1:
            feedback = build_reflection(history, code, result, submitted, gate_info)
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": feedback})
            if len(messages) > 7:
                messages = messages[:1] + messages[-6:]

    print(f"\n{'='*60}\nDONE.")
    print(f"Best LOCAL solver: {best['score']:.1f} at iter {best['iter']} (best_solver.py)")
    print(f"Best REAL judge score: {state['best_real_score']}")
    print(f"Submissions used today: {state['submissions_used']}/{DAILY_LIMIT}")
    return best


def gate_info_msg(gate_info):
    st = gate_info.get("stats", {})
    return (f"VALID covered={st.get('covered')}/{st.get('total_tasks')} "
            f"total_score={st.get('total_score', 0):.2f} (local estimate)")


def bootstrap_calibration(case_name):
    """Spend ONE submission on the verified reference solver to capture
    per-assignment detail, then fit + validate the cost model."""
    os.environ["JUDGE_MODE"] = "http"  # calibration always probes the real judge
    state = load_state()
    if state["daily_remaining"] <= 0:
        print("Budget exhausted — cannot run calibration probe today."); return
    with open("best_solver.py") as f:
        code = f.read()
    print(f"Calibration probe: submitting reference solver "
          f"({state['daily_remaining']} submissions left)...")
    result = submit_and_score(code, case_name)
    # Only a real judge result carries per-case detail; bail before touching
    # budget/best if we somehow didn't get one (guards against local fallback).
    if not (result.ok and result.case_results):
        print(f"Probe did not return scored detail: {result.message[:160]}"); return
    state["submissions_used"] += 1
    state["daily_remaining"] = (result.daily_remaining
                                if result.daily_remaining is not None
                                else state["daily_remaining"] - 1)
    if result.score is not None and (
            state["best_real_score"] is None or result.score < state["best_real_score"]):
        state["best_real_score"] = result.score
    save_state(state)
    # Log the probe so calibrate can read its detail.
    h = code_hash(code)
    archive.archive_solver(h, code)
    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps({
            "iter": -1, "hash": h, "algorithm": "calibration probe (reference)",
            "gate_ok": True, "submitted": True, "ok": True,
            "score": result.score, "covered": result.accepted_orders,
            "case_results": result.case_results,
            "note": "calibration probe", "daily_remaining": state["daily_remaining"],
            "ts": time.time(),
        }) + "\n")
    print(f"Probe scored avg={result.score:.4f}. Fitting cost model...")
    calibrate.recalibrate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--model", default="LongCat-2.0-Preview")
    ap.add_argument("--case", default="large_seed301")
    ap.add_argument("--calibrate", action="store_true",
                    help="spend 1 submission on the reference solver to fit the "
                         "real cost formula, then exit")
    args = ap.parse_args()

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    if args.calibrate:
        bootstrap_calibration(args.case)
        return

    api_key = os.environ.get("LONGCAT_KEY")
    if not api_key:
        print("ERROR: LONGCAT_KEY not found in environment or .env"); sys.exit(1)

    print(f"AutoSolver Agent | model={args.model} | "
          f"judge_mode={os.environ.get('JUDGE_MODE','local')} | "
          f"iterations={args.iterations}")
    run_agent(args.model, args.iterations, api_key, args.case)


if __name__ == "__main__":
    main()
