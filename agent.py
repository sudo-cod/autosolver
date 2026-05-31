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
BASELINE_SCORE = 1710.58  # verified reference-solver avg_score (the bar to beat)


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
  2. Among solutions with equal coverage, MINIMIZE the judge's cost function.
  The judge's per-assignment cost is NOT raw total_score; the calibrated cost
  model (provided separately below, when available) is your true objective.

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
valid solution within the time limit. The judge's true cost function is given
by the calibrated cost model in the prompt — optimize for THAT, not for raw
total_score.

When given feedback (a score and notes on a previous attempt), you diagnose
WHY the score is what it is and make a concrete, targeted change to improve it.
Output only runnable Python code."""

COACH_SYSTEM_PROMPT = """\
You are a strategy coach for an autonomous optimization agent that writes
Python solvers for a courier task-assignment problem (weighted set-packing /
bipartite matching). You do NOT write code. Given the latest results, the
calibrated cost objective, and the trajectory, output ONE concise, concrete
directive telling the solver what to change next: what to fix, what algorithm
to try, or what to STOP doing. Reason about the calibrated cost (not raw
total_score) and the per-case weak spots. Be specific and actionable. If the
solver is stuck repeating an approach, push it toward a genuinely different
algorithm family. Respond with at most 8 lines of plain text — no code."""


def call_claude(model, messages, api_key, max_tokens=8000, system=SYSTEM_PROMPT):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
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


def build_facts(history, last_result: ScoreResult,
                submitted: bool, gate_info: dict):
    """Render the DETERMINISTIC facts (local-gate results, judge per-case
    results, trajectory) the solver/coach reason over. Exact and free — no LLM.

    `submitted` is True if this candidate was actually sent to the real judge
    (so last_result holds per-case feedback); False if it was only evaluated by
    the local gate (no submission spent)."""
    lines = []

    # --- Local gate outcome (always available) -----------------------------
    if gate_info is not None:
        # Multi-case breakdown
        cases = gate_info.get("cases", {})
        if cases:
            n_passed = sum(1 for c in cases.values() if c.get("ok"))
            n_total = len(cases)
            lines.append(f"LOCAL GATE: {n_passed}/{n_total} test cases passed.")
            for cname, cr in cases.items():
                st = cr.get("stats", {})
                elapsed = cr.get("elapsed", 0)
                if cr.get("error"):
                    lines.append(f"  [CRASH] {cname}: {cr['error'][:120]}  "
                                 f"(FIX THIS — solver must not crash/timeout)")
                elif cr.get("errors"):
                    errs = cr["errors"]
                    lines.append(f"  [INVALID] {cname}: covered={st.get('covered',0)}/{st.get('total_tasks',0)}  "
                                 f"errors: {'; '.join(errs[:3])}")
                else:
                    lines.append(f"  [OK] {cname}: covered={st.get('covered',0)}/{st.get('total_tasks',0)}  "
                                 f"score={st.get('total_score',0):.2f}  {elapsed:.1f}s")
            # Highlight the weakest case
            worst = None
            for cname, cr in cases.items():
                st = cr.get("stats", {})
                if st:
                    cov_pct = st.get("covered", 0) / max(st.get("total_tasks", 1), 1)
                    if worst is None or cov_pct < worst[1]:
                        worst = (cname, cov_pct, st)
            if worst and worst[1] < 1.0:
                lines.append(f"\n  WEAKEST CASE: {worst[0]} — only {worst[2].get('covered',0)}/{worst[2].get('total_tasks',0)} "
                             f"tasks covered ({worst[1]*100:.0f}%). Focus improvement here.")
        elif gate_info.get("error"):
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
            failed_cases = []
            ok_cases = []
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
                    failed_cases.append((name, tag, pen, c.get('assigned'), c.get('total_tasks'), errs[:2]))
                else:
                    lines.append(f"  [ok] {name}: score={c.get('score')} "
                                 f"assigned={c.get('assigned')}/{c.get('total_tasks')}")
                    ok_cases.append(name)

            # Actionable diagnosis for failed cases
            if failed_cases:
                lines.append("\nFAILURE ANALYSIS:")
                for name, tag, pen, assigned, total, errs in failed_cases[:5]:
                    if tag == "ERROR" and assigned is None:
                        lines.append(f"  - {name}: SOLVER CRASHED/TIMED OUT. "
                                     f"The solve() function did not return in time or raised an exception. "
                                     f"Simplify your algorithm or add a timeout guard.")
                    elif tag == "INVALID":
                        lines.append(f"  - {name}: INVALID OUTPUT. "
                                     f"The solution format was wrong. "
                                     f"Ensure task_id_list_str matches input rows VERBATIM.")
                    elif assigned is not None and total is not None and assigned < total:
                        uncovered = total - assigned
                        lines.append(f"  - {name}: only {assigned}/{total} assigned, "
                                     f"{uncovered} tasks uncovered (penalty={pen}). "
                                     f"Improve coverage on this case type.")
    else:
        lines.append("\n(This candidate was NOT submitted to the real judge — "
                     "we only spend one of the limited daily submissions on a "
                     "solver that is locally valid and improved.)")

    # --- Trajectory with scores ---------------------------------------------
    if history:
        lines.append("\nHistory (iter: local-valid? / real-score-if-submitted):")
        for h in history[-10:]:
            tag = "valid" if h.get("gate_ok") else "INVALID"
            score_str = ""
            if h.get("submitted") and h.get("ok") and h.get("score") is not None:
                score_str = f" real_score={h['score']:.2f}"
            pred = h.get("predicted_score")
            pred_str = f" predicted={pred:.1f}" if pred is not None else ""
            lines.append(f"  iter {h['iter']}: {tag}{score_str}{pred_str}  ({h['note'][:60]})")

    return "\n".join(lines)


def _default_directive(submitted, last_result, gate_info):
    """Cheap deterministic directive for ordinary (non-event) iterations."""
    if submitted and last_result.ok and last_result.score is not None:
        return (f"Real score {last_result.score:.2f}. Make ONE targeted change to "
                "the worst per-case result above. Output the full module.")
    return ("Make ONE concrete improvement to the worst-predicted case. "
            "Output ONLY the full revised Python module, starting with imports.")


def coach_directive(facts_text, formula_str, baseline, best_real, api_key, model):
    """Ask the LLM coach for the next strategic directive. Falls back to a
    deterministic line if the coach call fails, so it never blocks the solver."""
    user = (f"Calibrated per-assignment cost objective: {formula_str}\n"
            f"  where ts = the row's total_score, w = the row's willingness "
            f"(accept probability, 0..1). Per-case score = sum of chosen "
            f"assignments' cost + penalty for each UNASSIGNED task. Lower=better.\n"
            f"Reference baseline avg_score: {baseline} | best so far: {best_real}\n\n"
            f"Latest results and trajectory:\n{facts_text}\n\n"
            "Give the solver ONE concrete directive for its next attempt.")
    messages = [{"role": "user", "content": user}]
    for attempt in range(3):
        try:
            out = call_claude(model, messages, api_key,
                              max_tokens=400, system=COACH_SYSTEM_PROMPT)
            out = out.strip()
            if out:
                return "COACH DIRECTIVE:\n" + out
        except Exception as e:
            print(f"  coach call failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    return None


def _cost_formula_str(cm):
    """Render the calibrated per-assignment cost formula as readable text."""
    if cm is None:
        return "uncalibrated"
    if cm.form == "expected_value":
        P = cm.params.get("P", cm.penalty_per_task)
        return (f"cost = willingness*total_score + "
                f"(1 - willingness)*{P:.1f}*num_tasks")
    if cm.form.startswith("exact:"):
        return f"cost = {cm.form.split(':', 1)[1]}"
    terms = " ".join(f"{coef:+.3f}*{t}" for t, coef in cm.params.items())
    return f"cost = {terms}"


def _build_formula_desc(cm):
    """Return a human-readable description of the calibrated cost formula
    suitable for injection into the LLM's problem brief."""
    if cm is None or not cm.calibrated:
        return ""
    desc = _cost_formula_str(cm)
    if cm.form == "expected_value":
        desc += ("\n  Intuition: an assignment delivered with probability "
                 "`willingness` costs its total_score; if it FAILS, each of its "
                 "tasks costs 100. So PREFER high-willingness, low-total_score "
                 "assignments. Leaving a task unassigned also costs 100.")
    desc += f"\n  (max_residual={cm.max_resid:.4g}, "
    desc += f"unassigned penalty {cm.penalty_per_task:.1f}/task, "
    desc += f"calibrated on {cm.n_points} points.)"
    return ("\n\n[CALIBRATED COST MODEL — minimize this; it is your true objective]\n"
            + desc)


def _format_formula(cm):
    """Short one-line formula string for inline system messages."""
    if cm is None:
        return "uncalibrated"
    return f"{_cost_formula_str(cm)}, penalty={cm.penalty_per_task:.1f}/task"


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

    # Build calibrated formula description for the LLM
    formula_desc = _build_formula_desc(_cm)

    # Seed the opening prompt from accumulated knowledge so each run BUILDS ON
    # past attempts instead of restarting from scratch.
    past = archive.load_history()
    digest = archive.build_knowledge_digest(past)
    seed = archive.best_real_solver(past)
    if digest:
        print(f"Loaded knowledge from {len(past)} past attempts.")
    opening = PROBLEM_BRIEF + formula_desc + digest
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

        # 1) GENERATE / REWRITE code (retry up to 3 times on transient errors)
        raw = None
        for _retry in range(3):
            try:
                raw = call_claude(model, messages, api_key)
                break
            except Exception as e:
                print(f"  API call failed (attempt {_retry+1}/3): {e}")
                if _retry < 2:
                    wait = 5 * (_retry + 1)
                    print(f"  retrying in {wait}s...")
                    time.sleep(wait)
        if raw is None:
            print("  API call failed after 3 attempts — skipping this iteration.")
            continue
        code = strip_fences(raw)
        h = code_hash(code)
        algo = archive.parse_algorithm(raw)
        archive.archive_solver(h, code)
        print(f"  generated solver  [hash {h}]  ({len(code)} chars)  algo: {algo}")

        # 2) LOCAL GATE — run + validate on ALL local cases WITHOUT spending a submission
        gate_ok, gate_info = run_local_gate(code)
        if gate_ok:
            local_score = _local_score(gate_info["stats"])
            st = gate_info["stats"]
            n_cases = gate_info.get("n_cases", 1)
            print(f"  LOCAL VALID across {n_cases} case(s)  "
                  f"covered={st['covered']}/{st['total_tasks']} "
                  f"total_score={st['total_score']:.2f} "
                  f"predicted_score={local_score:.1f} ({gate_info['elapsed']:.1f}s)")
            # Print per-case breakdown
            for cname, cinfo in gate_info.get("cases", {}).items():
                cs = cinfo.get("stats", {})
                cpred = cs.get("predicted_score", 0.0)
                print(f"    {cname}: covered={cs.get('covered')}/{cs.get('total_tasks')} "
                      f"predicted={cpred:.1f} ({cinfo.get('elapsed', 0):.1f}s)")
        else:
            local_score = float("inf")
            # Summarize which cases failed and why
            fail_summary = []
            for cname, cinfo in gate_info.get("cases", {}).items():
                if cinfo.get("error"):
                    fail_summary.append(f"{cname}: CRASH")
                elif cinfo.get("errors"):
                    fail_summary.append(f"{cname}: INVALID")
            reason = gate_info.get("error") or "; ".join(fail_summary[:5])
            print(f"  LOCAL INVALID — not submittable. {reason[:200]}")

        # 3) DECIDE: spend a real submission?
        submitted = False
        result = ScoreResult(ok=False, message="not submitted (local-only)")
        if is_http and gate_ok:
            first_ever = state["best_real_score"] is None
            improved = local_score < last_submitted_local
            # Sanity check: if predicted_score is negative or absurd, the local
            # model is unreliable for this solver — don't waste a submission.
            sane = local_score > -1000  # real scores are ~200-3000
            if not sane:
                print(f"  LOCAL SCORE ABSURD ({local_score:.1f}) — skipping submission "
                      f"(calibrated model unreliable for this solver).")
            elif state["daily_remaining"] <= 0:
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
        predicted_score = gate_info.get("stats", {}).get("predicted_score") if gate_ok else None
        history.append({"iter": it, "hash": h, "gate_ok": gate_ok,
                        "submitted": submitted, "ok": result.ok,
                        "score": result.score if result.ok else float("inf"),
                        "predicted_score": predicted_score,
                        "algo": algo,
                        "note": result.message})
        with open("run_log.jsonl", "a") as f:
            f.write(json.dumps({
                "iter": it, "hash": h, "algorithm": algo, "gate_ok": gate_ok,
                "gate_cases": gate_info.get("cases"),
                "gate_errors": gate_info.get("errors"),
                "gate_runtime_error": gate_info.get("error"),
                "predicted_score": predicted_score,
                "submitted": submitted,
                "ok": result.ok,
                "score": result.score if result.ok else None,
                "covered": result.accepted_orders,
                "case_results": result.case_results,
                "note": result.message,
                "daily_remaining": state["daily_remaining"],
                "ts": time.time(),
            }) + "\n")

        # 5b) AUTO-RECALIBRATE — refit cost()+penalty using only the latest
        # submission's detail. This keeps the formula current with the best
        # solver's behavior and ignores outliers from old bad solvers.
        if submitted and result.ok and result.case_results:
            report = calibrate.recalibrate(only_best=True)
            if report:
                _cm = calibrate.load_model()
                if _cm and _cm.calibrated:
                    formula_str = _format_formula(_cm)
                    print(f"  [calibrate] Updated formula: {formula_str}")
                    print(f"  [calibrate] max_resid={report['cost_max_err']:.3f} "
                          f"case_max_err={report['case_max_err']:.4f} "
                          f"points={report['n_points']}")
                    # Inject updated formula into the conversation so the LLM
                    # sees the latest cost model for the next generation
                    formula_msg = (
                        f"\n[SYSTEM: Cost formula updated after last submission. "
                        f"Current calibrated model: {formula_str}. "
                        f"Use this as your optimization objective. "
                        f"Lower predicted cost → better real score.]")
                    messages[-1]["content"] = messages[-1]["content"] + formula_msg

        # 6) REFLECT -> next message (skip on last iteration)
        if it < iterations - 1:
            facts = build_facts(history, result, submitted, gate_info)
            # If the solver just produced byte-identical code to the previous
            # iteration, it's stuck in a deterministic loop — flag it loudly so
            # the coach pushes for a genuinely different approach.
            if len(history) >= 2 and history[-1]["hash"] == history[-2]["hash"]:
                facts += ("\n\nWARNING: this solver is BYTE-IDENTICAL to the "
                          "previous iteration — you are repeating yourself. The "
                          "next attempt MUST be a fundamentally different algorithm.")
            # The coach analyzes EVERY iteration and writes the next directive.
            print("  COACH: analyzing results for next directive...")
            directive = coach_directive(
                facts, _format_formula(calibrate.load_model()),
                BASELINE_SCORE, state.get("best_real_score"),
                api_key, model)
            if directive is None:           # coach failed -> deterministic fallback
                directive = _default_directive(submitted, result, gate_info)
            feedback = (facts + "\n\n" + directive
                        + "\n\nHere is your previous code:\n\n" + code)
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
    cases = gate_info.get("cases", {})
    if cases:
        parts = []
        for name, c in cases.items():
            cs = c.get("stats", {})
            if c.get("error"):
                parts.append(f"{name}: ERROR")
            elif c.get("errors"):
                parts.append(f"{name}: INVALID")
            else:
                parts.append(f"{name}: {cs.get('covered', '?')}/{cs.get('total_tasks', '?')}")
        return "VALID [" + "; ".join(parts) + "]"
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
    calibrate.recalibrate(only_best=True)


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

    # Fix 4: Auto-calibrate before the agent loop when in http mode.
    # This ensures the local score predicts the real judge score, so the
    # "is this better?" decision is meaningful.
    is_http = os.environ.get("JUDGE_MODE", "local") == "http"
    if is_http:
        _cm = calibrate.load_model()
        if not (_cm and _cm.calibrated):
            print("No calibrated cost model found — running calibration probe "
                  "(costs 1 submission)...")
            bootstrap_calibration(args.case)
        else:
            # Re-fit from best submission only — avoids distortion from old bad solvers
            print(f"Refreshing calibrated model (current: {_cm.form}, "
                  f"penalty={_cm.penalty_per_task:.3f})...")
            calibrate.recalibrate(only_best=True)

    print(f"AutoSolver Agent | model={args.model} | "
          f"judge_mode={os.environ.get('JUDGE_MODE','local')} | "
          f"iterations={args.iterations}")
    run_agent(args.model, args.iterations, api_key, args.case)


if __name__ == "__main__":
    main()
