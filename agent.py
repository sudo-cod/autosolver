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
import ast
import json
import time
import random
import argparse
import hashlib
import urllib.request
import urllib.error

from judge_adapter import submit_and_score, run_local_gate, ScoreResult
import archive
import calibrate
import reconcile
import memory

API_URL = "https://api.longcat.chat/anthropic/v1/messages"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
STATE_FILE = os.path.join(os.path.dirname(__file__), "agent_state.json")

_PROVIDER = "longcat"   # "longcat" or "deepseek"; set by main() from --provider
DAILY_LIMIT = 20  # judge allows 20 submissions/day per team
BASELINE_SCORE = 1710.58  # verified reference-solver avg_score (the bar to beat)


def _today():
    """Date string in Beijing time (judge resets at Beijing midnight)."""
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600))


def _seconds_to_beijing_midnight(now=None):
    """Seconds remaining until the next Beijing (UTC+8) midnight, when the daily
    submission quota resets."""
    now = time.time() if now is None else now
    bj = now + 8 * 3600                       # shift into Beijing local time
    into_day = bj % 86400.0                   # seconds since Beijing midnight
    return 86400.0 - into_day


# Probe-submission tuning. Submissions are use-it-or-lose-it at Beijing midnight,
# so the acceptance bar loosens as the day runs out AND budget stays unspent.
_URGENCY_BOOST = 1.6      # scales raw (time x budget) pressure into 0..1
_URGENCY_U_SPAN = 0.8     # how far urgency slides the bar (in rmse units)
_PROBE_RESERVE = 2        # keep this many submissions until the final window
_FINAL_WINDOW_S = 2 * 3600  # "final window": last 2h before Beijing reset

# Exploration probing. The global best across teams (real=699) sits ~32 pts below
# the champion, and the suite UNDER-rates full-coverage solvers, so spending some
# submissions mid-day to probe fresh full-coverage candidates near the champion is
# high-value (it both might win AND sharpens the suite->real map toward 699).
# Capped so it can't drain the whole daily quota — the rest is left for the
# time-decay path and clear wins.
_EXPLORE_CAP = 8          # max clock-independent exploration probes per day
_EXPLORE_WINDOW = 1.0     # probe full-coverage candidates within this*rmse of champ


def submit_urgency(state, now=None):
    """0..1 pressure to spend a submission on a merely-tied (not clearly-better)
    full-coverage probe. High only when little Beijing-day time remains AND many
    submissions are still unspent — so a quiet morning does not dump the quota."""
    frac_left = _seconds_to_beijing_midnight(now) / 86400.0
    remaining = state.get("daily_remaining", DAILY_LIMIT)
    budget_ratio = max(0.0, min(1.0, remaining / float(DAILY_LIMIT)))
    raw = (1.0 - frac_left) * budget_ratio * _URGENCY_BOOST
    return max(0.0, min(1.0, raw))


def _candidate_full_coverage(gate_info, min_cov=None):
    """True iff the candidate covers at least `min_cov` tasks in total across all
    local suite cases.  When `min_cov` is None (default) we require every task to
    be covered (the strict definition).  Pass the champion's actual local coverage
    to relax to "at least as good as the champion" — important because the best
    greedy solvers top out at 289/291 on syn_scarce (the two uncoverable tasks
    cost ~100 pts each to force-cover with ILP, which is far worse than 289/291).
    """
    cases = (gate_info or {}).get("cases", {})
    if not cases:
        st = (gate_info or {}).get("stats", {})
        cov, tot = st.get("covered"), st.get("total_tasks")
        if cov is None:
            return False
        threshold = min_cov if min_cov is not None else (tot or 0)
        return cov >= threshold
    total_cov = 0
    total_tot = 0
    for cinfo in cases.values():
        if cinfo.get("error") or cinfo.get("errors"):
            return False
        cs = cinfo.get("stats", {})
        cov, tot = cs.get("covered"), cs.get("total_tasks")
        if cov is None or tot is None:
            return False
        total_cov += cov
        total_tot += tot
    threshold = min_cov if min_cov is not None else total_tot
    return total_cov >= threshold


def _champion_local_coverage():
    """Return (total_covered, total_tasks) for the champion solver on the local
    suite, cached.  Used to set the min-coverage gate so we don't require more
    coverage than the champion itself achieves (e.g. syn_scarce tops out at 38/40
    with greedy — forcing 40/40 costs ~200 pts in ILP penalty)."""
    recs = archive.load_history()
    result = archive.best_real_solver(recs)
    if result is None:
        return None, None
    _, _, code = result
    if not code:
        return None, None
    try:
        _, info = run_local_gate(code)
        st = info.get("stats", {})
        return st.get("covered"), st.get("total_tasks")
    except Exception:
        return None, None


def _submitted_hashes():
    """Set of code hashes already sent to the real judge (from run_log.jsonl), so
    a probe never re-spends a submission on an already-scored solver."""
    seen = set()
    for r in archive.load_history():
        if r.get("submitted") and r.get("hash"):
            seen.add(r["hash"])
    return seen


def load_state():
    """Persisted submission budget. Daily COUNTERS reset when the date rolls
    over, but best_real_score / best_local are PERSISTENT records and carry
    across days (otherwise the agent forgets its champion every midnight and
    regresses)."""
    state = {"date": _today(), "submissions_used": 0,
             "daily_remaining": DAILY_LIMIT, "explore_used": 0,
             "best_real_score": None, "best_local": None}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            # persistent records — keep regardless of date
            for k in ("best_real_score", "best_local"):
                if saved.get(k) is not None:
                    state[k] = saved[k]
            # daily counters — only restore for the same day
            if saved.get("date") == state["date"]:
                state["submissions_used"] = saved.get("submissions_used", 0)
                state["daily_remaining"] = saved.get("daily_remaining", DAILY_LIMIT)
                state["explore_used"] = saved.get("explore_used", 0)
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
assignments. The courier list may hold ONE courier, or SEVERAL (primary +
backups) all assigned to the same task. Format example:
    [("T0037,T0039", ["C028"]), ("T0012", ["C073", "C041"])]
                                          ^^^^^^^^^^^^^^  primary + 1 backup

CRITICAL: the task_id_list_str you output for a chosen row MUST be the EXACT
string from that input row (same task ids, same order, same commas). Do NOT
re-sort, dedupe, or reformat the task ids — the judge matches your string
verbatim against the input candidate rows, and any mismatch is counted INVALID.
Each courier you list (for a given task_str) must exist as an input row with
that exact task_str.

OBJECTIVE (lexicographic):
  1. MAXIMIZE the number of distinct orders (tasks) that get covered.
  2. Among solutions with equal coverage, MINIMIZE the judge's cost function.
  The judge's per-assignment cost is NOT raw total_score; the calibrated cost
  model (provided separately below) is your true objective. BACKUPS are usually
  the single biggest lever — adding a high-willingness backup courier to a task
  makes it almost certain to be covered, slashing the failure penalty.

CONSTRAINTS:
  - Each courier may appear at most ONCE across the whole solution (whether as a
    primary or a backup).
  - Each task may be covered by at most one item — but that item MAY list
    multiple couriers (primary + backups) for the task.
  - Only (task_str, courier) pairs that appear as rows in the input are valid.
  - Bundling: a single courier can take a 2-task bundle if such a row exists,
    covering two tasks with one assignment.

WINNING STRATEGY (proven levers — design your algorithm around these):
  1. COVERAGE IS PARAMOUNT. Every uncovered task costs 100 — far more than any
     assignment's cost. Maximize the number of distinct tasks covered FIRST.
  2. BUNDLE WHEN COURIERS ARE SCARCE. If the number of free couriers is less
     than the number of still-uncovered tasks, cover two tasks at once with a
     2-task bundle row (one courier, two tasks) instead of spending a courier on
     a single. This is the difference between covering ~half vs ALL tasks on
     courier-scarce cases.
  3. SPEND SPARE COURIERS AS BACKUPS. After every task is covered, use leftover
     couriers as backups on the tasks with the LOWEST p_complete (riskiest /
     lowest willingness). Each high-willingness backup drives p_complete→1 and
     strictly lowers cost. Iterate: always back up the currently riskiest task.
  4. For a single primary, prefer the courier minimizing willingness*total_score
     + (1-willingness)*100 (i.e. high willingness, low score).

HARD RUNTIME LIMIT: your solve() must finish within 10 seconds per case.

CRITICAL CORRECTNESS RULES (these are the ONLY ways past solvers failed — obey them):
  1. BACKUPS GO IN ONE TUPLE. To give task X a primary plus backups, output a
     SINGLE tuple ("X", [primary, backup1, backup2]). Do NOT output ("X",[p])
     and ("X",[backup]) as two separate tuples — listing the same task_str in
     two tuples is INVALID ("task already covered"). One task_str => one tuple.
  2. NO REUSE. Each courier id may appear AT MOST ONCE in your entire output
     (whether primary or backup, across all tuples). Track a used-courier set.
  3. ROBUSTNESS OVER CLEVERNESS. Write SIMPLE, correct code. A plain greedy that
     always returns a valid solution beats a fancy ILP/local-search that crashes
     or times out. If you use anything advanced (ILP, 2-opt, local search), wrap
     it in try/except and FALL BACK to a plain greedy so solve() NEVER raises.
  4. Your module must be syntactically valid Python and define solve(input_text)
     returning a list of (task_str, [courier,...]) tuples. Double-check brackets.

GOOD PATTERN: build a fast greedy first (assign each task its best primary by
calibrated cost, then add a high-willingness backup if a distinct courier is
free), keep that as `result`; only then try to improve it inside try/except.

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

NON-NEGOTIABLE: correctness first. Your code must be valid Python, must NEVER
raise, and must finish in <10s. Build a simple greedy that returns a valid
solution, then optionally improve it inside try/except. Put a task's primary +
backups in ONE tuple ("task", [primary, backup,...]); never repeat a task_str
across tuples; never reuse a courier id.

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


CONTROLLER_SYSTEM_PROMPT = """\
You are the CONTROLLER of an autonomous optimization agent solving a courier
task-assignment problem. Each step you see the current state and choose ONE
action. You do NOT write solver code yourself.

Reply with ONLY a JSON object, no prose around it:
  {"thought": "<1-2 sentence reasoning>", "action": "<ACTION>", "focus": "<...>"}

ACTIONS:
  GENERATE  - write/improve a solver. Put a concrete one-line strategy in "focus"
              (e.g. "raise scarce-courier coverage by bundling two tasks per
              courier, then add backups to lowest-willingness tasks"). This is
              your main action; use it to attack the WEAKEST case.
  INSPECT   - get a per-case diagnostic of the current best candidate (free, no
              submission). Use when you need to know which case is weak.
  SUBMIT    - spend ONE of the limited daily judge submissions on the current
              best candidate. Choose this when the state marks the candidate as
              "CLEARLY BEATS" the champion, OR as "PROBE-WORTHY" (tied locally but
              full-coverage, and submissions would otherwise be LOST at the
              Beijing-midnight reset — the local suite under-rates full-coverage
              solvers, so a probe can confirm a real win). Higher submit-urgency
              means spend sooner. Do not submit candidates the state says are
              worse than the champion.
  STOP      - end the run (e.g. no budget left, or converged).

Lower scores are better. Cost per assignment = willingness*total_score +
(1-willingness)*100*num_tasks; an uncovered task costs 100. Coverage first,
then bundle when couriers are scarce, then backups on risky tasks.

EFFICIENCY RULES (avoid wasting steps):
  - INSPECT at most ONCE per candidate. If the candidate has not changed since
    your last INSPECT, you already have the per-case breakdown — do NOT INSPECT
    again. GENERATE a different approach or SUBMIT instead.
  - When the state says the candidate CLEARLY BEATS the champion, SUBMIT it
    promptly — do not over-analyze. A submitted improvement becomes the new
    champion and you can keep improving from there.
  - Use GENERATE to actually change the solver; INSPECT only buys information."""


CALLS_LOG = os.path.join(os.path.dirname(__file__), "calls.jsonl")
CALIB_LOG = os.path.join(os.path.dirname(__file__), "calib_log.jsonl")


def _log_call(tag, system, messages, response, temperature):
    """Persist one LLM call (prompt + response) for the visualiser. Best-effort:
    never let logging break a run, and never write the API key."""
    try:
        with open(CALLS_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "tag": tag, "temperature": temperature,
                "system": system, "messages": messages, "response": response,
            }) + "\n")
    except Exception:
        pass


def _snapshot_calib():
    """Append a snapshot of the cost model + reconcile map so the dashboard can
    chart how calibration evolves over time."""
    try:
        cm = calibrate.load_model()
        rec = reconcile.load_model()
        st = load_state()
        with open(CALIB_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "cost_form": cm.form if cm else None,
                "P": (cm.params.get("P") if cm else None),
                "penalty": (cm.penalty_per_task if cm else None),
                "recon_a": rec["a"] if rec else None,
                "recon_b": rec["b"] if rec else None,
                "recon_rmse": rec["rmse"] if rec else None,
                "recon_r2": rec["r2"] if rec else None,
                "best_real": st.get("best_real_score"),
            }) + "\n")
    except Exception:
        pass


_last_call_ts = [0.0]            # module-level pacing clock
_MIN_CALL_INTERVAL = 1.5         # seconds between requests (avoid 429 bursts)


def call_claude(model, messages, api_key, max_tokens=16000, system=SYSTEM_PROMPT,
                attempts=6, timeout=300, temperature=None, log_tag=None,
                return_stop=False):
    """POST to the Longcat API with robust retry. The provider intermittently
    drops connections and rate-limits (429), so: pace requests, honor 429
    retry_after, and retry transport errors with exponential backoff + jitter.
    Returns the text (or `(text, stop_reason)` if return_stop=True).
    `temperature` > 0 diversifies output; `log_tag` persists to calls.jsonl."""
    payload = {"model": model, "max_tokens": max_tokens,
               "system": system, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    body = json.dumps(payload).encode()
    last_err = None
    for i in range(attempts):
        # pace: don't fire requests faster than the min interval
        gap = _MIN_CALL_INTERVAL - (time.time() - _last_call_ts[0])
        if gap > 0:
            time.sleep(gap)
        try:
            req = urllib.request.Request(API_URL, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            _last_call_ts[0] = time.time()
            out = "".join(b.get("text", "") for b in data.get("content", [])
                          if b.get("type") == "text")
            stop = data.get("stop_reason")
            if log_tag:
                _log_call(log_tag, system, messages, out, temperature)
            return (out, stop) if return_stop else out
        except urllib.error.HTTPError as e:
            last_err = e
            _last_call_ts[0] = time.time()
            wait = None
            if e.code == 429:                       # rate limited -> honor retry_after
                try:
                    err = json.loads(e.read().decode())
                    wait = float(err.get("error", {}).get("retry_after", 30))
                except Exception:
                    wait = 30.0
                print(f"  RATE LIMITED (429); waiting {wait:.0f}s...")
            if i < attempts - 1:
                if wait is None:
                    wait = min(45, 2 * (2 ** i)) + random.uniform(0, 1.5)
                    print(f"  API HTTP {e.code} (attempt {i+1}/{attempts}); "
                          f"retrying in {wait:.0f}s...")
                time.sleep(wait)
        except Exception as e:
            last_err = e
            _last_call_ts[0] = time.time()
            if i < attempts - 1:
                wait = min(45, 2 * (2 ** i)) + random.uniform(0, 1.5)
                print(f"  API call failed (attempt {i+1}/{attempts}): {e}; "
                      f"retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise last_err


def call_deepseek(model, messages, api_key, max_tokens=16000, system=SYSTEM_PROMPT,
                  attempts=6, timeout=300, temperature=None, log_tag=None,
                  return_stop=False):
    """Same interface as call_claude but targets the DeepSeek (OpenAI-compat) API."""
    # Convert Anthropic-style messages to OpenAI format: inject system as first message.
    oai_messages = [{"role": "system", "content": system}] + messages
    payload = {"model": model, "max_tokens": max_tokens, "messages": oai_messages}
    if temperature is not None:
        payload["temperature"] = temperature
    body = json.dumps(payload).encode()
    last_err = None
    for i in range(attempts):
        gap = _MIN_CALL_INTERVAL - (time.time() - _last_call_ts[0])
        if gap > 0:
            time.sleep(gap)
        try:
            req = urllib.request.Request(DEEPSEEK_API_URL, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            _last_call_ts[0] = time.time()
            choice = data.get("choices", [{}])[0]
            out = choice.get("message", {}).get("content", "") or ""
            stop = choice.get("finish_reason")
            if log_tag:
                _log_call(log_tag, system, messages, out, temperature)
            return (out, stop) if return_stop else out
        except urllib.error.HTTPError as e:
            last_err = e
            _last_call_ts[0] = time.time()
            wait = None
            if e.code == 429:
                try:
                    err = json.loads(e.read().decode())
                    wait = float(err.get("error", {}).get("retry_after", 30))
                except Exception:
                    wait = 30.0
                print(f"  RATE LIMITED (429); waiting {wait:.0f}s...")
            if i < attempts - 1:
                if wait is None:
                    wait = min(45, 2 * (2 ** i)) + random.uniform(0, 1.5)
                    print(f"  API HTTP {e.code} (attempt {i+1}/{attempts}); "
                          f"retrying in {wait:.0f}s...")
                time.sleep(wait)
        except Exception as e:
            last_err = e
            _last_call_ts[0] = time.time()
            if i < attempts - 1:
                wait = min(45, 2 * (2 ** i)) + random.uniform(0, 1.5)
                print(f"  API call failed (attempt {i+1}/{attempts}): {e}; "
                      f"retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise last_err


def call_llm(model, messages, api_key, **kw):
    """Dispatch to call_claude (Longcat) or call_deepseek based on _PROVIDER."""
    if _PROVIDER == "deepseek":
        return call_deepseek(model, messages, api_key, **kw)
    return call_claude(model, messages, api_key, **kw)


def call_claude_until_complete(model, messages, api_key, parse_ok,
                               max_cont=4, **kw):
    """Generate, and if the completion is cut off (hit the token ceiling),
    CONTINUE it exactly like the Claude CLI's 'continue': send the partial
    assistant text back UNTRIMMED as an assistant turn; the model resumes from
    that precise point. Concatenate partial + continuation. Repeat until the
    code parses or continuations are exhausted."""
    acc, stop = call_llm(model, messages, api_key, return_stop=True, **kw)
    cont = 0
    while not parse_ok(acc) and cont < max_cont:
        cont += 1
        print(f"  [continue] truncated (stop={stop}) — continuing (#{cont})...")
        # Faithful prefill: send `acc` AS-IS (no trimming). The model continues
        # the assistant message from exactly where it stopped.
        more, stop = call_llm(
            model, messages + [{"role": "assistant", "content": acc}],
            api_key, return_stop=True, **kw)
        if not more.strip():
            break                                  # nothing came back; give up
        combined = acc + more
        if parse_ok(combined):                     # extended -> done
            acc = combined
        elif parse_ok(more):                        # model restarted with a full answer
            acc = more
        else:                                      # still partial -> keep extending
            acc = combined
    return acc


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


import re as _re

# Safety net appended to every solver. The LLM's solve() is renamed to
# _user_solve; the new solve() runs it but falls back to a known-good greedy
# (primary + 1 backup per task, no courier reuse) on ANY error or bad output.
# This guarantees a valid solution both locally AND on the real judge, killing
# the residual crash/timeout class. Self-contained (stdlib only).
_FALLBACK_TEMPLATE = '''

# ===== harness safety net (auto-appended) =====
def _harness_fallback_solve(input_text):
    rows = {}
    lines = input_text.strip().split("\\n")
    for line in lines[1:]:
        p = line.split("\\t")
        if len(p) < 4:
            continue
        ts, c = p[0].strip(), p[1].strip()
        try:
            s = float(p[2]); w = float(p[3])
        except Exception:
            continue
        nt = ts.count(",") + 1
        cost = w * s + (1.0 - w) * 100.0 * nt
        rows.setdefault(ts, []).append((cost, c))
    for t in rows:
        rows[t].sort()
    order = sorted(rows.keys(), key=lambda t: rows[t][0][0])
    used = set(); result = []
    for t in order:
        cands = [x for x in rows[t] if x[1] not in used]
        if not cands:
            continue
        prim = cands[0]; used.add(prim[1]); cours = [prim[1]]
        rest = [x for x in rows[t] if x[1] not in used]
        if rest:
            used.add(rest[0][1]); cours.append(rest[0][1])
        result.append((t, cours))
    return result


def solve(input_text):
    try:
        r = _user_solve(input_text)
    except Exception:
        return _harness_fallback_solve(input_text)
    if not isinstance(r, list) or len(r) == 0:
        return _harness_fallback_solve(input_text)
    for item in r:
        if not (isinstance(item, (list, tuple)) and len(item) == 2
                and isinstance(item[0], str) and isinstance(item[1], list)
                and len(item[1]) > 0):
            return _harness_fallback_solve(input_text)
    return r
'''


def wrap_with_fallback(code: str) -> str:
    """Rename the LLM's top-level `def solve(` to `_user_solve` and append a
    safety-net `solve()` that falls back to a builtin greedy on any failure.
    If no top-level solve() is found, return the code unchanged. Idempotent:
    already-wrapped code (has the harness marker) is returned untouched."""
    if "_harness_fallback_solve" in code:
        return code                      # already wrapped
    new_code, n = _re.subn(r"(?m)^def solve\b", "def _user_solve", code, count=1)
    if n == 0:
        return code                      # nothing to wrap; leave as-is
    return new_code.rstrip() + "\n" + _FALLBACK_TEMPLATE


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
                                 f"errors: {'; '.join(errs[:5])}")
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
            # Classify errors and append specific fix hints
            all_errs = []
            for cname, cr in cases.items():
                all_errs.extend(cr.get("errors", []))
            if all_errs:
                hint_lines = []
                if any("already covered" in e for e in all_errs):
                    hint_lines.append(
                        "BUG FIX: Tasks are assigned to multiple output tuples. "
                        "Maintain a `covered_tasks` set. Before adding a tuple, check "
                        "that none of its tasks are already in `covered_tasks`.")
                if any("not a valid input row" in e for e in all_errs):
                    hint_lines.append(
                        "BUG FIX: Your task_id_list strings don't match input rows verbatim. "
                        "Use the EXACT string from the input row — do not re-sort, reformat, "
                        "or construct new task strings.")
                if any("used more than once" in e for e in all_errs):
                    hint_lines.append(
                        "BUG FIX: Courier IDs are reused across tuples. "
                        "Maintain a `used_couriers` set and never add a courier already in it.")
                if any("not a (task_str" in e for e in all_errs):
                    hint_lines.append(
                        "BUG FIX: Output items are not (task_str, [courier]) tuples. "
                        "Each item must be a 2-element tuple with a string and a non-empty list.")
                for hl in hint_lines:
                    lines.append(f"\n  >>> {hl}")
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
            # Error-type-specific fix hints
            if any("already covered" in e for e in errs):
                lines.append("BUG FIX: Tasks assigned to multiple tuples. Maintain a "
                             "`covered_tasks` set and skip already-covered tasks.")
            if any("not a valid input row" in e for e in errs):
                lines.append("BUG FIX: task_id_list strings don't match input rows verbatim. "
                             "Use the EXACT string from the input row.")
            if any("used more than once" in e for e in errs):
                lines.append("BUG FIX: Courier IDs reused. Maintain a `used_couriers` set.")
            if any("not a (task_str" in e for e in errs):
                lines.append("BUG FIX: Items are not (task_str, [courier]) tuples.")
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


def _default_directive(submitted, last_result, gate_info, prev_gate_ok=None):
    """Cheap deterministic directive for ordinary (non-event) iterations.
    When the previous solver was INVALID, tell the model to fix specific bugs
    rather than write a new algorithm."""
    # If the previous solver was INVALID, force a fix-not-restart directive
    if prev_gate_ok is False:
        # Collect the first few errors from gate_info
        err_snippets = []
        cases = gate_info.get("cases", {}) if gate_info else {}
        for cname, cr in cases.items():
            for e in cr.get("errors", [])[:2]:
                err_snippets.append(e[:100])
            if len(err_snippets) >= 3:
                break
        if not err_snippets:
            err_snippets = [gate_info.get("error", "unknown error")[:100]] if gate_info else ["unknown"]
        bugs = "\n  - ".join(err_snippets[:5])
        return (
            f"Your previous code was INVALID. Fix these specific bugs:\n"
            f"  - {bugs}\n"
            f"Fix THIS code. Do NOT write a completely new algorithm — correct the "
            f"bugs in the existing one. Keep the same overall approach but fix the "
            f"correctness issues. Output the full corrected module."
        )
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
    try:
        # Fewer attempts than the solver — the coach is optional; on failure the
        # caller falls back to a deterministic directive.
        out = call_llm(model, messages, api_key, max_tokens=400,
                       system=COACH_SYSTEM_PROMPT, attempts=3,
                       log_tag="coach").strip()
        return ("COACH DIRECTIVE:\n" + out) if out else None
    except Exception as e:
        print(f"  coach unavailable after retries: {e}")
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
        P = cm.params.get("P", cm.penalty_per_task)
        desc += (
            "\n  Single courier: cost = willingness*total_score + "
            f"(1-willingness)*{P:.0f}*num_tasks.\n"
            "  BACKUPS (assign a task to MULTIPLE couriers as primary+backups, "
            "i.e. output (task_str, [c1, c2, ...])): the task is covered unless "
            "ALL decline, which slashes the failure penalty. Exact formula for a "
            "task served by couriers i with score s_i, willingness w_i:\n"
            "      p_complete     = 1 - prod(1 - w_i)\n"
            "      expected_score = sum(w_i*s_i) / sum(w_i)\n"
            f"      cost           = p_complete*expected_score + (1-p_complete)*{P:.0f}*num_tasks\n"
            "  Adding a high-willingness backup to a task is usually a BIG win "
            "(drives p_complete -> 1). Every courier may be used at most once "
            f"overall. Leaving a task unassigned costs {P:.0f}/task.")
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


# ===========================================================================
# Reusable tools (used by both the linear loop and the ReAct controller).
# ===========================================================================
def _weakest_case_label(cases):
    """Short 'name=pred' for the highest-cost (or least-covered) case in a
    gate_cases dict, or '' when unavailable."""
    worst = None
    for cname, cinfo in (cases or {}).items():
        if cinfo.get("error"):
            return f"{cname}=CRASH"
        if cinfo.get("errors"):
            return f"{cname}=INVALID"
        cs = cinfo.get("stats", {})
        cov, tot = cs.get("covered"), cs.get("total_tasks")
        pred = cs.get("predicted_score", 0.0)
        key = (pred, -(cov or 0))
        if worst is None or key > worst[0]:
            worst = (key, cname, pred)
    return f"{worst[1]}={worst[2]:.0f}" if worst else ""


def _built_ledger(limit=8):
    """A compact 'already-built' ledger from run_log.jsonl: the last `limit`
    DISTINCT candidate hashes, each as `algo · pred=.. · weak=.. [· real=..]`.
    Feeding this to the SOLVER (not just the controller) breaks the duplicate-
    hash loops where it re-derives a solver it already tried."""
    recs = archive.load_history()
    seen, rows = set(), []
    for r in reversed(recs):
        h = r.get("hash")
        if not h or h in seen:
            continue
        pred = r.get("predicted_score")
        if pred is None and not (r.get("submitted") and r.get("ok")):
            continue
        seen.add(h)
        algo = (r.get("algorithm") or "unlabeled")[:46]
        weak = _weakest_case_label(r.get("gate_cases"))
        parts = [f"{algo}"]
        if pred is not None:
            parts.append(f"pred={pred:.0f}")
        if weak:
            parts.append(f"weak={weak}")
        if r.get("submitted") and r.get("ok") and r.get("score") is not None:
            parts.append(f"REAL={r['score']:.0f}")
        rows.append(f"  [{h[:8]}] " + " · ".join(parts))
        if len(rows) >= limit:
            break
    if not rows:
        return ""
    return ("\n\n[ALREADY BUILT — these solvers already exist with the results "
            "shown; do NOT reproduce them. Make a GENUINELY different change, "
            "and target the weakest case]:\n" + "\n".join(rows))


def generate_candidates(model, messages, api_key, n_candidates=1, focus=None):
    """Best-of-N: generate N solvers (temperature-diversified when N>1), wrap +
    gate each, return the best-local candidate dict (or None if all failed).
    `focus` is a one-line directive appended to the conversation for this call."""
    # Append the controller focus + an "already-built" ledger to the last user
    # turn so the solver sees what it has already tried (and their results) and
    # stops re-deriving identical solvers.
    extra = ""
    if focus:
        extra += f"\n\n[CONTROLLER FOCUS]: {focus}"
    extra += _built_ledger()
    msgs = messages
    if extra:
        msgs = messages[:-1] + [{"role": messages[-1]["role"],
                                 "content": messages[-1]["content"] + extra}]
    # A generation is only usable if its extracted code actually parses; the
    # API often truncates mid-statement under load. parse_ok drives the
    # continuation loop and the accept/discard decision.
    def _parses(raw):
        try:
            ast.parse(strip_fences(raw))
            return True
        except Exception:
            return False

    cands = []
    for k in range(n_candidates):
        temp = None if n_candidates == 1 else 0.8
        try:
            raw_k = call_claude_until_complete(model, msgs, api_key, _parses,
                                               temperature=temp, log_tag="solver")
        except Exception as e:
            print(f"  cand {k+1}/{n_candidates}: API unavailable: {e}")
            continue
        if not _parses(raw_k):
            print(f"  cand {k+1}/{n_candidates}: still truncated/unparseable "
                  "after continuations — discarded.")
            continue
        code_k = wrap_with_fallback(strip_fences(raw_k))
        h_k = code_hash(code_k)
        algo_k = archive.parse_algorithm(raw_k)
        archive.archive_solver(h_k, code_k)
        gok_k, ginfo_k = run_local_gate(code_k)
        ls_k = _local_score(ginfo_k["stats"]) if gok_k else float("inf")
        if gok_k:
            _st = ginfo_k.get("stats", {})
            _cov = f"{_st.get('covered','?')}/{_st.get('total_tasks','?')}"
            _champ_min, _ = _champion_local_coverage()
            _full = _candidate_full_coverage(ginfo_k, min_cov=_champ_min)
            _tag = "" if _full else "  *** SUB-COVERAGE (below champion level) ***"
            print(f"  cand {k+1}/{n_candidates} [hash {h_k}] {algo_k[:38]} -> "
                  f"valid pred={ls_k:.1f} cov={_cov}{_tag}")
        else:
            print(f"  cand {k+1}/{n_candidates} [hash {h_k}] {algo_k[:38]} -> INVALID")
        cands.append(dict(raw=raw_k, code=code_k, h=h_k, algo=algo_k,
                          gate_ok=gok_k, gate_info=ginfo_k, local_score=ls_k))
    if not cands:
        return None
    return min(cands, key=lambda c: c["local_score"])


def inspect_candidate(gate_info):
    """Per-case covered/predicted breakdown + the weakest case (highest
    predicted, or lowest coverage). Read-only; no LLM/judge call."""
    cases = gate_info.get("cases", {})
    rows, worst = [], None
    for cname, cinfo in cases.items():
        cs = cinfo.get("stats", {})
        if cinfo.get("error"):
            rows.append(f"  {cname}: CRASH"); continue
        if cinfo.get("errors"):
            rows.append(f"  {cname}: INVALID"); continue
        cov, tot = cs.get("covered"), cs.get("total_tasks")
        pred = cs.get("predicted_score", 0.0)
        full = cov is not None and tot is not None and cov >= tot
        # A fully-covered case can only be improved by lowering COST (better
        # primaries / backups), not by chasing coverage. Label it so the
        # controller stops sending "improve coverage" focus to e.g. syn_low_willing.
        lever = "COST lever (100% covered — reduce cost, not coverage)" if full \
            else f"COVERAGE lever ({(tot or 0) - (cov or 0)} uncovered)"
        rows.append(f"  {cname}: covered={cov}/{tot} predicted={pred:.1f}  [{lever}]")
        key = (pred, -(cov or 0))
        if worst is None or key > worst[0]:
            worst = (key, cname, cov, tot, pred, full)
    out = "\n".join(rows)
    if worst:
        lever = ("reduce COST here (it is already fully covered)" if worst[5]
                 else "raise COVERAGE here")
        out += (f"\nWEAKEST: {worst[1]} (covered {worst[2]}/{worst[3]}, "
                f"predicted {worst[4]:.1f}) — target this next: {lever}.")
    return out


def do_submit(code, local_score, state, case_name="large_seed301"):
    """Spend ONE real submission on `code`, update champion/state, then refit the
    cost model + suite->real map. Returns the ScoreResult. Caller enforces the
    budget/validity/not-clearly-worse guardrails BEFORE calling this."""
    print(f"  SUBMITTING to real judge ({state['daily_remaining']} left)...")
    result = submit_and_score(code, case_name)
    state["submissions_used"] += 1
    state["daily_remaining"] = (result.daily_remaining
                                if result.daily_remaining is not None
                                else state["daily_remaining"] - 1)
    if result.ok:
        print(f"  REAL avg_penalty = {result.score:.4f} "
              f"(success {result.accepted_orders}/10)")
        if state["best_real_score"] is None or result.score < state["best_real_score"]:
            state["best_real_score"] = result.score
            state["best_local"] = local_score
            with open("best_solver.py", "w") as f:
                f.write(code)
            print(f"  *** NEW REAL BEST: {result.score:.4f} (saved best_solver.py)")
    else:
        print(f"  REAL JUDGE rejected: {result.message[:120]}")
    save_state(state)
    # Persist the submission so calibrate/reconcile can mine its detail.
    h = code_hash(code)
    archive.archive_solver(h, code)
    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps({
            "hash": h, "algorithm": archive.parse_algorithm(code),
            "submitted": True, "ok": result.ok,
            "score": result.score if result.ok else None,
            "covered": result.accepted_orders,
            "case_results": result.case_results,
            "predicted_score": local_score,
            "note": result.message[:80], "daily_remaining": state["daily_remaining"],
            "ts": time.time(),
        }) + "\n")
    if result.ok and result.case_results:
        calibrate.recalibrate(only_best=True)
        reconcile.recompute()
        # Best-effort: refresh the distilled strategy memory with the new result.
        try:
            memory.consolidate(call_llm, _SUBMIT_MODEL[0], _SUBMIT_MODEL[1])
        except Exception:
            pass
    _snapshot_calib()
    return result


_SUBMIT_MODEL = [None, None]   # (model, api_key) set by run_agent_react for hooks


_ACTIONS = {"GENERATE", "INSPECT", "SUBMIT", "STOP"}


def _extract_json_objects(text):
    """Yield every top-level balanced {...} substring in `text`, last-to-first.
    A greedy `\\{.*\\}` regex grabs one span from the first '{' to the last '}',
    which breaks when a reasoning model (e.g. deepseek-reasoner) emits braces in
    its prose around the real answer. This brace-counting scan is robust to that:
    it returns each complete object so the caller can pick the one that parses
    into a valid decision (we try the LAST one first — models put the final
    answer last)."""
    objs, depth, start = [], 0, None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None
    return list(reversed(objs))


def probe_decision(cand, state, rec=None, now=None):
    """Single source of truth for "should we submit this candidate?". Returns a
    dict {verdict, est_real, champ, bar, urgency, reason}. Both the controller
    state message (which nudges the LLM to pick SUBMIT) and the SUBMIT guardrail
    consult this, so they never disagree.

    verdicts (LOWER score = better):
      clear_win - est_real beats champion by > 0.5*rmse: always submit.
      probe     - merely tied/marginal, but full-coverage + unsubmitted hash, and
                  EITHER the time-decayed bar permits it (urgency rises toward the
                  Beijing reset) OR there is exploration budget left (clock-
                  independent probing within ~1*rmse of champion, to exploit the
                  known headroom to the global best and sharpen the suite->real map).
      hold      - valid but not worth a submission yet (GENERATE a stronger one).
      refuse    - clearly worse than the champion (hard floor) or unsubmittable.

    out["probe_kind"] is "clear_win" | "time_decay" | "explore" when verdict is
    a submit, so the caller can charge exploration probes against _EXPLORE_CAP."""
    out = {"verdict": "hold", "est_real": None, "champ": None,
           "bar": None, "urgency": 0.0, "reason": "", "probe_kind": None}
    if not (cand and cand.get("gate_ok")):
        out["verdict"] = "refuse"; out["reason"] = "no valid candidate"; return out
    rec = rec if rec is not None else reconcile.load_model()
    if not reconcile.is_reliable(rec):
        out["reason"] = "recon map not reliable"; return out
    er = reconcile.estimate_real(cand["local_score"], rec)
    champ = state.get("best_real_score")
    champ = champ if champ is not None else float("inf")
    urgency = submit_urgency(state, now)
    out.update(est_real=er, champ=champ, urgency=urgency)
    if er is None:
        out["reason"] = "no est_real"; return out
    rmse = rec["rmse"]
    # Clear win — independent of urgency.
    if er < champ - 0.5 * rmse:
        out["verdict"] = "clear_win"; out["probe_kind"] = "clear_win"
        out["reason"] = f"est_real {er:.1f} beats champ {champ:.1f} by >0.5*rmse"
        return out
    # Hard floor — never probe something clearly worse.
    if er > champ + 1.0 * rmse:
        out["verdict"] = "refuse"
        out["reason"] = f"est_real {er:.1f} clearly worse than champ {champ:.1f}"
        return out
    # Time-decayed probe bar: slides from "clear win only" (urgency 0) toward
    # "accept marginal full-coverage ties" as the Beijing day runs out.
    bar = champ + (-0.5 + urgency * _URGENCY_U_SPAN) * rmse
    out["bar"] = bar
    remaining = state.get("daily_remaining", DAILY_LIMIT)
    in_final = _seconds_to_beijing_midnight(now) <= _FINAL_WINDOW_S
    has_budget = remaining > _PROBE_RESERVE or in_final
    champ_cov_min, _ = _champion_local_coverage()
    full_cov = _candidate_full_coverage(cand.get("gate_info"), min_cov=champ_cov_min)
    fresh = cand.get("h") not in _submitted_hashes()
    # Exploration budget: clock-independent probing of fresh champion-level-coverage
    # candidates within _EXPLORE_WINDOW*rmse of the champion. Capped per day.
    explore_left = _EXPLORE_CAP - state.get("explore_used", 0)
    out["explore_left"] = explore_left
    time_ok = er <= bar
    explore_ok = explore_left > 0 and er <= champ + _EXPLORE_WINDOW * rmse
    if full_cov and fresh and has_budget and (time_ok or explore_ok):
        out["verdict"] = "probe"
        if time_ok:
            out["probe_kind"] = "time_decay"
            out["reason"] = (f"est_real {er:.1f} <= probe bar {bar:.1f} "
                             f"(urgency {urgency:.2f}); full-coverage, unsubmitted")
        else:
            out["probe_kind"] = "explore"
            out["reason"] = (f"EXPLORATION probe ({explore_left} left): est_real "
                             f"{er:.1f} within {_EXPLORE_WINDOW:.0f}*rmse of champ "
                             f"{champ:.1f}; full-coverage, unsubmitted. Headroom to "
                             f"global best (699) + suite under-rates full-coverage.")
    else:
        why = []
        if not (time_ok or explore_ok):
            why.append(f"est_real {er:.1f} > probe bar {bar:.1f}"
                       + ("" if explore_left > 0 else "; explore budget spent"))
        if not full_cov: why.append("not full-coverage")
        if not fresh: why.append("already submitted")
        if not has_budget: why.append(f"holding reserve ({remaining} left)")
        out["reason"] = "; ".join(why) or "hold"
    return out


def controller_decide(state_text, api_key, model):
    """Ask the controller LLM for the next action. Robustly parse the JSON
    {thought, action, focus}. On any failure, default to GENERATE so the loop
    always makes progress."""
    messages = [{"role": "user", "content": state_text
                 + "\n\nChoose the next action as a JSON object."}]
    try:
        out = call_llm(model, messages, api_key, max_tokens=400,
                       system=CONTROLLER_SYSTEM_PROMPT, attempts=3,
                       log_tag="controller")
    except Exception as e:
        print(f"  controller unavailable ({e}) — defaulting to GENERATE")
        return {"thought": "(controller offline)", "action": "GENERATE", "focus": ""}
    # Try each balanced {...} object (last first); accept the first that parses
    # into a valid action. Robust to reasoning-model prose around the answer.
    for blob in _extract_json_objects(out):
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        act = str(d.get("action", "")).upper().strip()
        if act in _ACTIONS:
            return {"thought": str(d.get("thought", ""))[:300],
                    "action": act, "focus": str(d.get("focus", ""))[:300]}
    # Fallback: look for a bare action keyword, else GENERATE.
    for a in _ACTIONS:
        if a in out.upper():
            return {"thought": "(unparsed)", "action": a, "focus": ""}
    return {"thought": "(unparsed)", "action": "GENERATE", "focus": ""}


def build_controller_state(state, cand, run_trace, best_full=None):
    """Compact state text for the controller: budget, champion, reconcile map,
    current best candidate (local + est_real + weakest case), approaches tried,
    and the last few action->observation entries.

    `cand` is the best-LOCAL candidate (may be sub-coverage; used for INSPECT and
    generation feedback). `best_full` is the best FULL-COVERAGE candidate this run
    — the only thing actually submittable — and drives the submit verdict."""
    rec = reconcile.load_model()
    is_http = os.environ.get("JUDGE_MODE", "local") == "http"
    hrs_left = _seconds_to_beijing_midnight() / 3600.0
    urgency = submit_urgency(state)
    lines = ["=== AGENT STATE ==="]
    if is_http:
        explore_left = max(0, _EXPLORE_CAP - state.get("explore_used", 0))
        lines.append(f"Budget: {state['daily_remaining']}/{DAILY_LIMIT} submissions "
                     f"left today (used {state['submissions_used']}). "
                     f"{hrs_left:.1f}h to Beijing reset; submit-urgency={urgency:.2f} "
                     f"(unused submissions are LOST at reset). "
                     f"Exploration probes left today: {explore_left}/{_EXPLORE_CAP}.")
    else:
        lines.append("Mode: LOCAL (no real judge). SUBMIT is a no-op — do NOT "
                     "choose SUBMIT; only GENERATE/INSPECT/STOP make progress.")
    lines.append(f"Champion: real_score={state.get('best_real_score')} "
                 f"local(suite-avg)={state.get('best_local')}  (LOWER is better). "
                 f"GLOBAL BEST across all teams = 699 (~32 below champion) — it IS "
                 f"beatable; full-coverage solvers are under-rated locally, so SUBMIT "
                 f"to find real wins.")
    if reconcile.is_reliable(rec):
        lines.append(f"Suite->real map: real ~= {rec['a']:.3f}*pred + {rec['b']:.1f} "
                     f"(rmse={rec['rmse']:.0f}). margin to beat champion ~= "
                     f"{0.3*rec['rmse']:.0f}.")
    else:
        lines.append("Suite->real map: not reliable yet.")

    if cand is None:
        lines.append("\nCurrent best candidate THIS run: none yet (GENERATE one).")
    elif not cand.get("gate_ok"):
        lines.append("\nCurrent best candidate: INVALID locally (not submittable).")
    else:
        ls = cand["local_score"]
        er = reconcile.estimate_real(ls, rec) if reconcile.is_reliable(rec) else None
        st = cand["gate_info"]["stats"]
        lines.append(f"\nCurrent best candidate [hash {cand['h']}] algo='{cand['algo'][:60]}'")
        lines.append(f"  local(suite-avg)={ls:.1f}  est_real="
                     f"{('%.1f' % er) if er is not None else 'n/a'}  "
                     f"covered={st.get('covered')}/{st.get('total_tasks')}")
        lines.append(inspect_candidate(cand["gate_info"]))

    # --- Submit verdict is based on the best FULL-COVERAGE candidate, since only
    # full-coverage solvers are submittable (a dropped task loses on the real,
    # coverage-first judge). This is a DIFFERENT candidate from `cand` whenever a
    # lower-local sub-coverage solver exists. ---
    champ = state.get("best_real_score")
    if is_http and best_full and best_full.get("gate_ok") and reconcile.is_reliable(rec):
        bf_ls = best_full["local_score"]
        bf_er = reconcile.estimate_real(bf_ls, rec)
        pd = probe_decision(best_full, state, rec)
        same = cand is not None and best_full.get("h") == cand.get("h")
        tag = "" if same else f" [hash {best_full['h']}]"
        lines.append(f"\nBest SUBMITTABLE (full-coverage) candidate{tag}: "
                     f"local={bf_ls:.1f} est_real="
                     f"{('%.1f' % bf_er) if bf_er is not None else 'n/a'}")
        if champ is not None:
            if pd["verdict"] == "clear_win":
                lines.append(f"  >>> CLEARLY BEATS champion ({bf_er:.1f} vs {champ:.1f})."
                             f" SUBMIT it now. <<<")
            elif pd["verdict"] == "probe" and pd.get("probe_kind") == "explore":
                lines.append(f"  >>> EXPLORATION PROBE worth spending "
                             f"({pd.get('explore_left')} explore-probes left): "
                             f"full-coverage, near champion ({bf_er:.1f} vs {champ:.1f}). "
                             f"The suite under-rates full-coverage solvers and the global "
                             f"best (699) proves there's headroom — SUBMIT to get real "
                             f"feedback and sharpen the map. <<<")
            elif pd["verdict"] == "probe":
                lines.append(f"  >>> PROBE-WORTHY: only tied with champion "
                             f"({bf_er:.1f} vs {champ:.1f}) BUT it is full-coverage and "
                             f"the suite under-rates full-coverage solvers, so its real "
                             f"score may be better. {state['daily_remaining']} "
                             f"submissions will be LOST at reset — SUBMIT this probe. <<<")
            elif pd["verdict"] == "refuse":
                lines.append(f"  (worse than champion {champ:.1f}; do NOT submit — "
                             f"GENERATE a better full-coverage one.)")
            else:
                lines.append(f"  (not yet better than champion {champ:.1f}: "
                             f"{pd['reason']}. GENERATE a stronger full-coverage one.)")
    elif is_http and reconcile.is_reliable(rec):
        lines.append("\nBest SUBMITTABLE (full-coverage) candidate: none yet this run "
                     "— GENERATE a FULL-COVERAGE solver (sub-coverage cannot be submitted).")

    tried = archive._approaches_tried(archive.load_history())
    if tried:
        lines.append("\nApproaches tried (algorithm -> outcome):")
        for algo, _, outcome in tried[:8]:
            lines.append(f"  - {algo[:55]}: {outcome}")

    if run_trace:
        lines.append("\nRecent actions this run:")
        for t in run_trace[-4:]:
            lines.append(f"  {t['action']}: {t['obs'][:90]}")
    return "\n".join(lines)


def run_agent(model, iterations, api_key, case_name, n_candidates=2):
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

    # Suite -> real reconciliation: how well the local suite avg predicts the
    # real judge avg. When reliable, the gate decides in real units.
    _rec = reconcile.load_model()
    if reconcile.is_reliable(_rec):
        print(f"Suite->real map: real ~= {_rec['a']:.3f}*pred + {_rec['b']:.1f} "
              f"(R2={_rec['r2']:.2f}, rmse={_rec['rmse']:.0f}, n={_rec['n']}) "
              f"— gate decides in REAL units.")
    else:
        print("Suite->real map: not reliable yet — gate uses raw suite scores.")

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
    last_submitted_gate = float("inf")  # gate-value of the last solver submitted this run

    for it in range(iterations):
        print(f"\n{'='*60}\nITERATION {it}\n{'='*60}")

        # 1+2) BEST-OF-N: generate N candidates (temperature-diversified when
        # N>1), gate each locally, and keep the best LOCAL one. This is free
        # exploration — only the winner is considered for a real submission.
        def _parses(raw):
            try:
                ast.parse(strip_fences(raw)); return True
            except Exception:
                return False
        cands = []
        for k in range(n_candidates):
            temp = None if n_candidates == 1 else 0.8
            try:
                raw_k = call_claude_until_complete(model, messages, api_key,
                                                   _parses, temperature=temp,
                                                   log_tag="solver")
            except Exception as e:
                print(f"  candidate {k+1}/{n_candidates}: API unavailable: {e}")
                continue
            if not _parses(raw_k):
                print(f"  candidate {k+1}/{n_candidates}: truncated/unparseable — discarded.")
                continue
            code_k = wrap_with_fallback(strip_fences(raw_k))
            h_k = code_hash(code_k)
            algo_k = archive.parse_algorithm(raw_k)
            archive.archive_solver(h_k, code_k)
            gok_k, ginfo_k = run_local_gate(code_k)
            ls_k = _local_score(ginfo_k["stats"]) if gok_k else float("inf")
            print(f"  cand {k+1}/{n_candidates} [hash {h_k}] {algo_k[:38]} -> "
                  f"{'valid pred=%.1f' % ls_k if gok_k else 'INVALID'}")
            cands.append(dict(raw=raw_k, code=code_k, h=h_k, algo=algo_k,
                              gate_ok=gok_k, gate_info=ginfo_k, local_score=ls_k))
        if not cands:
            print("  all candidates failed to generate — skipping iteration.")
            continue
        chosen = min(cands, key=lambda c: c["local_score"])
        raw, code, h, algo = chosen["raw"], chosen["code"], chosen["h"], chosen["algo"]
        gate_ok, gate_info = chosen["gate_ok"], chosen["gate_info"]
        local_score = chosen["local_score"]
        if len(cands) > 1:
            print(f"  best-of-{len(cands)}: picked [hash {h}] local={local_score:.1f}")

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
            sane = local_score > -1000  # real scores ~200-3000; guards a broken model
            # Decide in REAL units when the suite->real map is reliable: estimate
            # this solver's real score and require it to beat the real champion by
            # more than the proxy's noise (rmse). Otherwise fall back to comparing
            # the raw suite-predicted score against the champion's suite score.
            rec = reconcile.load_model()
            if reconcile.is_reliable(rec):
                gate_val = reconcile.estimate_real(local_score, rec)
                champ = state.get("best_real_score")
                champ = champ if champ is not None else float("inf")
                # Require the estimated improvement to exceed ~0.3x the proxy's
                # noise (rmse). The suite is a PESSIMISTIC proxy for good (full-
                # coverage) solvers — est_real over-estimates their real score —
                # so a moderate margin still lets clear local wins through while
                # skipping noise-level "improvements" (the old iter-4 waste).
                margin = max(5.0, 0.3 * rec["rmse"])
                unit = "est_real"
            else:
                gate_val = local_score
                champ = state.get("best_local")
                champ = champ if champ is not None else float("inf")
                margin = 1e-6
                unit = "local"
            bar = min(champ, last_submitted_gate)
            worth = gate_val is not None and gate_val < bar - margin
            if not sane:
                print(f"  LOCAL SCORE ABSURD ({local_score:.1f}) — skipping submission "
                      f"(calibrated model unreliable for this solver).")
            elif state["daily_remaining"] <= 0:
                print("  BUDGET EXHAUSTED — iterating locally only (no submission).")
            elif worth:
                print(f"  SUBMITTING to real judge ({state['daily_remaining']} left; "
                      f"{unit} {gate_val:.1f} < bar {bar:.1f} - margin {margin:.0f})...")
                result = submit_and_score(code, case_name)
                submitted = True
                state["submissions_used"] += 1
                state["daily_remaining"] = (
                    result.daily_remaining if result.daily_remaining is not None
                    else state["daily_remaining"] - 1)
                last_submitted_gate = gate_val
                if result.ok:
                    print(f"  REAL avg_penalty = {result.score:.4f} "
                          f"(success {result.accepted_orders}/10; "
                          f"predicted real ~{gate_val:.1f})")
                    if state["best_real_score"] is None or result.score < state["best_real_score"]:
                        state["best_real_score"] = result.score
                        state["best_local"] = local_score   # champion's suite-local bar
                        with open("best_solver.py", "w") as f:
                            f.write(code)
                        print(f"  *** NEW REAL BEST: {result.score:.4f} "
                              f"(saved best_solver.py)")
                else:
                    print(f"  REAL JUDGE rejected: {result.message[:120]}")
                save_state(state)
            else:
                gv = f"{gate_val:.1f}" if gate_val is not None else "n/a"
                print(f"  SKIP submission — {unit} {gv} not better than "
                      f"champion {bar:.1f} (margin {margin:.0f}).")

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

        # 5c) RECONCILE — refit the suite-predicted -> real-judge mapping with
        # the fresh (predicted, real) pair, so the gate judges in real units.
        if submitted and result.ok and result.score is not None:
            reconcile.recompute()

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
            # When the solver was INVALID, skip the coach — the deterministic
            # "fix these bugs" directive is more useful than vague strategy.
            if not gate_ok:
                directive = _default_directive(submitted, result, gate_info,
                                               prev_gate_ok=False)
            else:
                print("  COACH: analyzing results for next directive...")
                directive = coach_directive(
                    facts, _format_formula(calibrate.load_model()),
                    BASELINE_SCORE, state.get("best_real_score"),
                    api_key, model)
                if directive is None:       # coach failed -> deterministic fallback
                    directive = _default_directive(submitted, result, gate_info,
                                                   prev_gate_ok=True)
            feedback = (facts + "\n\n" + directive
                        + "\n\nHere is your previous code:\n\n" + code)
            messages.append({"role": "assistant", "content": code})
            messages.append({"role": "user", "content": feedback})
            # Prune but keep the original problem brief (msg 0) and the last
            # 6 messages (3 assistant + 3 user turns). When pruning would drop
            # the problem brief entirely, append a compact rules reminder.
            if len(messages) > 7:
                # Check if msg 0 (the opening prompt) would be kept
                # messages[:1] keeps it, messages[-6:] keeps last 6
                # So msg 0 is always kept. But if we grow further, ensure
                # the last user message always includes a reminder.
                messages = messages[:1] + messages[-6:]
                # Append rules reminder to the latest user message
                _REMINDER = (
                    "\n\n[REMINDER — critical rules:\n"
                    " (1) task_id_list_str must match input rows VERBATIM\n"
                    " (2) each courier ID used at most once in entire output\n"
                    " (3) each individual task in at most one tuple\n"
                    " (4) backups go in ONE tuple: (\"task\", [primary, backup])\n"
                    " (5) Write simple, correct code. Wrap complex parts in try/except.\n"
                    " (6) If your last solver was INVALID, FIX the bugs, don't restart.]")
                messages[-1]["content"] += _REMINDER

    print(f"\n{'='*60}\nDONE.")
    print(f"Best LOCAL solver: {best['score']:.1f} at iter {best['iter']} (best_solver.py)")
    print(f"Best REAL judge score: {state['best_real_score']}")
    print(f"Submissions used today: {state['submissions_used']}/{DAILY_LIMIT}")
    return best


def _check_solver_env():
    """Verify pulp + a runnable CBC are available locally. The champion is an
    EXACT ILP solver; if CBC can't run here, ILP solvers silently fall back to
    greedy and get mis-scored, so the agent would wrongly reject its best solver
    class. Warn loudly when that's the case."""
    try:
        import pulp
    except Exception:
        print("  [env] WARNING: pulp NOT installed — ILP solvers will be scored "
              "as their greedy fallback only. `pip install pulp`. The agent cannot "
              "evaluate its best (ILP) solver class until this is fixed.")
        return False
    try:
        p = pulp.LpProblem("t", pulp.LpMinimize)
        x = pulp.LpVariable("x", 0, 1, cat="Binary"); p += x; p += x >= 1
        p.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[p.status] != "Optimal":
            raise RuntimeError("CBC did not solve trivial LP")
        print("  [env] pulp+CBC OK — ILP solvers are scored faithfully locally.")
        return True
    except Exception as e:
        print(f"  [env] WARNING: pulp present but CBC cannot run ({type(e).__name__}: "
              f"{str(e)[:80]}). ILP solvers will mis-score as greedy fallback. "
              "Install an arch-matched CBC (e.g. `brew install cbc`) and point pulp "
              "at it. The agent is BLIND to ILP quality until fixed.")
        return False


def _print_startup(state):
    print(f"Submission budget: {state['daily_remaining']}/{DAILY_LIMIT} left today "
          f"(used {state['submissions_used']}); best real: {state['best_real_score']}")
    _check_solver_env()
    cm = calibrate.load_model()
    if cm and cm.calibrated:
        print(f"Local scoring: CALIBRATED ({cm.form}, penalty/task="
              f"{cm.penalty_per_task:.1f}).")
    rec = reconcile.load_model()
    if reconcile.is_reliable(rec):
        print(f"Suite->real map: real ~= {rec['a']:.3f}*pred + {rec['b']:.1f} "
              f"(rmse={rec['rmse']:.0f}) — submissions judged in REAL units.")
        rho = rec.get("rank_rho")
        if rho is not None:
            print(f"Local<->real RANK fidelity: rho={rho:.3f} over {rec.get('rank_n')} "
                  f"solvers (does local rank like real? 1.0=perfect).")
    _snapshot_calib()
    return cm


def _seed_messages(cm):
    past = archive.load_history()
    digest = archive.build_knowledge_digest(past)
    seed = archive.best_real_solver(past)
    if digest:
        print(f"Loaded knowledge from {len(past)} past attempts (agent-only).")
    opening = PROBLEM_BRIEF + _build_formula_desc(cm) + digest
    opening += "\n\n" + _SOLVER_GUIDANCE
    if seed and seed[2]:
        opening += (f"\n\nHere is the BEST solver so far (real avg_score="
                    f"{seed[1]:.4f}). It is the EXACT ILP champion — IMPROVE it "
                    f"toward the winning direction above (joint set-packing / exact "
                    f"repair / more restarts); do NOT rewrite it as plain greedy. "
                    f"Output the full module:\n\n" + seed[2])
    else:
        opening += "\n\nWrite the first version now."
    return [{"role": "user", "content": opening}]


def _clearly_beats(cand, state):
    """(worth_submitting: bool, est_real or None). True if the candidate is a
    clear win OR a time-decayed probe — i.e. the loop guard should SUBMIT rather
    than re-INSPECT an unchanged candidate."""
    pd = probe_decision(cand, state)
    return (pd["verdict"] in ("clear_win", "probe")), pd["est_real"]


def run_agent_react(model, max_steps, api_key, case_name, n_candidates=2):
    """Hybrid ReAct: an LLM controller chooses GENERATE/INSPECT/SUBMIT/STOP each
    step; the harness executes with hard budget/validity guardrails."""
    is_http = os.environ.get("JUDGE_MODE", "local") == "http"
    _SUBMIT_MODEL[0], _SUBMIT_MODEL[1] = model, api_key   # for do_submit's memory hook
    state = load_state()
    cm = _print_startup(state)
    messages = _seed_messages(cm)
    print("Mode: ReAct controller.")

    cand = None          # best-LOCAL valid candidate (may be sub-coverage)
    best_full = None     # best FULL-COVERAGE valid candidate — the submittable one
    run_trace = []
    last_inspected_hash = None   # candidate hash last INSPECTed (loop guard)
    last_gen_hash = None         # last GENERATEd hash (byte-identical repeat guard)
    stall = 0                    # GENERATE steps since best_full last improved
    consec_invalid = 0           # consecutive GENERATE steps that were all-INVALID
    last_target = None           # weakest-case name we last steered the solver at

    for step in range(max_steps):
        print(f"\n{'='*60}\nSTEP {step}\n{'='*60}")
        step_hash = step_algo = step_pred = step_cases = None
        if is_http and state["daily_remaining"] <= 0:
            print("  Budget exhausted — stopping."); break

        dec = controller_decide(
            build_controller_state(state, cand, run_trace, best_full),
            api_key, model)
        action, thought, focus = dec["action"], dec["thought"], dec.get("focus", "")
        print(f"  THOUGHT: {thought}")
        print(f"  ACTION: {action}" + (f"  | focus: {focus}" if focus else ""))

        # --- LOOP GUARD: redundant INSPECT (same unchanged candidate) gives no
        # new info. Override: submit the best FULL-COVERAGE candidate if it's
        # worth it, else generate. ---
        if action == "INSPECT":
            cur_h = cand["h"] if (cand and cand.get("gate_ok")) else None
            if cur_h is None or cur_h == last_inspected_hash:
                beats, er = _clearly_beats(best_full, state)
                if beats and is_http and state["daily_remaining"] > 0:
                    print(f"  [guard] best full-coverage candidate worth submitting "
                          f"(est_real {er:.1f}) -> SUBMIT")
                    action = "SUBMIT"
                else:
                    print("  [guard] candidate unchanged -> GENERATE a different approach")
                    action = "GENERATE"
                    focus = focus or "try a DIFFERENT algorithm to cut the weakest case"

        if action == "STOP":
            run_trace.append({"action": "STOP", "obs": "controller stopped"}); break

        elif action == "INSPECT":
            obs = (inspect_candidate(cand["gate_info"])
                   if cand and cand.get("gate_ok")
                   else "no valid candidate yet — GENERATE one first.")
            last_inspected_hash = cand["h"] if (cand and cand.get("gate_ok")) else None
            print("  OBS:\n" + obs)

        elif action == "SUBMIT":
            # Only the best FULL-COVERAGE candidate is submittable — a dropped
            # task loses on the coverage-first real judge, and est_real for a
            # sub-coverage solver is untrustworthy.
            if not is_http:
                obs = "local mode: no real judge (no-op); keep GENERATEing."
            elif best_full is None or not best_full.get("gate_ok"):
                obs = ("REFUSED: no FULL-COVERAGE candidate to submit yet — "
                       "GENERATE a solver that covers every task.")
            elif state["daily_remaining"] <= 0:
                obs = "REFUSED: no submissions left."
            else:
                ls = best_full["local_score"]
                pd = probe_decision(best_full, state)
                # Allow clear wins always; allow probes under the time-decayed
                # bar; refuse only clearly-worse candidates.
                if pd["verdict"] == "refuse" and pd["est_real"] is not None:
                    obs = f"REFUSED by guardrail: {pd['reason']}."
                else:
                    if pd["verdict"] == "probe":
                        print(f"  [probe:{pd.get('probe_kind')}] {pd['reason']}")
                    # Charge exploration probes against the daily cap (so they
                    # can't drain the whole quota). do_submit persists state.
                    if pd.get("probe_kind") == "explore":
                        state["explore_used"] = state.get("explore_used", 0) + 1
                    res = do_submit(best_full["code"], ls, state, case_name)
                    obs = (f"real={res.score:.2f}" if res.ok
                           else f"rejected: {res.message[:80]}")
                    best_full = None  # consumed; force re-evaluation next
            print("  OBS: " + obs)

        else:  # GENERATE (default / fallback)
            # --- ANTI-STALL: break monotone loops before generating. ---
            if consec_invalid >= 3:
                seed = archive.best_real_solver(archive.load_history())
                if seed and seed[2]:
                    focus = ("Your last 3+ solvers were INVALID. STOP inventing new "
                             "algorithms. Take the known-good CHAMPION solver and make "
                             "ONE small, safe change; output simple, valid Python that "
                             "keeps FULL coverage.")
                    print("  [anti-stall] 3+ consecutive INVALID -> minimal-change-from-seed")
            elif stall >= 4:
                focus = (f"No full-coverage improvement in {stall} attempts. "
                         + (f"STOP targeting {last_target} (likely at its cost floor). "
                            if last_target else "")
                         + "Pick a DIFFERENT case and a FUNDAMENTALLY different algorithm "
                           "family than your recent attempts. Keep FULL coverage on every case.")
                print(f"  [anti-stall] stall={stall} -> diversify "
                      f"(away from {last_target})")

            chosen = generate_candidates(model, messages, api_key, n_candidates, focus)
            if chosen is None:
                obs = "all candidates failed to generate (API)."
                consec_invalid += 1
                stall += 1
            else:
                step_hash = chosen["h"]
                step_algo = chosen["algo"]
                step_pred = chosen["local_score"] if chosen["gate_ok"] else None
                step_cases = chosen["gate_info"].get("cases")
                # Promote to `cand` (best-LOCAL, used for INSPECT/feedback) only if
                # it improves and doesn't regress coverage unnecessarily.
                _champ_cov_min, _champ_tot = _champion_local_coverage()
                new_full = _candidate_full_coverage(chosen["gate_info"],
                                                    min_cov=_champ_cov_min)
                cur_full = cand is not None and cand.get("gate_ok") and \
                    _candidate_full_coverage(cand["gate_info"], min_cov=_champ_cov_min)
                if chosen["gate_ok"] and (
                        cand is None or not cand.get("gate_ok")
                        or (chosen["local_score"] < cand["local_score"]
                            and (new_full or not cur_full))):
                    cand = chosen
                # Track the best champion-level-coverage candidate — the only
                # thing actually submittable. This is what unblocks the probe path
                # when a lower-local sub-coverage solver also exists.
                improved_full = False
                if chosen["gate_ok"] and new_full and (
                        best_full is None
                        or chosen["local_score"] < best_full["local_score"]):
                    best_full = chosen
                    improved_full = True
                    _bf_cov = chosen["gate_info"].get("stats", {}).get("covered", "?")
                    print(f"  *** new best submittable candidate cov={_bf_cov}/{_champ_tot} "
                          f"local={chosen['local_score']:.1f} [hash {chosen['h']}]")
                # Update stall / invalid counters + last-targeted case.
                if not chosen["gate_ok"]:
                    consec_invalid += 1
                else:
                    consec_invalid = 0
                    lbl = _weakest_case_label(chosen["gate_info"].get("cases"))
                    if lbl:
                        last_target = lbl.split("=")[0]
                stall = 0 if improved_full else stall + 1
                if chosen["gate_ok"]:
                    st = chosen["gate_info"]["stats"]
                    cov_tag = "" if new_full else "  [COVERAGE REGRESSION]"
                    obs = (f"valid local={chosen['local_score']:.1f} "
                           f"covered={st['covered']}/{st['total_tasks']}{cov_tag}")
                else:
                    obs = "generated solver INVALID locally."
                # Anti-repeat: if the solver produced a byte-identical solver to
                # last GENERATE, push hard for a different algorithm family.
                repeat = chosen["gate_ok"] and chosen["h"] == last_gen_hash
                last_gen_hash = chosen["h"]
                next_user = "[controller] " + (focus or "improve further.")
                if repeat:
                    next_user += ("\n\nWARNING: this is BYTE-IDENTICAL to your "
                                  "previous solver — you are repeating yourself. "
                                  "The next attempt MUST be a fundamentally "
                                  "different algorithm, not a tweak.")
                if not new_full and chosen["gate_ok"]:
                    next_user += ("\n\nNOTE: this solver REGRESSED coverage (left a "
                                  "task uncovered). A dropped task costs ~100 and "
                                  "can never beat a full-coverage champion. Restore "
                                  "FULL coverage first, then cut cost.")
                messages.append({"role": "assistant", "content": chosen["code"]})
                messages.append({"role": "user", "content": next_user})
                if len(messages) > 7:
                    messages = messages[:1] + messages[-6:]
            print("  OBS: " + obs)

        run_trace.append({"action": action, "obs": obs})
        with open("run_log.jsonl", "a") as f:
            f.write(json.dumps({"mode": "react", "step": step, "action": action,
                                "thought": thought, "focus": focus, "obs": obs[:300],
                                "hash": step_hash, "algorithm": step_algo,
                                "predicted_score": step_pred, "gate_cases": step_cases,
                                "daily_remaining": state["daily_remaining"],
                                "ts": time.time()}) + "\n")

        # Long-stall STOP: if we've gone many GENERATEs with no full-coverage
        # improvement, the run is stuck — end it rather than burn more API calls.
        if stall >= 18:
            print(f"  [anti-stall] no full-coverage improvement in {stall} attempts "
                  f"— stopping this run."); break

    print(f"\n{'='*60}\nDONE (ReAct).")
    print(f"Best REAL judge score: {state['best_real_score']}")
    print(f"Submissions used today: {state['submissions_used']}/{DAILY_LIMIT}")


# ===========================================================================
# Judge-driven SUBMIT mode: deliberate hard, spend a real submission, then learn
# from the REAL per-case feedback (not the local proxy). The local gate is used
# ONLY as a sanity filter (valid + full-coverage + fresh hash) so a submission is
# never wasted on broken/sub-coverage/duplicate code.
# ===========================================================================
_SOLVER_GUIDANCE = (
    "WINNING DIRECTION (the current champion is an EXACT ILP solver using "
    "pulp/CBC — that is the best approach, build on it, do NOT downgrade to pure "
    "greedy):\n"
    "  - The true per-task-group cost is NONLINEAR: "
    "P(any accept)*E(score|accept) + P(no accept)*100*num_tasks, where "
    "P(no accept)=prod(1-willingness). A linear primary cost is only an "
    "approximation — prefer optimizing the TRUE expected penalty.\n"
    "  - HIGHEST-VALUE upgrade: a SET-PACKING / column-generation ILP where each "
    "column = (task_str, a subset of <=3 couriers) priced by its EXACT expected "
    "penalty; pick columns so each task has <=1 column and each courier is used "
    "<=1 time. This optimizes primary+backups JOINTLY (beats the two-stage "
    "primary-then-backup funnel).\n"
    "  - Also strong: exact small-neighbourhood repair (destroy the ~8 worst-cost "
    "tasks + their candidate couriers, re-solve that sub-ILP exactly, reinsert; "
    "iterate) and MANY restarts within the time budget keeping the best by the "
    "true expected penalty.\n"
    "HARD REQUIREMENTS:\n"
    "  - Each courier_id may appear AT MOST ONCE across the whole solution.\n"
    "  - Wrap the ILP so it ALWAYS returns a valid full-coverage solution even if "
    "CBC is unavailable/slow (greedy fallback) — never return invalid/empty.\n"
    "  - Call random.seed(12345) at the start of solve() so scoring is "
    "DETERMINISTIC (otherwise real improvements can't be told apart from noise).\n"
    "  - Keep total runtime under ~9s; give CBC a per-solve timeLimit.")


def _anchor_focus(weak_case=None, real_hint=None):
    """Champion-anchored generation directive (folds in the Round-3 lesson): edit
    the champion shown at the top of the conversation, don't reinvent."""
    tgt = f" Target the worst case: {weak_case}." if weak_case else ""
    rh = f" {real_hint}" if real_hint else ""
    return ("REQUIREMENT #1 — FULL COVERAGE: your solver MUST cover every single "
            "task (all 40/40 on syn_scarce and every other case). A solver that "
            "drops even ONE task is REJECTED regardless of its cost score. If your "
            "previous attempt dropped tasks on syn_scarce, fix coverage FIRST by "
            "ensuring every task gets at least one valid courier assigned.\n"
            "REQUIREMENT #2 — improve cost: Improve the CHAMPION solver shown at "
            "the top of this conversation with ONE surgical change that lowers "
            f"cost.{tgt}{rh} " + _SOLVER_GUIDANCE)


def _weak_case_from_result(res):
    """Worst REAL case name from a ScoreResult's per-case detail (failed/invalid
    first, then highest cost).  Returns None when unavailable OR when the real
    judge doesn't expose per-case scores (score=None for all) — in that situation
    the caller should fall back to local gate_info scores instead."""
    crs = getattr(res, "case_results", None) if res else None
    if not crs:
        return None
    worst = None
    any_real_score = False
    for c in crs:
        name = c.get("case_file")
        if name is None:
            continue
        bad = c.get("status") != "ok" or c.get("validity") is False
        score = c.get("penalty_score") if bad else c.get("score")
        if score is not None:
            any_real_score = True
        else:
            tot = c.get("total_tasks") or 0
            score = tot * 100 if bad else None  # don't fabricate 0 for ok+None
        if score is None:
            continue
        key = (1 if bad else 0, score)
        if worst is None or key > worst[0]:
            worst = (key, name)
    # If no real scores were revealed (real judge returns score=None for all cases),
    # return None so the caller can use the local gate_info as a fallback — picking
    # a random case from the score=None list (previously high_noise_seed601.txt, the
    # *best* local case) is worse than no guidance at all.
    if not any_real_score:
        return None
    return worst[1] if worst else None


def _weakest_local_case(gate_info):
    """Case name with the highest local predicted_score (worst cost), from gate_info.
    This is the fallback when the real judge doesn't expose per-case scores."""
    cases = (gate_info or {}).get("cases", {})
    if not cases:
        return None
    worst = None
    for cname, cinfo in cases.items():
        st = cinfo.get("stats", {})
        pred = st.get("predicted_score")
        if pred is None:
            continue
        if worst is None or pred > worst[0]:
            worst = (pred, cname)
    return worst[1] if worst else None


def run_agent_submit(model, max_submits, api_key, case_name,
                     n_candidates=4, deliberate_rounds=3):
    """Heavy-deliberation, judge-driven loop. For each of a few submissions:
    deliberate (generate N candidates + critique/refine, R rounds), keep the best
    locally-valid FULL-COVERAGE fresh candidate, SUBMIT it to the real judge, then
    let the REAL per-case feedback drive the next deliberation."""
    is_http = os.environ.get("JUDGE_MODE", "local") == "http"
    if not is_http:
        print("ERROR: --mode submit requires JUDGE_MODE=http — it spends REAL "
              "judge submissions for ground-truth feedback. Set "
              "`export JUDGE_MODE=http` and retry."); return
    _SUBMIT_MODEL[0], _SUBMIT_MODEL[1] = model, api_key   # for do_submit's memory hook
    state = load_state()
    cm = _print_startup(state)
    messages = _seed_messages(cm)

    # Determine champion's local coverage — used as the min-coverage gate.
    # The champion (real=731.48) itself only covers 289/291 locally because
    # syn_scarce tops out at 38/40 for greedy solvers (the 2 un-coverable tasks
    # require ILP which costs ~200 pts extra).  Requiring 291/291 blocks every
    # greedy candidate.  Instead we require ≥ champion_cov (289).
    champ_cov, champ_tot = _champion_local_coverage()
    if champ_cov is None:
        champ_cov = 0   # no champion yet — accept any valid solver
    print(f"Mode: JUDGE-DRIVEN submit — deliberate {deliberate_rounds} round(s) x "
          f"{n_candidates} candidate(s) per submission; gate = valid + "
          f"cov≥{champ_cov}/{champ_tot or '?'} (champion's local coverage) "
          f"+ fresh hash (NOT 'beats champion locally').")

    history = []
    last_result = None
    submitted_h = _submitted_hashes()
    # Track the best local pred score we've actually submitted so we don't waste
    # submissions on near-identical solvers that score the same real value.
    # Each new submission must improve by at least this margin (in local pred units).
    _MIN_PRED_IMPROVEMENT = 0.3
    last_submitted_pred = None   # set after each submission

    for sub in range(max_submits):
        if state["daily_remaining"] <= 0:
            print("  Budget exhausted — stopping."); break
        print(f"\n{'='*60}\nSUBMISSION {sub+1}/{max_submits}  "
              f"(budget {state['daily_remaining']}/{DAILY_LIMIT})\n{'='*60}")

        best = None   # best locally-valid champion-coverage fresh candidate this cycle
        # Require meaningful pred improvement over the last submission so we don't
        # waste a submission on a near-identical solver that maps to the same real score.
        pred_threshold = (last_submitted_pred - _MIN_PRED_IMPROVEMENT
                          if last_submitted_pred is not None else None)
        for r in range(deliberate_rounds):
            # Weak case: prefer REAL per-case signal; fall back to local gate_info
            # when the real judge returns score=None for all cases (which it currently
            # does — in that case _weak_case_from_result returns None to avoid
            # falsely pointing at the alphabetically-first case, e.g. high_noise,
            # which is actually the BEST local case, not the worst).
            weak = _weak_case_from_result(last_result)
            if weak is None:
                # Use local gate_info: highest predicted_score = costliest case
                ref_gi = (best or {}).get("gate_info") if best else None
                weak = _weakest_local_case(ref_gi)
            focus = _anchor_focus(weak)
            print(f"  [deliberate {r+1}/{deliberate_rounds}]"
                  + (f" weak={weak}" if weak else ""))
            chosen = generate_candidates(model, messages, api_key, n_candidates, focus)
            if (chosen and chosen["gate_ok"]
                    and _candidate_full_coverage(chosen["gate_info"], min_cov=champ_cov)
                    and chosen["h"] not in submitted_h
                    and (pred_threshold is None
                         or chosen["local_score"] < pred_threshold)):
                if best is None or chosen["local_score"] < best["local_score"]:
                    best = chosen
                    _cov = chosen["gate_info"].get("stats", {}).get("covered", "?")
                    print(f"    -> new best cov={_cov}/{champ_tot} local="
                          f"{chosen['local_score']:.1f} [hash {chosen['h']}]")
            # Critique/refine: feed the current best's per-case breakdown back and
            # ask for ONE surgical improvement (this is the "think longer" loop).
            base = best or chosen
            if base:
                bd = (inspect_candidate(base["gate_info"]) if base.get("gate_ok")
                      else "your last solver was INVALID locally — fix it")
                messages.append({"role": "assistant", "content": base["code"]})
                base_cov = (base["gate_info"].get("stats", {}).get("covered", 0)
                            if base.get("gate_ok") else 0)
                if not base.get("gate_ok"):
                    cov_tag = " *** INVALID — fix the code first ***"
                elif base_cov < champ_cov:
                    cov_tag = (f" *** SUB-COVERAGE ({base_cov}/{champ_tot}) — "
                               f"must be ≥{champ_cov} (champion level) ***")
                else:
                    cov_tag = f" ✓ coverage {base_cov}/{champ_tot} (≥ champion)"
                # Tell the model what local pred score it needs to beat
                pred_bar_msg = ""
                if pred_threshold is not None:
                    pred_bar_msg = (f"\nIMPROVEMENT REQUIRED: your solver must score "
                                    f"pred < {pred_threshold:.1f} locally (last submission "
                                    f"was {last_submitted_pred:.1f}; need ≥"
                                    f"{_MIN_PRED_IMPROVEMENT:.1f} improvement). "
                                    f"Solvers scoring ≥{pred_threshold:.1f} will NOT "
                                    f"be submitted — they duplicate the last submission.")
                messages.append({"role": "user", "content":
                                 f"[deliberate] Per-case breakdown of your current best"
                                 f"{cov_tag}:\n" + bd
                                 + f"\n\nCOVERAGE REQUIREMENT: your solver must cover "
                                 f"≥{champ_cov} tasks (the champion covers {champ_cov}/{champ_tot}). "
                                 f"Note: syn_scarce 40/40 is NOT required — even the champion "
                                 f"only achieves 38/40 there. Do NOT use ILP/CP-SAT to force "
                                 f"40/40 — it costs ~200 pts extra.\n"
                                 + pred_bar_msg
                                 + "\nREQUIREMENT: ONE surgical cost improvement on the "
                                 "worst case above. " + _SOLVER_GUIDANCE})
                if len(messages) > 9:
                    messages = messages[:1] + messages[-8:]

        if best is None:
            # Distinguish coverage failure from pred-improvement failure
            if pred_threshold is not None:
                print(f"  No candidate with pred < {pred_threshold:.1f} (improvement "
                      f"over last submission {last_submitted_pred:.1f}) this cycle "
                      f"— NOT spending a submission. Pushing for more improvement.")
                # Find weakest local case from the last best candidate's gate_info
                _last_gi = None
                for _m in reversed(messages):
                    pass  # can't easily extract gate_info from messages; use champion's
                _champ_weak = _weakest_local_case(None)  # None -> returns None
                _weak_hint = _champ_weak or "syn_low_willing (score~1133) or syn_scarce (score~897)"
                messages.append({"role": "user", "content":
                                 f"[deliberate] NO IMPROVEMENT: all your candidates scored "
                                 f"pred ≥ {pred_threshold:.1f} locally, which maps to the "
                                 f"same real score as the previous submission. You must "
                                 f"achieve pred < {pred_threshold:.1f} to be worth submitting.\n"
                                 f"The costliest local cases are syn_low_willing (~1133) and "
                                 f"syn_scarce (~897). Make ONE targeted improvement on one of "
                                 f"those — reduce their cost without breaking coverage.\n"
                                 "Output the FULL module. " + _SOLVER_GUIDANCE})
            else:
                print(f"  No candidate with coverage ≥{champ_cov}/{champ_tot} this cycle "
                      f"— NOT spending a submission. Re-deliberating from champion.")
                messages.append({"role": "user", "content":
                                 f"[deliberate] COVERAGE GATE FAILED: none of your last "
                                 f"attempts covered ≥{champ_cov}/{champ_tot} tasks (the "
                                 f"champion's level). Most solvers scored "
                                 f"{champ_cov - 1}/{champ_tot} or worse.\n"
                                 f"CRITICAL: this is NOT about syn_scarce 40/40 — the champion "
                                 f"itself only gets 38/40 on syn_scarce. The issue is some OTHER "
                                 f"case is dropping tasks.\n"
                                 f"FIX: take the CHAMPION solver from the very top of this "
                                 f"conversation and make ONE minimal change. Do NOT rewrite from "
                                 f"scratch. Keep coverage ≥{champ_cov}.\n"
                                 "Output the FULL module. " + _SOLVER_GUIDANCE})
            if len(messages) > 9:
                messages = messages[:1] + messages[-8:]
            continue

        # ---- SUBMIT to the real judge ----
        print(f"  SUBMITTING best candidate local={best['local_score']:.1f} "
              f"cov={best['gate_info'].get('stats',{}).get('covered','?')}/{champ_tot} "
              f"[hash {best['h']}] algo='{best['algo'][:50]}'...")
        res = do_submit(best["code"], best["local_score"], state, case_name)
        submitted_h.add(best["h"])
        last_submitted_pred = best["local_score"]  # update threshold for next round
        last_result = res
        history.append({"iter": sub, "hash": best["h"], "gate_ok": True,
                        "submitted": True, "ok": res.ok,
                        "score": res.score if res.ok else float("inf"),
                        "predicted_score": best["local_score"],
                        "algo": best["algo"], "note": res.message})

        # ---- REFLECT on the REAL per-case feedback (the primary signal) ----
        facts = build_facts(history, res, True, best["gate_info"])
        print("  REAL JUDGE FEEDBACK:")
        for line in facts.splitlines():
            if line.strip():
                print("    " + line)
        directive = coach_directive(
            facts, _format_formula(calibrate.load_model()),
            BASELINE_SCORE, state.get("best_real_score"), api_key, model)
        if not directive:
            directive = ("Use the REAL per-case feedback above (it is ground truth, "
                         "unlike the local suite). Make ONE surgical change to fix the "
                         "worst REAL case next.")
        messages.append({"role": "assistant", "content": best["code"]})
        messages.append({"role": "user", "content":
                         facts + "\n\n" + directive + "\n\n" + _anchor_focus(
                             _weak_case_from_result(res),
                             "This is REAL judge feedback — trust it over local scores.")})
        if len(messages) > 9:
            messages = messages[:1] + messages[-8:]

    print(f"\n{'='*60}\nDONE (submit mode).")
    print(f"Best REAL judge score: {state['best_real_score']}")
    print(f"Submissions used today: {state['submissions_used']}/{DAILY_LIMIT}")


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
    ap.add_argument("--candidates", type=int, default=2,
                    help="best-of-N candidates generated per iteration (local search)")
    ap.add_argument("--mode", choices=["react", "linear", "submit"], default="react",
                    help="react: LLM controller chooses actions; linear: fixed loop; "
                         "submit: judge-driven — deliberate hard then spend real "
                         "submissions and learn from REAL per-case feedback")
    ap.add_argument("--max-submits", type=int, default=6,
                    help="submit mode: max real submissions to spend this run "
                         "(also capped by the daily budget)")
    ap.add_argument("--deliberate-rounds", type=int, default=3,
                    help="submit mode: generate+critique rounds before each submission")
    ap.add_argument("--provider", choices=["longcat", "deepseek"], default="longcat",
                    help="LLM provider: longcat (default) or deepseek")
    ap.add_argument("--model", default=None,
                    help="model name (default: LongCat-2.0-Preview for longcat, "
                         "deepseek-chat for deepseek)")
    ap.add_argument("--case", default="large_seed301")
    ap.add_argument("--calibrate", action="store_true",
                    help="spend 1 submission on the reference solver to fit the "
                         "real cost formula, then exit")
    ap.add_argument("--consolidate", action="store_true",
                    help="distill run history into memory_brief.md, then exit")
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

    global _PROVIDER
    _PROVIDER = args.provider
    if _PROVIDER == "deepseek":
        api_key = os.environ.get("DEEPSEEK_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_KEY not found in environment or .env"); sys.exit(1)
        default_model = "deepseek-chat"
    else:
        api_key = os.environ.get("LONGCAT_KEY")
        if not api_key:
            print("ERROR: LONGCAT_KEY not found in environment or .env"); sys.exit(1)
        default_model = "LongCat-2.0-Preview"
    model = args.model or default_model

    if args.consolidate:
        memory.consolidate(call_llm, model, api_key)
        return

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

    print(f"AutoSolver Agent | provider={_PROVIDER} | model={model} | "
          f"judge_mode={os.environ.get('JUDGE_MODE','local')} | "
          f"mode={args.mode} | steps={args.iterations}")
    if args.mode == "submit":
        # Heavy deliberation defaults to >=4 candidates unless the user raised it.
        n_cand = max(args.candidates, 4)
        run_agent_submit(model, args.max_submits, api_key, args.case,
                         n_candidates=n_cand,
                         deliberate_rounds=args.deliberate_rounds)
    elif args.mode == "react":
        run_agent_react(model, args.iterations, api_key, args.case, args.candidates)
    else:
        run_agent(model, args.iterations, api_key, args.case, args.candidates)


if __name__ == "__main__":
    main()
