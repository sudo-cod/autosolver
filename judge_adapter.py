"""
judge_adapter.py
================
THE ONLY FILE YOU MUST CUSTOMIZE for your environment.

The agent needs one thing from the judge: given solver code (a string) or a
produced solution, return a numeric SCORE (lower = better, per the problem's
"minimize total score" objective). Everything else in the agent is generic.

You said you're not sure yet how hackathon.mykeeta.com takes submissions.
Three implementations are provided below. Pick one by setting JUDGE_MODE,
or wire your own. The agent only ever calls `submit_and_score(...)`.

ScoreResult contract:
    .score          float   the judge's objective value (lower is better).
                            If the judge reports higher-is-better, NEGATE it
                            here so the agent always minimizes.
    .accepted_orders int    optional, for logging/reflection.
    .raw            any     the raw judge response, for debugging.
    .ok             bool    did we get a valid score back?
    .message        str     human-readable status / error.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import subprocess
import tempfile
import os
import json
import sys
import time
import urllib.request
import urllib.error

# ---- choose how the judge is reached --------------------------------------
#   "http"    : hackathon.mykeeta.com JSON API (login → judge → poll result)
#               Required env vars: JUDGE_TEAM, JUDGE_EMAIL
#               Optional:         JUDGE_BASE (default https://hackathon.mykeeta.com)
#   "browser" : judge is a web form; we drive a headless browser (Playwright)
#   "manual"  : no automation possible; agent pauses and you paste the score
#   "local"   : no judge call at all; score locally from the data (offline dev)
HERE = os.path.dirname(__file__)
JUDGE_MODE = os.environ.get("JUDGE_MODE", "local")


@dataclass
class ScoreResult:
    ok: bool
    score: float = float("inf")          # lower is better; inf = failed
    accepted_orders: Optional[int] = None
    raw: Any = None
    message: str = ""
    case_results: Optional[list] = None  # full per-case breakdown from judge
    daily_remaining: Optional[int] = None  # submissions left today (judge-authoritative)


# ===========================================================================
# IMPLEMENTATION 1 — hackathon.mykeeta.com HTTP API
#   Flow: login → POST /judge → poll /result/{job_id}
#   Env vars: JUDGE_TEAM, JUDGE_EMAIL, JUDGE_BASE (optional)
# ===========================================================================
JUDGE_BASE = os.environ.get("JUDGE_BASE", "https://hackathon.mykeeta.com")
_cached_token: Optional[str] = None


def _http_post(url: str, payload: dict, timeout: int = 30) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _login() -> str:
    global _cached_token
    team = os.environ.get("JUDGE_TEAM", "")
    email = os.environ.get("JUDGE_EMAIL", "")
    if not team or not email:
        raise RuntimeError("JUDGE_TEAM and JUDGE_EMAIL must be set for http mode")
    data = _http_post(f"{JUDGE_BASE}/login", {"team": team, "email": email})
    if "token" not in data:
        raise RuntimeError(f"Login failed: {data}")
    _cached_token = data["token"]
    return _cached_token


def _submit_http(solver_code: str, case_name: str) -> ScoreResult:
    global _cached_token

    # 1. Authenticate
    try:
        token = _cached_token or _login()
    except Exception as e:
        return ScoreResult(ok=False, message=f"login error: {e}")

    # 2. Submit code; retry once if token is stale
    for attempt in range(2):
        try:
            resp = _http_post(f"{JUDGE_BASE}/judge", {"code": solver_code, "token": token})
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err = json.loads(body)
            except Exception:
                err = {}
            if err.get("error") == "today_limit_exceeded":
                return ScoreResult(ok=False, message="daily submission limit exceeded")
            if err.get("error") == "unauthorized" and attempt == 0:
                _cached_token = None
                try:
                    token = _login()
                except Exception as le:
                    return ScoreResult(ok=False, message=f"re-login failed: {le}")
                continue
            return ScoreResult(ok=False, message=f"submit error {e.code}: {body[:200]}")
        except Exception as e:
            return ScoreResult(ok=False, message=f"submit error: {e}")
        break

    job_id = resp.get("job_id")
    daily_remaining = resp.get("daily_remaining")
    if not job_id:
        return ScoreResult(ok=False, message=f"no job_id in response: {resp}")
    print(f"  [http] job_id={job_id}  daily_remaining={daily_remaining}")

    # 3. Poll for result
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(2)
        try:
            result = _http_get(f"{JUDGE_BASE}/result/{job_id}")
        except Exception as e:
            return ScoreResult(ok=False, message=f"poll error: {e}")
        status = result.get("status")
        if status in ("queued", "running"):
            continue
        if status == "ok":
            avg = float(result["avg_score"])
            success = result.get("success_count", 0)
            cases = result.get("case_count", 0)
            case_results = result.get("case_results", []) or []
            msg = (f"cases={cases} success={success} avg_score={avg:.4f}\n"
                   + _format_case_results(case_results))
            return ScoreResult(ok=True, score=avg, accepted_orders=success,
                               raw=result, message=msg,
                               case_results=case_results,
                               daily_remaining=daily_remaining)
        return ScoreResult(ok=False, message=f"judge returned status={status}: {result}")

    return ScoreResult(ok=False, message="timed out waiting for judge result")


def _format_case_results(case_results: list) -> str:
    """Render per-case judge feedback into lines the agent can reflect on."""
    lines = []
    for c in case_results:
        name = c.get("case_file", "?")
        status = c.get("status", "?")
        validity = c.get("validity")
        total = c.get("total_tasks")
        assigned = c.get("assigned", c.get("assigned_count"))
        elapsed = c.get("elapsed_ms")
        errors = c.get("errors") or []
        if status != "ok" or validity is False:
            penalty = c.get("penalty_score")
            if penalty is None and total is not None:
                penalty = total * 100
            tag = "INVALID" if validity is False else "ERROR"
            err = f"  err: {errors[0]}" if errors else ""
            lines.append(f"  [{tag}] {name}: penalty={penalty} "
                         f"assigned={assigned}/{total} {elapsed}ms{err}")
        else:
            score = c.get("score", c.get("total_score"))
            lines.append(f"  [ok]      {name}: score={score} "
                         f"assigned={assigned}/{total} {elapsed}ms")
    return "\n".join(lines)


# ===========================================================================
# IMPLEMENTATION 2 — Headless browser (web form)
# Use if the judge is a page where you paste code and read a score.
# Requires: pip install playwright && playwright install chromium
# Fill in the selectors (the agent will tell you exactly which to find).
# ===========================================================================
def _submit_browser(solver_code: str, case_name: str) -> ScoreResult:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return ScoreResult(ok=False, message=f"playwright not installed: {e}")

    JUDGE_URL = os.environ.get("JUDGE_URL", "https://hackathon.mykeeta.com/")
    # SELECTORS — adapt to the real page (devtools → inspect element):
    CODE_INPUT_SEL = os.environ.get("SEL_CODE", "textarea#code")
    SUBMIT_BTN_SEL = os.environ.get("SEL_SUBMIT", "button#submit")
    SCORE_SEL      = os.environ.get("SEL_SCORE", "#score")
    STORAGE_STATE  = os.environ.get("JUDGE_STORAGE", "")  # saved login cookies

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                storage_state=STORAGE_STATE if STORAGE_STATE else None)
            page = ctx.new_page()
            page.goto(JUDGE_URL, wait_until="networkidle")
            page.fill(CODE_INPUT_SEL, solver_code)
            page.click(SUBMIT_BTN_SEL)
            page.wait_for_selector(SCORE_SEL, timeout=60000)
            score_text = page.inner_text(SCORE_SEL)
            browser.close()
        score = float("".join(c for c in score_text if (c.isdigit() or c in ".-")))
        return ScoreResult(ok=True, score=score, raw=score_text, message="ok")
    except Exception as e:
        return ScoreResult(ok=False, message=f"browser judge error: {e}")


# ===========================================================================
# IMPLEMENTATION 3 — Manual relay
# No automation: the agent prints the code/score request and waits for you
# to submit on the website and type the score back. Slow but always works.
# ===========================================================================
def _submit_manual(solver_code: str, case_name: str) -> ScoreResult:
    print("\n" + "=" * 70)
    print(f"MANUAL SUBMISSION NEEDED for case: {case_name}")
    print("The current solver code has been written to: ./current_solver.py")
    print("1. Submit that file at https://hackathon.mykeeta.com/")
    print("2. Read the score the site reports.")
    print("=" * 70)
    with open("current_solver.py", "w") as f:
        f.write(solver_code)
    try:
        raw = input("Enter the score the judge reported (lower=better), or 'skip': ").strip()
        if raw.lower() == "skip":
            return ScoreResult(ok=False, message="skipped by user")
        return ScoreResult(ok=True, score=float(raw), raw=raw, message="manual")
    except Exception as e:
        return ScoreResult(ok=False, message=f"manual parse error: {e}")


# ===========================================================================
# IMPLEMENTATION 4 — Local scorer (offline development)
# Runs the solver code on a local case file and scores it with our best
# understanding of the objective. Lets you develop/test the whole agent loop
# WITHOUT the judge, then switch JUDGE_MODE to a real one for deployment.
# ===========================================================================
def _run_solver_locally(solver_code: str, input_text: str, timeout_s: int = 12):
    """Execute untrusted solver code in a subprocess; return its result list."""
    runner = f'''
import sys, json
{solver_code}

if __name__ == "__main__":
    data = sys.stdin.read()
    sol = solve(data)
    # normalize to list of [task_str, [couriers]]
    out = [[ts, list(cs)] for ts, cs in sol]
    print("<<<RESULT>>>" + json.dumps(out))
'''
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(runner)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], input=input_text,
                              capture_output=True, text=True, timeout=timeout_s)
        for line in proc.stdout.splitlines():
            if line.startswith("<<<RESULT>>>"):
                return json.loads(line[len("<<<RESULT>>>"):]), proc.stderr
        return None, (proc.stderr or "no result marker in output")
    except subprocess.TimeoutExpired:
        return None, f"solver exceeded {timeout_s}s time limit"
    except Exception as e:
        return None, f"solver crashed: {e}"
    finally:
        os.unlink(path)


def _parse_candidates(input_text: str):
    """Map each input row to its candidate: key=(task_id_list_str, courier_id)
    using the RAW input string (no re-sorting),
    value=(task_list, total_score, willingness).
    Also return the set of all distinct task ids seen."""
    cmap = {}
    all_tasks = set()
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    for line in lines[start:]:
        p = line.split("\t")
        if len(p) < 4:
            continue
        ts_str = p[0].strip()
        courier = p[1].strip()
        try:
            score = float(p[2])
            willingness = float(p[3])
        except ValueError:
            continue
        tasks = [t.strip() for t in ts_str.split(",")]
        cmap[(ts_str, courier)] = (tasks, score, willingness)
        all_tasks.update(tasks)
    return cmap, all_tasks


_COST_MODEL_CACHE = {"mtime": None, "model": None}
_COST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "cost_model.json")


def _get_cost_model():
    """Return the calibrated CostModel if cost_model.json exists and is marked
    calibrated, else None. Cached by file mtime so recalibration is picked up."""
    try:
        mtime = os.path.getmtime(_COST_MODEL_PATH)
    except OSError:
        _COST_MODEL_CACHE["mtime"] = None
        _COST_MODEL_CACHE["model"] = None
        return None
    if _COST_MODEL_CACHE["mtime"] != mtime:
        try:
            import calibrate  # lazy to avoid import cycle
            m = calibrate.load_model()
            _COST_MODEL_CACHE["model"] = m if (m and m.calibrated) else None
        except Exception:
            _COST_MODEL_CACHE["model"] = None
        _COST_MODEL_CACHE["mtime"] = mtime
    return _COST_MODEL_CACHE["model"]


def validate_solution(solution, input_text):
    """
    Mirror the judge's validity contract as closely as we know it.
    Returns (ok: bool, errors: list[str], stats: dict).

    Rules enforced:
      - each item must be (task_id_list_str, [courier_id, ...])
      - the (task_id_list_str, courier_id) pair must exist VERBATIM as an
        input row (NO re-sorting/normalizing the task string)
      - no courier reused, no task covered twice
    """
    cmap, all_tasks = _parse_candidates(input_text)
    cost_model = _get_cost_model()
    errors = []
    covered = set()
    total_score = 0.0
    predicted_cost = 0.0
    used_couriers = set()

    for i, item in enumerate(solution):
        # shape check
        try:
            ts, couriers = item
        except Exception:
            errors.append(f"item {i}: not a (task_str, [courier]) pair: {item!r}")
            continue
        if not isinstance(ts, str):
            errors.append(f"item {i}: task_id_list must be a string, got {type(ts).__name__}")
            continue
        if not isinstance(couriers, (list, tuple)) or len(couriers) == 0:
            errors.append(f"item {i}: courier field must be a non-empty list, got {couriers!r}")
            continue
        for c in couriers:
            key = (ts, str(c))
            if key not in cmap:
                errors.append(f"item {i}: pair ({ts!r},{c!r}) is not a valid input row")
                continue
            if c in used_couriers:
                errors.append(f"item {i}: courier {c!r} used more than once")
            used_couriers.add(c)
            tasks, sc, willingness = cmap[key]
            dup = covered.intersection(tasks)
            if dup:
                errors.append(f"item {i}: task(s) {sorted(dup)} already covered")
            covered.update(tasks)
            total_score += sc
            if cost_model is not None:
                predicted_cost += cost_model.predict_cost(sc, willingness, len(tasks))

    stats = {
        "covered": len(covered),
        "total_tasks": len(all_tasks),
        "total_score": total_score,
    }
    # When a calibrated cost model exists, estimate the REAL per-case score:
    # sum(cost over assignments) + penalty * uncovered_tasks.
    if cost_model is not None:
        uncovered = len(all_tasks) - len(covered)
        stats["predicted_score"] = predicted_cost + cost_model.penalty_per_task * uncovered
    return (len(errors) == 0), errors, stats


def _submit_local(solver_code: str, case_name: str) -> ScoreResult:
    ok, info = run_local_gate(solver_code)
    if not ok:
        # Find the first error to report
        for cname, cr in info.get("cases", {}).items():
            if cr.get("error"):
                return ScoreResult(ok=False, score=float("inf"),
                                   message=f"[{cname}] solver failed: {cr['error']}")
            if cr.get("errors"):
                shown = "; ".join(cr["errors"][:5])
                return ScoreResult(ok=False, score=float("inf"),
                                   message=f"[{cname}] INVALID: {shown}")
        return ScoreResult(ok=False, message="local gate failed (unknown reason)")

    st = info["stats"]
    n_cases = st.get("n_cases", 1)
    n_passed = st.get("n_cases_passed", 0)

    if "predicted_score" in st:
        avg_score = st["predicted_score"]
    else:
        covered = st.get("covered", 0)
        total = st.get("total_tasks", 0)
        avg_score = (-1e6 * covered + st.get("total_score", 0)) / max(n_cases, 1)

    return ScoreResult(
        ok=True, score=avg_score,
        accepted_orders=st.get("covered", 0),
        raw={"stats": st, "cases": info.get("cases", {})},
        message=f"VALID {n_passed}/{n_cases} cases passed, "
                f"covered={st.get('covered')}/{st.get('total_tasks')} "
                f"avg_score={avg_score:.2f} (local estimate)")


# ===========================================================================
# Local gate — run + validate a solver WITHOUT spending a real submission.
# The orchestrator calls this before deciding whether to submit.
# ===========================================================================
def _local_case_files():
    """Return a dict of {case_name: file_path} for every available local
    test case. Scans the project root and common subdirectories (dataset/, data/)
    for .txt files, excluding example_solution.txt."""
    cases = {}
    exclude = {"example_solution.txt", "README.md"}
    search_dirs = [HERE]
    for subdir in ("dataset", "data"):
        subpath = os.path.join(HERE, subdir)
        if os.path.isdir(subpath):
            search_dirs.append(subpath)
    for sdir in search_dirs:
        for fname in os.listdir(sdir):
            if fname.endswith(".txt") and fname not in exclude:
                cases[fname[:-4]] = os.path.join(sdir, fname)
    return cases


def run_local_gate(solver_code: str, case_path: Optional[str] = None):
    """Execute the solver locally on one or more case files and validate output.

    When case_path is given (or LOCAL_CASE is set), tests ONLY that case.
    When neither is set, tests ALL locally available .txt cases.

    Returns (ok: bool, info: dict).
      ok   = True only if ALL cases ran and validated cleanly.
      info = {elapsed, cases: {case_name: {ok, error, errors, stats, predicted_score}}, ...}
    """
    # Determine which cases to test
    if case_path is not None:
        cases = {"user_specified": case_path}
    else:
        env_case = os.environ.get("LOCAL_CASE")
        if env_case:
            cases = {"user_specified": env_case}
        else:
            cases = _local_case_files()
            if not cases:
                # fallback to the default single case
                default = os.path.join(HERE, "dataset/large_seed301.txt")
                cases = {"large_seed301": default}

    total_elapsed = 0.0
    all_case_results = {}
    any_failure = False

    # --- Phase 1: run solver + validate on each case (single execution per case) ---
    for case_name, cpath in cases.items():
        try:
            with open(cpath) as f:
                input_text = f.read()
        except Exception as e:
            all_case_results[case_name] = {"ok": False, "error": f"cannot read {cpath}: {e}"}
            any_failure = True
            continue

        t0 = time.time()
        sol, err = _run_solver_locally(solver_code, input_text)
        elapsed = time.time() - t0
        total_elapsed += elapsed

        if sol is None:
            all_case_results[case_name] = {"ok": False, "error": err, "elapsed": elapsed}
            any_failure = True
            continue

        ok, errors, stats = validate_solution(sol, input_text)
        all_case_results[case_name] = {
            "ok": ok, "errors": errors, "stats": stats, "elapsed": elapsed,
            "_sol": sol, "_input": input_text,   # cached for phase 2
        }
        if not ok:
            any_failure = True

    # --- Phase 2: compute predicted score reusing the already-run solutions ---
    cost_model = _get_cost_model()
    per_case_predicted = {}   # case_name -> predicted cost (or None)
    if cost_model is not None:
        for case_name, r in all_case_results.items():
            if r.get("error") or not r.get("stats"):
                per_case_predicted[case_name] = None
                continue
            if not r.get("ok"):
                # invalid solution: penalty = all tasks unassigned
                per_case_predicted[case_name] = r["stats"].get("total_tasks", 0) * 100
                continue
            try:
                sol = r["_sol"]
                input_text = r["_input"]
                cmap, all_t = _parse_candidates(input_text)
                covered = set()
                case_cost = 0.0
                for ts, couriers in sol:
                    for c in couriers:
                        row = cmap.get((ts, str(c)))
                        if row:
                            _, sc, w = row
                            case_cost += cost_model.predict_cost(sc, w)
                            covered.update(row[0])
                uncovered = len(all_t) - len(covered)
                per_case_predicted[case_name] = (
                    case_cost + cost_model.penalty_per_task * uncovered
                )
            except Exception:
                per_case_predicted[case_name] = r["stats"].get("total_tasks", 0) * 100

    # Attach per-case predicted_score into the case result dict
    valid_predicted = [v for v in per_case_predicted.values() if v is not None]
    avg_predicted = (sum(valid_predicted) / len(valid_predicted)) if valid_predicted else None
    for case_name, pred in per_case_predicted.items():
        if case_name in all_case_results:
            all_case_results[case_name]["predicted_score"] = pred
            # remove cached internals so they don't leak into logs
            all_case_results[case_name].pop("_sol", None)
            all_case_results[case_name].pop("_input", None)

    # Aggregate stats across all cases for backward compat
    total_covered = sum(
        r["stats"]["covered"] for r in all_case_results.values()
        if r.get("stats"))
    total_tasks = sum(
        r["stats"]["total_tasks"] for r in all_case_results.values()
        if r.get("stats"))
    total_score = sum(
        r["stats"]["total_score"] for r in all_case_results.values()
        if r.get("stats"))

    agg_stats = {
        "covered": total_covered,
        "total_tasks": total_tasks,
        "total_score": total_score,
        "n_cases": len(cases),
        "n_cases_passed": sum(1 for r in all_case_results.values() if r.get("ok")),
    }
    if avg_predicted is not None:
        agg_stats["predicted_score"] = avg_predicted

    return (not any_failure), {
        "cases": all_case_results,
        "stats": agg_stats,
        "elapsed": total_elapsed,
    }


# ===========================================================================
# Public entry point — the agent calls ONLY this.
# ===========================================================================
def submit_and_score(solver_code: str, case_name: str = "large_seed301") -> ScoreResult:
    # Read mode at CALL time so it can be set programmatically (e.g. --calibrate).
    mode = os.environ.get("JUDGE_MODE", "local")
    if mode == "http":
        return _submit_http(solver_code, case_name)
    if mode == "browser":
        return _submit_browser(solver_code, case_name)
    if mode == "manual":
        return _submit_manual(solver_code, case_name)
    return _submit_local(solver_code, case_name)
