"""Direct real-judge submission of specific solver files (last-day scarce bet).
Loads .env, submits each file, prints real avg_score + per-case, updates
agent_state.json / best_solver.py / archive on a new real best."""
import os, sys, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
# load .env BEFORE importing judge_adapter (JUDGE_BASE/login read env)
env_path = os.path.join(HERE, ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ["JUDGE_MODE"] = "http"

import judge_adapter as J
import agent, archive

STATE = os.path.join(HERE, "agent_state.json")

def _state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}

def submit_file(path, label):
    code = open(path).read()
    h = hashlib.sha1(code.encode()).hexdigest()[:10]
    st = _state()
    print(f"\n=== SUBMIT {label}  [{os.path.basename(path)} hash {h}]  "
          f"budget_remaining={st.get('daily_remaining','?')} ===", flush=True)
    res = J.submit_and_score(code)
    if not res.ok:
        print(f"  FAILED: {res.message}", flush=True)
        return None
    print(f"  REAL avg_score = {res.score:.4f}", flush=True)
    # persist + champion update
    st = _state()
    if res.daily_remaining is not None:
        st["daily_remaining"] = res.daily_remaining
        st["submissions_used"] = agent.DAILY_LIMIT - res.daily_remaining
    best = st.get("best_real_score")
    if best is None or res.score < best:
        st["best_real_score"] = res.score
        archive.archive_solver(h, code)
        with open(os.path.join(HERE, "best_solver.py"), "w") as f:
            f.write(code)
        print(f"  *** NEW REAL BEST {res.score:.4f} (saved best_solver.py, hash {h}) ***", flush=True)
    json.dump(st, open(STATE, "w"), indent=2)
    # log to run_log
    with open(os.path.join(HERE, "run_log.jsonl"), "a") as f:
        f.write(json.dumps({"hash": h, "submitted": True, "ok": True,
                            "score": res.score, "algorithm": label,
                            "note": "scarce-bet manual submit"}) + "\n")
    return res.score

if __name__ == "__main__":
    files = sys.argv[1:] or ["champion_v18.py:baseline-40/40",
                             "champion_scarce39.py:scarce-39/40-bet"]
    for spec in files:
        path, _, label = spec.partition(":")
        submit_file(os.path.join(HERE, path), label or path)
