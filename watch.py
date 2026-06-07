"""
watch.py — live terminal view of the running agent.
===================================================
Run in a SECOND terminal while `agent.py` runs. Tails run_log.jsonl,
calls.jsonl and agent_state.json and redraws a compact panel every ~2s:
champion + budget + suite->real map, the latest ReAct thought/action/obs (or
linear iter), the last solver (hash/algo/pred/coverage), and the head of the
last LLM prompt + response. Pure stdlib; read-only.
"""
import os
import json
import time

HERE = os.path.dirname(__file__)
C = {"dim": "\033[2m", "b": "\033[1m", "cyan": "\033[36m", "yel": "\033[33m",
     "grn": "\033[32m", "red": "\033[31m", "blu": "\033[34m", "x": "\033[0m"}


def last_jsonl(name, n=1):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-n:]


def load_json(name):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {}


def render():
    st = load_json("agent_state.json")
    recon = load_json("recon.json")
    log = last_jsonl("run_log.jsonl", 1)
    call = last_jsonl("calls.jsonl", 1)
    lines = []
    A = lines.append
    A(f"{C['b']}{C['cyan']}== AutoSolver — live =={C['x']}   "
      f"{time.strftime('%H:%M:%S')}")
    A(f"  champion real={C['grn']}{st.get('best_real_score')}{C['x']}  "
      f"local={st.get('best_local')}  "
      f"budget={C['yel']}{st.get('daily_remaining')}/{20}{C['x']} "
      f"(used {st.get('submissions_used')})")
    if recon:
        A(f"  {C['dim']}suite→real: real≈{recon['a']:.3f}·pred+{recon['b']:.1f} "
          f"rmse={recon['rmse']:.0f} R²={recon['r2']:.2f}{C['x']}")
    A("")
    if log:
        r = log[0]
        if r.get("mode") == "react":
            A(f"  {C['b']}STEP {r.get('step')} · {C['blu']}{r.get('action')}{C['x']}")
            if r.get("thought"):
                A(f"  {C['dim']}THOUGHT{C['x']} {r['thought'][:200]}")
            if r.get("focus"):
                A(f"  {C['yel']}FOCUS{C['x']}   {r['focus'][:200]}")
            if r.get("obs"):
                A(f"  {C['grn']}OBS{C['x']}     {r['obs'][:200]}")
            if r.get("hash"):
                A(f"  {C['dim']}solver {r['hash']} pred={r.get('predicted_score')}{C['x']}")
        else:
            A(f"  {C['b']}iter {r.get('iter')}{C['x']} {str(r.get('algorithm',''))[:60]}")
            A(f"  pred={r.get('predicted_score')} submitted={r.get('submitted')} "
              f"real={r.get('score')}")
    else:
        A(f"  {C['dim']}(no run_log yet — start the agent){C['x']}")
    A("")
    if call:
        c = call[0]
        A(f"  {C['b']}last LLM call{C['x']} [{C['cyan']}{c.get('tag')}{C['x']}] "
          f"temp={c.get('temperature')}")
        resp = (c.get("response") or "").strip().replace("\n", " ")
        A(f"  {C['dim']}response:{C['x']} {resp[:220]}")
    return "\n".join(lines)


def main():
    try:
        while True:
            os.system("clear")
            print(render())
            print(f"\n{C['dim']}refreshing every 2s · Ctrl-C to quit{C['x']}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
