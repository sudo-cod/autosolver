# AutoSolver Agent

DEMO 视频： https://www.bilibili.com/video/BV16EEh6KEWh/?spm_id_from=333.1387.homepage.video_card.click&vd_source=90744a174c9137c1d226dab5a09f657b

设计方案：AutoSolver系统设计方案_团优解.pdf

An **agentic system** for the food-delivery courier task-assignment problem ([hackathon.mykeeta.com](https://hackathon.mykeeta.com)). The agent **writes Python solver code**, runs it through a local gate and/or the real judge, **reflects on the result**, and rewrites the code in a loop to drive the score down.

```
  ┌──────────── LLM (Longcat / DeepSeek) ────────────┐
  │  writes / revises solve(input_text) -> list       │
  └───────────────────────┬───────────────────────────┘
                          │ solver code
                          ▼
                 run_local_gate (validate + score)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
     local test suite              real judge
     (offline dev)                 (20 submits/day)
            │                           │
            └──────── score (lower = better) ────────┘
                          │
                          ▼
              reflect → archive → next iteration
```

## Problem

**Input:** tab-separated candidate assignments (`task_id_list`, `courier_id`, `total_score`, `willingness`).

**Output:** a list of `(task_str, [courier, ...])` tuples. The courier list may include a primary and backup couriers for the same task.

**Objective (lexicographic):**
1. Maximize the number of covered tasks.
2. Among equal coverage, minimize cost (the judge's formula is calibrated from real submissions).

Key levers: 2-task bundles (one courier, two tasks), backup couriers to reduce risk, penalty of 100 per uncovered task.

## Requirements

- Python 3.10+
- Stdlib only for the agent loop (optional: `playwright` for browser mode; `pulp`/`ortools` inside generated solvers)
- API key in `.env`:

```bash
LONGCAT_KEY=...          # default provider
# DEEPSEEK_KEY=...       # alternative: --provider deepseek
```

## Quick start (offline)

Generate the synthetic suite (10 case archetypes) and run the agent locally:

```bash
python3 make_synthetic.py
export JUDGE_MODE=local
python3 agent.py --iterations 12 --mode react
```

The local gate runs the solver on every `.txt` in `dataset/` (real `large_seed301.txt` plus synthetic `syn_*.txt`), validates the output, and scores via `cost_model.json` when calibrated.

## Real judge (HTTP)

`http` mode is already implemented in `judge_adapter.py` — login → POST `/judge` → poll `/result/{job_id}`.

```bash
# .env
JUDGE_MODE=http
JUDGE_TEAM=your_team
JUDGE_EMAIL=your@email.com
# JUDGE_BASE=https://hackathon.mykeeta.com   # default

python3 agent.py --mode submit --max-submits 6
```

On the first run in `http` mode, the agent automatically spends 1 submission to calibrate the cost formula (`--calibrate`).

Other judge modes (see `judge_adapter.py`):

| `JUDGE_MODE` | Description |
|---|---|
| `local` | Offline scoring on local cases (default) |
| `http` | hackathon.mykeeta.com API |
| `browser` | Headless Playwright (CSS selectors required) |
| `manual` | Agent writes `current_solver.py`; you type the score back |

## Agent modes

```bash
python3 agent.py --mode react      # LLM controller picks actions (default)
python3 agent.py --mode linear     # fixed generate → score → reflect loop
python3 agent.py --mode submit     # deliberate real submits with daily budget
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--iterations N` | Number of steps (react/linear) |
| `--candidates N` | Best-of-N candidates per iteration |
| `--provider longcat\|deepseek` | LLM provider |
| `--model NAME` | Model name (default: `LongCat-2.0-Preview` / `deepseek-chat`) |
| `--max-submits N` | Cap on real submits in `submit` mode |
| `--calibrate` | 1 judge probe → `cost_model.json`, then exit |
| `--consolidate` | Distill history into `memory_brief.md`, then exit |

Judge daily limit: **20 submissions per team** (resets at Beijing midnight). The agent tracks the remaining budget in `agent_state.json` and adjusts probe aggressiveness accordingly.

## Project layout

| File | Role |
|---|---|
| `agent.py` | Main loop: generate → gate → submit → reflect |
| `judge_adapter.py` | **The only file to customize for your judge** — `submit_and_score()`, local gate |
| `archive.py` | Cross-run memory: `run_log.jsonl`, `solver_archive/` |
| `calibrate.py` | Reverse-engineer cost formula from judge responses → `cost_model.json` |
| `reconcile.py` | Suite→real mapping (`real ≈ a·pred + b`) → `recon.json` |
| `memory.py` | Consolidate history into a tight `memory_brief.md` |
| `make_synthetic.py` | Generate `dataset/syn_*.txt` (10 archetypes) |
| `calibrate_synthetic.py` | Tune synthetic cases to match real per-case scores |
| `dashboard.py` | Build static `dashboard.html` from logs |
| `watch.py` | Live terminal panel (run in a second terminal) |
| `submit_bet.py` | Manually submit ready-made solvers to the judge |
| `best_solver.py` | Best solver found so far (generated) |
| `standalone_solver.py` | Hand-written ILP/greedy solver (not the agent) |
| `dataset/` | Local cases (`large_seed301.txt`, `syn_*.txt`) |
| `Docs/AutoSolver系统设计方案_团优解.pdf` | AutoSolver 系统设计方案 |
| `Docs/介绍视频_团优解.movf` | 产品介绍与展示 |


### Generated artifacts (in `.gitignore`)

`run_log.jsonl`, `calls.jsonl`, `agent_state.json`, `cost_model.json`, `recon.json`, `solver_archive/`, `dashboard.html`, `memory_brief.md`

## Monitoring

```bash
# Terminal 1
python3 agent.py --mode submit

# Terminal 2 — live status
python3 watch.py

# After a run — HTML dashboard
python3 dashboard.py && open dashboard.html
```

## Calibration and local scoring accuracy

1. **`calibrate.py`** — mines `case_results[].detail` from real judge responses to learn `cost(total_score, willingness)` and the uncovered-task penalty.
2. **`reconcile.py`** — fits a linear correction from (local pred, real avg) pairs so the `submit`-mode gate filters candidates in real units.
3. **`calibrate_synthetic.py`** — tunes willingness in synthetic cases to match the champion's real per-case scores.

Without calibration, the local suite is only an approximation. After one or more real submits, the agent uses the calibrated model for go/no-go decisions.

## Notes

- The agent **always minimizes** score. If the judge reports higher-is-better, negate it inside `judge_adapter.py`.
- Solver code runs in a subprocess with a hard timeout; invalid solutions are rejected before scoring.
- LLM context is trimmed each iteration; long-term memory lives in `archive` + `memory_brief.md`.
- Synthetic cases use the `syn_` prefix so their names never collide with real judge case names.

## Example `.env`

```bash
LONGCAT_KEY=sk-...
JUDGE_MODE=http
JUDGE_TEAM=MyTeam
JUDGE_EMAIL=me@example.com
```
