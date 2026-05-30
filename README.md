# AutoSolver Agent

An **agentic system** for the food-delivery task-assignment problem. It does not
just solve the problem once — it *writes solver code, gets it scored, reflects on
the score, and rewrites the code to drive the score down*, looping autonomously.

```
   ┌──────────── Claude API (the "brain") ────────────┐
   │  writes / revises a solve() function              │
   └───────────────────────┬───────────────────────────┘
                           │ solver code
                           ▼
                  run in sandbox + SUBMIT
                           │
                           ▼ score (lower = better)
                  ┌──────── judge ────────┐
                  │ real judge OR local   │
                  └───────────┬───────────┘
                              │ score + notes
                              ▼
                  REFLECT: "why this score? what to change?"
                              │
                              └──────────► next revision (loop)
```

## Files

| file | role |
|---|---|
| `agent.py` | the loop: generate → submit → score → reflect → rewrite |
| `judge_adapter.py` | **the only file you customize** — how a score is obtained |
| `best_solver.py` | (output) best-scoring solver found so far |
| `run_log.jsonl` | (output) every iteration: code hash, score, notes |

## Quick start (offline, no judge needed)

Develop and watch the loop work using the **local scorer** first:

```bash
export ANTHROPIC_API_KEY=sk-...
export JUDGE_MODE=local
export LOCAL_CASE=/path/to/large_seed301.txt
python3 agent.py --iterations 12 --model claude-opus-4-8
```

The local scorer mirrors the stated objective (maximize covered orders, then
minimize total score). Use it to confirm the agent improves across iterations
before spending submissions on the real judge.

## Wiring the real judge (hackathon.mykeeta.com)

You weren't sure how the site takes submissions. Open it, submit once by hand
with browser devtools open, and check the **Network** tab. Then pick a mode:

**If there's an HTTP API** (you see a POST request when you submit):
```bash
export JUDGE_MODE=http
export JUDGE_URL="<the POST url you saw>"
export JUDGE_TOKEN="<bearer/cookie if required>"
```
Then in `judge_adapter._submit_http`, adjust the request field names
(`code`/`case`/`language`) and the response parsing (`data["score"]`) to match
what you actually saw in devtools.

**If it's a web form** (paste code, click submit, score appears on the page):
```bash
pip install playwright && playwright install chromium
export JUDGE_MODE=browser
export JUDGE_URL="https://hackathon.mykeeta.com/..."
export SEL_CODE="<css selector for the code textarea>"
export SEL_SUBMIT="<css selector for the submit button>"
export SEL_SCORE="<css selector for the score element>"
# if login is needed, save cookies once and point JUDGE_STORAGE at them
```

**If neither can be automated:**
```bash
export JUDGE_MODE=manual
```
The agent writes the current solver to `current_solver.py`, pauses, you submit
it on the site, and type the score back. Slower, but fully functional.

## IMPORTANT — score direction

The agent always **minimizes**. If the judge reports a score where *higher is
better*, negate it inside the adapter (`score = -raw_score`) so the loop
optimizes the right direction.

## Calibrating the objective

The competition prompt is internally ambiguous (can one task go to multiple
couriers? does `willingness` enter the score?). Two-step approach:

1. Run a few iterations against the **real judge** with deliberately different
   solvers (max coverage; min score; with/without bundling).
2. Compare the judge's scores to what the local scorer predicted. Where they
   diverge, fix `_score_solution_locally` in `judge_adapter.py` and update
   `PROBLEM_BRIEF`/`OBJECTIVE` so the agent optimizes the *real* metric.

## Notes

- Solver code from the LLM runs in a subprocess with a hard timeout, so a buggy
  or slow solver can't hang the loop.
- Invalid solutions (reused courier/task, unknown pair) are rejected before
  scoring and fed back as a failure for the LLM to fix.
- Context is trimmed each iteration to keep token usage bounded over long runs.
- `claude-opus-4-8` is the strongest brain; `claude-sonnet-4-6` is cheaper/faster
  for more iterations per dollar.
