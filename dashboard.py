"""
dashboard.py — self-contained HTML view of what the agent is doing.
==================================================================
Reads run_log.jsonl, calls.jsonl, calib_log.jsonl, solver_archive/,
cost_model.json, recon.json and writes ONE openable dashboard.html (inline
CSS/JS, no server, no external libs). Re-run to refresh.

Shows: score trajectory, the ReAct chain-of-thought (thought->action->obs) /
linear iterations, the prompts fed to the LLM, the generated solver code per
step, per-case score tables, and how the cost model / suite->real map evolve.
"""
import os
import json
import html
import time

HERE = os.path.dirname(__file__)


def _load_jsonl(name):
    path = os.path.join(HERE, name)
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def _load_json(name):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return None


def _archived(h):
    if not h:
        return None
    p = os.path.join(HERE, "solver_archive", f"{h}.py")
    return open(p).read() if os.path.exists(p) else None


def esc(x):
    return html.escape(str(x))


def _svg_line(points, w=720, h=160, pad=28, color="#3b82f6"):
    """Minimal inline SVG line chart from a list of (x_label, y) — lower y plotted."""
    if not points:
        return "<div class='muted'>no data</div>"
    ys = [p[1] for p in points]
    ymin, ymax = min(ys), max(ys)
    rng = (ymax - ymin) or 1.0
    n = len(points)
    def X(i): return pad + (w - 2 * pad) * (i / max(1, n - 1))
    def Y(v): return pad + (h - 2 * pad) * (v - ymin) / rng  # lower value -> top
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, (_, v) in enumerate(points))
    dots = "".join(
        f"<circle cx='{X(i):.1f}' cy='{Y(v):.1f}' r='3' fill='{color}'>"
        f"<title>{esc(lbl)}: {v:.2f}</title></circle>"
        for i, (lbl, v) in enumerate(points))
    return (f"<svg width='{w}' height='{h}' class='chart'>"
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
            f"{dots}"
            f"<text x='{pad}' y='14' class='axis'>{ymax:.0f}</text>"
            f"<text x='{pad}' y='{h-6}' class='axis'>{ymin:.0f}</text></svg>")


def _collapsible(summary, body, open_=False):
    o = " open" if open_ else ""
    return f"<details{o}><summary>{summary}</summary>{body}</details>"


def _case_table(cases, kind):
    """cases: dict name->{stats:{covered,total_tasks,predicted_score}} (gate) OR
    list of judge case_results dicts (submission)."""
    rows = []
    if isinstance(cases, dict):
        for name, ci in sorted(cases.items()):
            st = ci.get("stats", {})
            if ci.get("error"):
                rows.append((name, "CRASH", "", ""))
            elif ci.get("errors"):
                rows.append((name, "INVALID", "", ""))
            else:
                rows.append((name, f"{st.get('covered')}/{st.get('total_tasks')}",
                             f"{st.get('predicted_score', 0):.1f}", ""))
        head = "<tr><th>case</th><th>covered</th><th>predicted</th><th></th></tr>"
    elif isinstance(cases, list):
        for c in cases:
            sc = c.get("score", c.get("total_score"))
            rows.append((c.get("case_file", "?"),
                         f"{c.get('assigned', c.get('assigned_count'))}/{c.get('total_tasks')}",
                         f"{sc:.1f}" if isinstance(sc, (int, float)) else "-",
                         c.get("status", "")))
        head = "<tr><th>case</th><th>assigned</th><th>real score</th><th>status</th></tr>"
    else:
        return ""
    body = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table class='cases'>{head}{body}</table>"


def build():
    log = _load_jsonl("run_log.jsonl")
    calls = _load_jsonl("calls.jsonl")
    calib = _load_jsonl("calib_log.jsonl")
    cost = _load_json("cost_model.json")
    recon = _load_json("recon.json")
    state = _load_json("agent_state.json") or {}

    # --- header facts ---
    champ = state.get("best_real_score")
    champ_local = state.get("best_local")
    budget = f"{state.get('daily_remaining','?')}/{20} left (used {state.get('submissions_used','?')})"
    formula = "uncalibrated"
    if cost and cost.get("cost_form") == "expected_value":
        P = cost.get("cost_params", {}).get("P")
        formula = f"cost = w·ts + (1−w)·{P:.0f}·num_tasks (+ {P:.0f}/unassigned)"
    reconstr = ("not reliable yet" if not recon else
                f"real ≈ {recon['a']:.3f}·pred + {recon['b']:.1f}  "
                f"(R²={recon['r2']:.2f}, rmse={recon['rmse']:.0f}, n={recon['n']})")

    # --- score trajectory (submitted+ok, agent-only) ---
    subs = [r for r in log if r.get("submitted") and r.get("ok") and r.get("score")
            and not r.get("manual")]
    traj = [(r.get("algorithm", "")[:18] or f"#{i}", r["score"]) for i, r in enumerate(subs)]

    # --- calibration evolution ---
    pser = [(time.strftime("%H:%M", time.localtime(s["ts"])), s["P"])
            for s in calib if s.get("P") is not None]
    rmser = [(time.strftime("%H:%M", time.localtime(s["ts"])), s["recon_rmse"])
             for s in calib if s.get("recon_rmse") is not None]

    # --- timeline (steps / iterations) ---
    cards = []
    for r in log:
        is_react = r.get("mode") == "react"
        tag = (f"STEP {r.get('step')} · {r.get('action')}" if is_react
               else f"iter {r.get('iter')} · {esc(r.get('algorithm','') )[:40]}")
        badge = ""
        if r.get("submitted") and r.get("ok"):
            nb = "NEW BEST" if (champ is not None and abs(r['score'] - champ) < 1e-6) else ""
            badge = f"<span class='badge sub'>real {r['score']:.2f} {nb}</span>"
        elif r.get("gate_ok") is False or (r.get("obs", "").find("INVALID") >= 0):
            badge = "<span class='badge bad'>invalid</span>"
        elif r.get("predicted_score") is not None:
            badge = f"<span class='badge ok'>pred {r['predicted_score']:.1f}</span>"

        inner = []
        if is_react:
            if r.get("thought"):
                inner.append(f"<div class='th'><b>THOUGHT</b> {esc(r['thought'])}</div>")
            if r.get("focus"):
                inner.append(f"<div class='fo'><b>FOCUS</b> {esc(r['focus'])}</div>")
            if r.get("obs"):
                inner.append(f"<div class='ob'><b>OBS</b> {esc(r['obs'])}</div>")
        code = _archived(r.get("hash"))
        if code:
            inner.append(_collapsible(f"solver code [{esc(r.get('hash'))}]",
                                      f"<pre class='code'>{esc(code)}</pre>"))
        cases = r.get("gate_cases") or r.get("case_results")
        if cases:
            kind = "judge" if r.get("case_results") else "gate"
            inner.append(_collapsible(f"per-case ({kind})", _case_table(cases, kind)))
        cards.append(f"<div class='card'><div class='cardhead'>{esc(tag)} {badge}</div>"
                     + "".join(inner) + "</div>")

    # --- prompts (LLM calls) ---
    call_cards = []
    for c in calls[-40:][::-1]:
        when = time.strftime("%H:%M:%S", time.localtime(c.get("ts", 0)))
        msgs = c.get("messages", [])
        last_user = next((m["content"] for m in reversed(msgs)
                          if m.get("role") == "user"), "")
        body = (f"<pre class='code'><b># system</b>\n{esc(c.get('system',''))}\n\n"
                f"<b># last user message</b>\n{esc(last_user)}\n\n"
                f"<b># response</b>\n{esc(c.get('response',''))}</pre>")
        call_cards.append(_collapsible(
            f"[{esc(c.get('tag'))}] {when} · temp={esc(c.get('temperature'))}", body))

    # --- assemble HTML ---
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
    .wrap{max-width:980px;margin:0 auto;padding:24px}
    h1{font-size:20px} h2{font-size:16px;margin-top:28px;border-bottom:1px solid #334155;padding-bottom:6px}
    .hdr{display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;background:#1e293b;padding:16px;border-radius:10px}
    .hdr b{color:#93c5fd}
    .card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;margin:10px 0}
    .cardhead{font-weight:600;margin-bottom:6px}
    .th{color:#cbd5e1} .fo{color:#fcd34d} .ob{color:#86efac}
    .th b,.fo b,.ob b{display:inline-block;width:64px;color:#64748b;font-size:11px}
    .badge{font-size:11px;padding:2px 8px;border-radius:10px;margin-left:8px}
    .badge.sub{background:#0e7490} .badge.ok{background:#334155} .badge.bad{background:#7f1d1d}
    details{margin:6px 0} summary{cursor:pointer;color:#93c5fd;font-size:12px}
    pre.code{background:#0b1220;border:1px solid #334155;border-radius:8px;padding:10px;
      overflow:auto;max-height:420px;font:12px/1.45 ui-monospace,Menlo,monospace;color:#cbd5e1}
    table.cases{border-collapse:collapse;font-size:12px;margin:6px 0}
    table.cases th,table.cases td{border:1px solid #334155;padding:3px 8px;text-align:left}
    .chart{background:#0b1220;border:1px solid #334155;border-radius:8px}
    .axis{fill:#64748b;font-size:10px} .muted{color:#64748b}
    """
    h = ["<!doctype html><meta charset='utf-8'><title>AutoSolver dashboard</title>",
         f"<style>{css}</style><div class='wrap'>",
         f"<h1>AutoSolver agent dashboard <span class='muted'>· generated "
         f"{time.strftime('%Y-%m-%d %H:%M')}</span></h1>",
         "<div class='hdr'>",
         f"<div><b>Best REAL score:</b> {esc(champ)}</div>",
         f"<div><b>Champion local (suite-avg):</b> {esc(champ_local)}</div>",
         f"<div><b>Budget:</b> {esc(budget)}</div>",
         f"<div><b>Cost formula:</b> {esc(formula)}</div>",
         f"<div style='grid-column:1/3'><b>Suite→real map:</b> {esc(reconstr)}</div>",
         "</div>",
         "<h2>Score trajectory (real submissions, lower is better)</h2>",
         _svg_line(traj, color="#22d3ee"),
         "<h2>Calibration evolution</h2>",
         "<div class='muted'>cost penalty P over time</div>", _svg_line(pser, h=120, color="#a78bfa"),
         "<div class='muted'>suite→real rmse over time (lower = tighter proxy)</div>",
         _svg_line(rmser, h=120, color="#f472b6"),
         f"<h2>Timeline — {len(cards)} steps/iterations</h2>",
         "".join(reversed(cards)),
         "<h2>Prompts fed to the LLM (most recent 40)</h2>",
         "".join(call_cards) or "<div class='muted'>no calls.jsonl yet — run the agent.</div>",
         "</div>"]
    out = os.path.join(HERE, "dashboard.html")
    with open(out, "w") as f:
        f.write("\n".join(h))
    print(f"wrote {out}  ({len(cards)} timeline cards, {len(calls)} llm calls, "
          f"{len(subs)} submissions)")
    return out


if __name__ == "__main__":
    build()
