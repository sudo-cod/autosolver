"""
memory.py — consolidate the run history into a TIGHT strategy brief.
====================================================================
The flat, ever-growing knowledge digest dilutes the prompt. This distills the
~200 logged attempts into a structured brief (Neo failure-signature / Mem0
consolidation style): best approach per archetype, CONFIRMED dead-ends (so the
solver stops repeating them), and per-weak-case tactics. Written to
`memory_brief.md`; `archive.build_knowledge_digest` prefers it when present.
"""
import os
import time

import archive

HERE = os.path.dirname(__file__)
BRIEF_PATH = os.path.join(HERE, "memory_brief.md")

MEMORY_SYSTEM_PROMPT = """\
You are the memory consolidator for an autonomous agent that writes Python
solvers for a courier task-assignment problem (LOWER score is better; cost per
assignment = willingness*total_score + (1-willingness)*100*num_tasks; an
uncovered task costs 100). Distill the run history into a TIGHT brief with
EXACTLY these three sections (markdown headers), <= 40 lines total:

## BEST PER ARCHETYPE
For each case family, the best approach found and its score.

## CONFIRMED DEAD-ENDS
Approaches that reliably FAILED or scored WORSE, each with a one-line WHY, so the
solver never wastes a turn on them again (e.g. "ILP ignoring willingness ->
worse: picks low-score/high-risk pairs").

## WEAK-CASE TACTICS
For the 2-3 hardest cases, concrete tactics that reduced their cost.

Be specific and terse. No preamble, no closing remarks — just the three sections."""


def _build_input(records):
    tried = archive._approaches_tried(records)            # (algo, score, outcome)
    cases = archive.per_case_analysis(records)            # weak-first rows
    lines = ["Approaches tried (algorithm -> outcome; real score lower=better):"]
    for algo, _, outcome in tried[:25]:
        lines.append(f"  - {algo[:70]}: {outcome}")
    lines.append("\nPer-case best real scores (weak spots first):")
    for r in cases:
        lines.append(f"  {r['case']}: best={r['best_score']:.1f} "
                     f"assigned={r['assigned']}/{r['total']}")
    lines.append("\nDistill into the 3-section brief.")
    return "\n".join(lines)


def consolidate(call_fn, model, api_key, verbose=True):
    """`call_fn` is agent.call_claude (passed in to avoid an import cycle)."""
    records = archive.load_history()
    if not records:
        return None
    try:
        brief = call_fn(model, [{"role": "user", "content": _build_input(records)}],
                        api_key, system=MEMORY_SYSTEM_PROMPT, max_tokens=2000).strip()
    except Exception as e:
        if verbose:
            print(f"  [memory] consolidation failed: {e}")
        return None
    if not brief:
        return None
    with open(BRIEF_PATH, "w") as f:
        f.write(f"<!-- consolidated {time.strftime('%Y-%m-%d %H:%M')} from "
                f"{len(records)} attempts -->\n" + brief + "\n")
    if verbose:
        print(f"  [memory] wrote memory_brief.md ({len(brief)} chars).")
    return brief


def load_brief():
    if os.path.exists(BRIEF_PATH):
        try:
            return open(BRIEF_PATH).read()
        except Exception:
            return None
    return None
