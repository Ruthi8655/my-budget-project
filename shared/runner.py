"""
shared/runner.py — shared scenario runner for both API and SDK harnesses.

Handles:
  - Multi-session scenarios (session with fresh_agent=true gets a new agent instance)
  - Per-turn result collection (tokens, cached_tokens, output_tokens, model_calls, cost, seconds)
  - Pass/fail checks on persisted state (budgets file, expenses file, turn answers)
  - Harness line counts (counts .py lines in a given harness directory)
  - Writes a single results JSON per run
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

_SHARED_DATA = Path(__file__).parent / "data"
_DATA_FILES  = ("history.jsonl", "memory.json", "memory.md",
                "expenses.json", "budgets.json", "sdk_sessions.db")


# ── harness line counter ──────────────────────────────────────────────────────

def count_harness_lines(harness_dir: Path) -> int:
    """Count total non-blank, non-comment Python lines in a directory."""
    total = 0
    for py in harness_dir.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


# ── check evaluator ───────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalise answer text for number-format-tolerant substring matching.

    Removes thousands-separator commas, non-breaking/regular spaces, and
    trailing ".0" so that "1,500 ILS", "1500.0" and "1500" all reduce to
    the same form before the substring search.
    """
    import re
    t = text.lower()
    t = t.replace("\u00a0", " ")          # non-breaking space → space
    t = re.sub(r"(\d),(?=\d{3})", r"\1", t)  # remove thousands commas: 1,500 → 1500
    t = re.sub(r"(\d)\.0(?=\D|$)", r"\1", t) # strip trailing .0: 1500.0 → 1500
    return t


def _eval_check(
    check: dict,
    turn_answers: dict[tuple[str, int], str],
    all_turns: list[dict],
) -> dict:
    """
    Evaluate one check dict.  Returns {"id", "description", "passed", "detail"}.
    turn_answers: {(session_id, turn_number): answer_text}
    all_turns: list of turn record dicts (for total_model_calls)
    """
    cid  = check["id"]
    desc = check["description"]
    op   = check.get("op", "eq")

    def _cmp(actual, expected) -> bool:
        if op == "eq":   return actual == expected
        if op == "gte":  return actual >= expected
        if op == "gt":   return actual > expected
        if op == "lte":  return actual <= expected
        return False

    try:
        # Import lazily so monkeypatching of module-level paths takes effect
        import shared.tools as _t
        ctype = check["type"]

        if ctype == "budgets_file":
            budgets = _t._load(_t.BUDGETS_FILE)
            actual  = budgets.get(check["key"])
            passed  = _cmp(actual, check["value"])
            detail  = f"budgets[{check['key']}]={actual}"

        elif ctype == "expenses_count":
            expenses = _t._load(_t.EXPENSES_FILE)
            passed   = _cmp(len(expenses), check["value"])
            detail   = f"expenses count={len(expenses)}"

        elif ctype == "expenses_category":
            expenses = _t._load(_t.EXPENSES_FILE)
            count    = sum(1 for e in expenses if e.get("category") == check["category"])
            passed   = _cmp(count, check["value"])
            detail   = f"category={check['category']} count={count}"

        elif ctype == "turn_answer_contains":
            key    = (check["session_id"], check["turn"])
            answer = turn_answers.get(key, "")
            sub    = check["substring"].lower()
            passed = sub in _normalize(answer)
            detail = f"answer={answer[:80]!r}"

        elif ctype == "total_model_calls":
            total  = sum(t["model_calls"] for t in all_turns)
            passed = _cmp(total, check["value"])
            detail = f"total_model_calls={total}"

        else:
            passed = False
            detail = f"unknown check type: {ctype}"

    except Exception as exc:
        passed = False
        detail = f"ERROR: {exc}"

    return {"id": cid, "description": desc, "passed": passed, "detail": detail}


# ── main runner ───────────────────────────────────────────────────────────────

def run_scenario(
    scenario_path: str | Path,
    make_agent: Callable[[], Any],
    harness_dir: Path,
    out_path: str | Path,
) -> dict:
    """
    Run a multi-session scenario.

    Args:
        scenario_path: path to scenario.json
        make_agent:    zero-arg factory that returns a fresh agent with .run_turn(text)->dict
        harness_dir:   directory whose .py lines are counted for the results
        out_path:      where to write the results JSON

    Returns the full results dict.
    """
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    sessions = scenario.get("sessions", [])
    checks   = scenario.get("checks", [])

    # Wipe all persisted state so each run starts from a clean slate
    _SHARED_DATA.mkdir(parents=True, exist_ok=True)
    for name in _DATA_FILES:
        f = _SHARED_DATA / name
        if f.exists():
            f.unlink()
            print(f"[runner] wiped {f.name}")

    all_turns: list[dict] = []
    turn_answers: dict[tuple[str, int], str] = {}

    run_start = time.perf_counter()

    for session in sessions:
        session_id  = session["session_id"]
        fresh_agent = session.get("fresh_agent", False)
        turns       = session["turns"]

        print(f"\n{'='*60}\nSession: {session_id}  fresh_agent={fresh_agent}\n{'='*60}")
        agent = make_agent()

        for turn in turns:
            turn_num  = turn["turn"]
            user_text = turn["user"]
            print(f"\n--- Turn {turn_num}: {user_text}")

            result  = agent.run_turn(user_text)
            answer  = result.get("answer", "")
            metrics = result.get("metrics", {})

            turn_record = {
                "session_id":    session_id,
                "turn":          turn_num,
                "user":          user_text,
                "answer":        answer,
                "input_tokens":  metrics.get("input_tokens",  0),
                "cached_tokens": metrics.get("cached_tokens", 0),
                "output_tokens": metrics.get("output_tokens", 0),
                "model_calls":   metrics.get("model_calls",   0),
                "tools":         metrics.get("tools",         []),
                "cost_usd":      metrics.get("cost_usd",      0.0),
                "wall_time_s":   metrics.get("wall_time_s",   0.0),
            }
            all_turns.append(turn_record)
            turn_answers[(session_id, turn_num)] = answer
            print(f"    Answer: {answer[:120]}")

        if callable(getattr(agent, "close", None)):
            agent.close()

    total_wall = round(time.perf_counter() - run_start, 3)

    # ── evaluate checks ───────────────────────────────────────────────────────
    check_results = [_eval_check(c, turn_answers, all_turns) for c in checks]
    passed = sum(1 for c in check_results if c["passed"])
    print(f"\n[runner] checks: {passed}/{len(check_results)} passed")
    for c in check_results:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['id']}: {c['description']} — {c['detail']}")

    # ── aggregate totals ──────────────────────────────────────────────────────
    totals = {
        "input_tokens":  sum(t["input_tokens"]  for t in all_turns),
        "cached_tokens": sum(t["cached_tokens"] for t in all_turns),
        "output_tokens": sum(t["output_tokens"] for t in all_turns),
        "model_calls":   sum(t["model_calls"]   for t in all_turns),
        "cost_usd":      round(sum(t["cost_usd"] for t in all_turns), 6),
        "wall_time_s":   total_wall,
    }

    results = {
        "scenario":      str(scenario_path),
        "harness_lines": count_harness_lines(harness_dir),
        "totals":        totals,
        "turns":         all_turns,
        "checks":        check_results,
        "checks_passed": passed,
        "checks_total":  len(check_results),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[runner] results written to {out_path}")
    return results
