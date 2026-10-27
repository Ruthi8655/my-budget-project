"""
tests/test_runner.py — offline tests for shared/runner.py.

All tests use a FakeAgent that writes real data to tmp_path via monkeypatched
shared.tools / shared.memory paths, then verifies the runner's check logic
against that persisted state.  No API key needed.
"""
import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch


ROOT = Path(__file__).parent.parent


# ── redirect shared data files to tmp_path ────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    import shared.tools as t
    import shared.memory as m
    monkeypatch.setattr(t, "DATA_DIR",      tmp_path)
    monkeypatch.setattr(t, "EXPENSES_FILE", tmp_path / "expenses.json")
    monkeypatch.setattr(t, "BUDGETS_FILE",  tmp_path / "budgets.json")
    monkeypatch.setattr(m, "MEMORY_FILE",   tmp_path / "memory.json")
    return tmp_path


# ── minimal fake agent ────────────────────────────────────────────────────────

class FakeAgent:
    """
    Executes real shared.tools calls so data is persisted to disk,
    then returns a scripted answer.
    """
    def __init__(self, turns: list[dict]):
        """turns: list of {answer, side_effect} where side_effect is a callable."""
        self._turns = list(turns)
        self._idx   = 0

    def run_turn(self, user_text: str) -> dict:
        spec = self._turns[self._idx % len(self._turns)]
        self._idx += 1
        if "side_effect" in spec:
            spec["side_effect"]()
        return {
            "answer":  spec.get("answer", "ok"),
            "metrics": {
                "input_tokens":  spec.get("input_tokens",  10),
                "cached_tokens": spec.get("cached_tokens",  0),
                "output_tokens": spec.get("output_tokens",  5),
                "model_calls":   spec.get("model_calls",    1),
                "tools":         spec.get("tools",         []),
                "cost_usd":      spec.get("cost_usd",    0.001),
                "wall_time_s":   spec.get("wall_time_s",  0.1),
            },
        }


def _make_scenario(tmp_path: Path, sessions: list[dict], checks: list[dict]) -> Path:
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps({"sessions": sessions, "checks": checks}), encoding="utf-8")
    return p


def _make_harness_dir(tmp_path: Path, lines: list[str]) -> Path:
    d = tmp_path / "harness"
    d.mkdir()
    (d / "mod.py").write_text("\n".join(lines), encoding="utf-8")
    return d


# ═════════════════════════════════════════════════════════════════════════════
# (1) runner writes results JSON with required top-level keys
# ═════════════════════════════════════════════════════════════════════════════

def test_runner_results_json_structure(tmp_path):
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "hello"}]}
    ], checks=[])
    harness = _make_harness_dir(tmp_path, ["def foo():", "    pass"])
    out = tmp_path / "out.json"

    run_scenario(scenario, lambda: FakeAgent([{"answer": "hi"}]), harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    for key in ("scenario", "harness_lines", "totals", "turns", "checks",
                "checks_passed", "checks_total"):
        assert key in result, f"missing key: {key}"

    totals = result["totals"]
    for k in ("input_tokens", "cached_tokens", "output_tokens",
              "model_calls", "cost_usd", "wall_time_s"):
        assert k in totals


# ═════════════════════════════════════════════════════════════════════════════
# (2) harness line counter ignores blank lines and comments
# ═════════════════════════════════════════════════════════════════════════════

def test_count_harness_lines(tmp_path):
    from shared.runner import count_harness_lines

    d = tmp_path / "h"
    d.mkdir()
    (d / "a.py").write_text(
        "# comment\n\ndef foo():\n    return 1\n",
        encoding="utf-8",
    )
    assert count_harness_lines(d) == 2   # only "def foo():" and "    return 1"


# ═════════════════════════════════════════════════════════════════════════════
# (3) check c1 — budgets_file: food budget persisted to disk
# ═════════════════════════════════════════════════════════════════════════════

def test_check_budgets_file_pass(tmp_path):
    from shared.runner import run_scenario
    from shared.tools import set_budget_limit

    def _set_food():
        set_budget_limit("food", 1500.0)

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "set food budget"}]}
    ], checks=[
        {"id": "c1", "description": "food budget = 1500",
         "type": "budgets_file", "key": "food", "op": "eq", "value": 1500.0}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario, lambda: FakeAgent([{"answer": "done", "side_effect": _set_food}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c1 = next(c for c in result["checks"] if c["id"] == "c1")
    assert c1["passed"] is True


# ═════════════════════════════════════════════════════════════════════════════
# (4) check c2 — expenses_count: at least one expense recorded
# ═════════════════════════════════════════════════════════════════════════════

def test_check_expenses_count_pass(tmp_path):
    from shared.runner import run_scenario
    from shared.tools import add_expense

    def _add():
        add_expense(45, "food", "groceries")

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "add expense"}]}
    ], checks=[
        {"id": "c2", "description": "at least 1 expense",
         "type": "expenses_count", "op": "gte", "value": 1}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario, lambda: FakeAgent([{"answer": "added", "side_effect": _add}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c2 = next(c for c in result["checks"] if c["id"] == "c2")
    assert c2["passed"] is True


# ═════════════════════════════════════════════════════════════════════════════
# (5) check c3 — expenses_category: transport expense exists
# ═════════════════════════════════════════════════════════════════════════════

def test_check_expenses_category_pass(tmp_path):
    from shared.runner import run_scenario
    from shared.tools import add_expense

    def _add():
        add_expense(120, "transport", "bus pass")

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "add transport"}]}
    ], checks=[
        {"id": "c3", "description": "transport expense exists",
         "type": "expenses_category", "category": "transport", "op": "gte", "value": 1}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario, lambda: FakeAgent([{"answer": "added", "side_effect": _add}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c3 = next(c for c in result["checks"] if c["id"] == "c3")
    assert c3["passed"] is True


# ═════════════════════════════════════════════════════════════════════════════
# (6) check turn_answer_contains — plain keyword match
# ═════════════════════════════════════════════════════════════════════════════

def test_check_turn_answer_contains_pass(tmp_path):
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "search groceries"}]}
    ], checks=[
        {"id": "c4", "description": "answer mentions grocer",
         "type": "turn_answer_contains", "session_id": "s1", "turn": 1,
         "substring": "grocer"}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([{"answer": "Found 1 grocery expense."}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c4 = next(c for c in result["checks"] if c["id"] == "c4")
    assert c4["passed"] is True


def test_check_turn_answer_contains_fail(tmp_path):
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "search groceries"}]}
    ], checks=[
        {"id": "c4", "description": "answer mentions grocer",
         "type": "turn_answer_contains", "session_id": "s1", "turn": 1,
         "substring": "grocer"}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([{"answer": "Nothing found."}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c4 = next(c for c in result["checks"] if c["id"] == "c4")
    assert c4["passed"] is False


# ═════════════════════════════════════════════════════════════════════════════
# (6b) check c6 — number-format-tolerant: "1,500 ILS" and "1500.0" both pass
# ═════════════════════════════════════════════════════════════════════════════

def _c6_check():
    return {"id": "c6", "description": "food budget references 1500",
            "type": "turn_answer_contains", "session_id": "s1", "turn": 1,
            "substring": "1500"}


def test_c6_comma_formatted_passes(tmp_path):
    """Answer '1,500 ILS' must pass after comma-stripping."""
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "food budget?"}]}
    ], checks=[_c6_check()])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([{"answer": "Your food budget is 1,500 ILS."}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c6 = next(c for c in result["checks"] if c["id"] == "c6")
    assert c6["passed"] is True


def test_c6_plain_number_passes(tmp_path):
    """Answer '1500 ILS' must pass (no formatting to strip)."""
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "food budget?"}]}
    ], checks=[_c6_check()])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([{"answer": "Your food budget is 1500 ILS."}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c6 = next(c for c in result["checks"] if c["id"] == "c6")
    assert c6["passed"] is True


def test_c6_absent_number_fails(tmp_path):
    """Answer with no budget number must fail."""
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "food budget?"}]}
    ], checks=[_c6_check()])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([{"answer": "No budget has been set for food."}]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c6 = next(c for c in result["checks"] if c["id"] == "c6")
    assert c6["passed"] is False


# ═════════════════════════════════════════════════════════════════════════════
# (7) check c7 — total_model_calls across all turns
# ═════════════════════════════════════════════════════════════════════════════

def test_check_total_model_calls(tmp_path):
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [
            {"turn": 1, "user": "t1"},
            {"turn": 2, "user": "t2"},
            {"turn": 3, "user": "t3"},
        ]}
    ], checks=[
        {"id": "c7", "description": "total model calls >= 3",
         "type": "total_model_calls", "op": "gte", "value": 3}
    ])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([
                     {"answer": "a", "model_calls": 1},
                     {"answer": "b", "model_calls": 1},
                     {"answer": "c", "model_calls": 1},
                 ]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    c7 = next(c for c in result["checks"] if c["id"] == "c7")
    assert c7["passed"] is True


# ═════════════════════════════════════════════════════════════════════════════
# (8) fresh_agent=True causes make_agent to be called again for session-2
# ═════════════════════════════════════════════════════════════════════════════

def test_fresh_agent_called_per_session(tmp_path):
    from shared.runner import run_scenario

    call_count = {"n": 0}

    def _factory():
        call_count["n"] += 1
        return FakeAgent([{"answer": "ok"}])

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [{"turn": 1, "user": "hi"}]},
        {"session_id": "s2", "fresh_agent": True, "turns": [{"turn": 2, "user": "bye"}]},
    ], checks=[])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario, _factory, harness, out)

    assert call_count["n"] == 2   # one per session


# ═════════════════════════════════════════════════════════════════════════════
# (9) totals aggregate correctly across all turns
# ═════════════════════════════════════════════════════════════════════════════

def test_totals_aggregation(tmp_path):
    from shared.runner import run_scenario

    scenario = _make_scenario(tmp_path, sessions=[
        {"session_id": "s1", "turns": [
            {"turn": 1, "user": "t1"},
            {"turn": 2, "user": "t2"},
        ]}
    ], checks=[])
    harness = _make_harness_dir(tmp_path, ["x = 1"])
    out = tmp_path / "out.json"

    run_scenario(scenario,
                 lambda: FakeAgent([
                     {"answer": "a", "input_tokens": 100, "output_tokens": 20,
                      "cached_tokens": 10, "model_calls": 2, "cost_usd": 0.002},
                     {"answer": "b", "input_tokens": 200, "output_tokens": 30,
                      "cached_tokens":  5, "model_calls": 1, "cost_usd": 0.003},
                 ]),
                 harness, out)

    result = json.loads(out.read_text(encoding="utf-8"))
    t = result["totals"]
    assert t["input_tokens"]  == 300
    assert t["output_tokens"] == 50
    assert t["cached_tokens"] == 15
    assert t["model_calls"]   == 3
    assert abs(t["cost_usd"] - 0.005) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# (10) scenario.json has 7 turns across 2 sessions and 7 checks
# ═════════════════════════════════════════════════════════════════════════════

def test_scenario_json_structure():
    scenario_path = ROOT / "scenario.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))

    sessions = scenario["sessions"]
    assert len(sessions) == 2

    total_turns = sum(len(s["turns"]) for s in sessions)
    assert total_turns == 7

    # session-2 must be marked fresh_agent
    assert sessions[1].get("fresh_agent") is True

    checks = scenario["checks"]
    assert len(checks) >= 6


# ═════════════════════════════════════════════════════════════════════════════
# (11) compare.py loads two JSONs without error
# ═════════════════════════════════════════════════════════════════════════════

def test_compare_script_runs(tmp_path, capsys):
    import scripts.compare as cmp

    def _make_result(label: str) -> dict:
        return {
            "scenario": "scenario.json",
            "harness_lines": 100,
            "totals": {"input_tokens": 500, "cached_tokens": 50,
                       "output_tokens": 100, "model_calls": 7,
                       "cost_usd": 0.01, "wall_time_s": 3.5},
            "turns": [
                {"session_id": "s1", "turn": 1, "user": "hi", "answer": "ok",
                 "input_tokens": 500, "cached_tokens": 50, "output_tokens": 100,
                 "model_calls": 7, "tools": [], "cost_usd": 0.01, "wall_time_s": 3.5}
            ],
            "checks": [],
            "checks_passed": 0,
            "checks_total": 0,
        }

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_make_result("api")), encoding="utf-8")
    b.write_text(json.dumps(_make_result("sdk")), encoding="utf-8")

    cmp.compare(str(a), str(b))
    out = capsys.readouterr().out
    assert "input_tokens" in out.lower() or "Input tokens" in out
