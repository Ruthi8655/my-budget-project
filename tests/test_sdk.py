"""
tests/test_sdk.py — offline tests for SDK/ using ScriptedModel (no API key needed).
"""
import asyncio
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

from agents import Agent, Runner, RunConfig, set_tracing_disabled
from agents.testing.model import ScriptedModel, assistant_message, function_call
from agents.memory.sqlite_session import SQLiteSession

set_tracing_disabled(True)

ROOT = Path(__file__).parent.parent


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    import shared.tools as t
    import shared.memory as m
    monkeypatch.setattr(t, "DATA_DIR",      tmp_path)
    monkeypatch.setattr(t, "EXPENSES_FILE", tmp_path / "expenses.json")
    monkeypatch.setattr(t, "BUDGETS_FILE",  tmp_path / "budgets.json")
    monkeypatch.setattr(m, "MEMORY_FILE",   tmp_path / "memory.json")


def _make_agent(model: ScriptedModel) -> Agent:
    """Build a minimal BudgetAssistant agent backed by a ScriptedModel."""
    from SDK.tools import ALL_TOOLS
    return Agent(
        name="BudgetAssistant",
        instructions="You are a concise budget assistant.",
        tools=ALL_TOOLS,
        model=model,
    )


def _run(agent: Agent, text: str, session=None) -> object:
    cfg = RunConfig(tracing_disabled=True)
    return asyncio.run(Runner.run(agent, input=text, run_config=cfg, session=session))


# ═════════════════════════════════════════════════════════════════════════════
# (a) single tool call executes and returns final answer
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_single_tool_call():
    model = ScriptedModel([
        [function_call("set_budget_limit", {"category": "food", "limit_ils": 1500}, call_id="c1")],
        [assistant_message("Budget set to 1500 ILS.")],
    ])
    agent  = _make_agent(model)
    result = _run(agent, "Set food budget to 1500 ILS")

    assert "1500" in str(result.final_output)
    assert model.remaining_steps == 0


# ═════════════════════════════════════════════════════════════════════════════
# (b) tool exception is returned as ERROR string and loop continues
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_tool_exception_continues():
    # add_expense with negative amount → tool returns "ERROR: ..."
    model = ScriptedModel([
        [function_call("add_expense", {"amount": -10, "category": "food"}, call_id="c2")],
        [assistant_message("I couldn't add that expense.")],
    ])
    agent  = _make_agent(model)
    result = _run(agent, "Add -10 to food")

    # The model received the ERROR tool output and produced a final answer
    tool_outputs = [
        item for item in result.new_items
        if getattr(item, "type", None) == "tool_call_output_item"
    ]
    assert any("ERROR" in str(getattr(o, "output", "")) for o in tool_outputs)
    assert result.final_output is not None


# ═════════════════════════════════════════════════════════════════════════════
# (c) SQLiteSession persists history across two separate Runner.run calls
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_session_persists(tmp_path):
    db = tmp_path / "test_sessions.db"
    session_id = "test-persist"

    model1 = ScriptedModel([[assistant_message("Hello!")]])
    agent1 = _make_agent(model1)
    session1 = SQLiteSession(session_id, db_path=str(db))
    _run(agent1, "Hi", session=session1)

    # Second run with a fresh session object pointing to the same DB
    model2 = ScriptedModel([[assistant_message("I remember you.")]])
    agent2 = _make_agent(model2)
    session2 = SQLiteSession(session_id, db_path=str(db))
    result2  = _run(agent2, "Do you remember me?", session=session2)

    # The second run received prior history from the DB
    assert result2.final_output is not None
    # History should contain items from both runs
    history = asyncio.run(session2.get_items())
    assert len(history) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# (d) usage metrics are extracted correctly from result
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_usage_extraction():
    from agents import Usage
    from agents.usage import InputTokensDetails

    model = ScriptedModel(
        [[assistant_message("Done.")]],
        default_usage=Usage(requests=1, input_tokens=50, output_tokens=10, total_tokens=60),
    )
    agent  = _make_agent(model)
    result = _run(agent, "Hello")

    from SDK.agent import _extract_metrics
    metrics = _extract_metrics(result, "gpt-4o-mini", 0.5)

    assert metrics["input_tokens"]  == 50
    assert metrics["output_tokens"] == 10
    assert metrics["model_calls"]   == 1
    assert metrics["cost_usd"]      >= 0
    assert metrics["wall_time_s"]   == 0.5


# ═════════════════════════════════════════════════════════════════════════════
# (e) save_memory tool writes to shared memory store
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_save_memory_tool():
    import shared.memory as mem_mod

    model = ScriptedModel([
        [function_call("save_memory", {"key": "food_limit", "value": "1500 ILS"}, call_id="c3")],
        [assistant_message("Saved.")],
    ])
    agent  = _make_agent(model)
    _run(agent, "Remember food limit is 1500 ILS")

    assert mem_mod.read_memory().get("food_limit") == "1500 ILS"


# ═════════════════════════════════════════════════════════════════════════════
# (f) ALL_TOOLS contains exactly the expected tool names
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_tool_names():
    from SDK.tools import ALL_TOOLS
    names = {t.name for t in ALL_TOOLS}
    assert names == {
        "add_expense", "get_summary", "set_budget_limit",
        "convert_currency", "search_expenses", "save_memory",
    }


# ═════════════════════════════════════════════════════════════════════════════
# (h) BudgetAgent.close() shuts the loop; double-close is safe
# ═════════════════════════════════════════════════════════════════════════════

def test_budget_agent_close(tmp_path, monkeypatch):
    """close() must close the loop; calling it twice must not raise."""
    from unittest.mock import MagicMock, patch
    import SDK.agent as sdk_mod

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    fake_client = MagicMock()
    with patch.object(sdk_mod, "AsyncOpenAI", return_value=fake_client), \
         patch.object(sdk_mod, "SQLiteSession", return_value=MagicMock()), \
         patch.object(sdk_mod, "Agent", return_value=MagicMock()), \
         patch.object(sdk_mod.time, "sleep"):
        agent = sdk_mod.BudgetAgent(
            session_id="test-close",
            db_path=tmp_path / "s.db",
        )

    assert not agent._loop.is_closed()
    agent.close()
    assert agent._loop.is_closed()
    agent.close()   # second call must be a no-op, not raise


# ═════════════════════════════════════════════════════════════════════════════
# (i) BudgetAgent uses max_retries=6 on its AsyncOpenAI client
# ═════════════════════════════════════════════════════════════════════════════

def test_budget_agent_max_retries(tmp_path, monkeypatch):
    """AsyncOpenAI must be constructed with max_retries=6."""
    from unittest.mock import MagicMock, patch, call
    import SDK.agent as sdk_mod

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured = {}

    def _fake_async_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch.object(sdk_mod, "AsyncOpenAI", side_effect=_fake_async_openai), \
         patch.object(sdk_mod, "SQLiteSession", return_value=MagicMock()), \
         patch.object(sdk_mod, "Agent", return_value=MagicMock()), \
         patch.object(sdk_mod.time, "sleep"):
        agent = sdk_mod.BudgetAgent(
            session_id="test-retries",
            db_path=tmp_path / "s.db",
        )
        agent.close()

    assert captured.get("max_retries") == 6

def test_shared_pricing_consistency():
    from shared.pricing import compute_cost, PRICE_TABLE
    # gpt-4o-mini: 1M input = $0.15
    cost = compute_cost("gpt-4o-mini", 1_000_000, 0, 0)
    assert abs(cost - 0.15) < 1e-9
    # cached tokens cheaper
    cost_cached = compute_cost("gpt-4o-mini", 1_000_000, 1_000_000, 0)
    assert cost_cached < cost
    # both versions see the same table
    assert "gpt-4o-mini" in PRICE_TABLE
    assert "gpt-4o" in PRICE_TABLE


# ═════════════════════════════════════════════════════════════════════════════
# (j) SDK/main.py reconfigures stdout/stderr to utf-8 at startup
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# (k) SDK/agent.py sleeps before every Runner.run call
# ═════════════════════════════════════════════════════════════════════════════

def test_sdk_agent_sleeps_before_run_turn(tmp_path, monkeypatch):
    """BudgetAgent.run_turn must call time.sleep once per turn."""
    from unittest.mock import MagicMock, patch, AsyncMock
    import SDK.agent as sdk_mod

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_CALL_DELAY", "4")

    sleep_calls = []
    fake_result = MagicMock()
    fake_result.final_output = "done"
    fake_result.context_wrapper.usage = None

    with patch.object(sdk_mod, "AsyncOpenAI", return_value=MagicMock()), \
         patch.object(sdk_mod, "SQLiteSession", return_value=MagicMock()), \
         patch.object(sdk_mod, "Agent", return_value=MagicMock()), \
         patch.object(sdk_mod.time, "sleep", side_effect=lambda s: sleep_calls.append(s)), \
         patch.object(sdk_mod.Runner, "run", new=AsyncMock(return_value=fake_result)):
        agent = sdk_mod.BudgetAgent(session_id="t", db_path=tmp_path / "s.db")
        agent.run_turn("hello")
        agent.close()

    assert sleep_calls == [4.0]


def test_sdk_main_reconfigures_streams(monkeypatch):
    """main() must call reconfigure(encoding='utf-8', errors='replace') on both streams."""
    import SDK.main as sdk_main

    stdout_calls, stderr_calls = [], []
    monkeypatch.setattr(sys.stdout, "reconfigure",
                        lambda **kw: stdout_calls.append(kw), raising=False)
    monkeypatch.setattr(sys.stderr, "reconfigure",
                        lambda **kw: stderr_calls.append(kw), raising=False)
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        sdk_main.main()

    assert stdout_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr_calls == [{"encoding": "utf-8", "errors": "replace"}]
