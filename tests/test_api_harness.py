"""
tests/test_api_harness.py — offline tests for API/harness/.
Uses a FakeClient that returns scripted responses; no API key needed.
"""
import json
import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# ── make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers to build fake OpenAI response objects ────────────────────────────

def _make_usage(prompt=100, completion=20, cached=0):
    details = SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=details,
    )


def _make_tool_call(tc_id: str, name: str, args: dict, extra_content: dict | None = None):
    tc = SimpleNamespace(
        id=tc_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )
    if extra_content is not None:
        tc.extra_content = extra_content
    return tc


def _make_response(content=None, tool_calls=None, finish="stop", usage=None):
    tcs = tool_calls or []

    def _tc_dump(tc):
        d = {
            "id":       tc.id,
            "type":     tc.type,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        if hasattr(tc, "extra_content"):
            d["extra_content"] = tc.extra_content
        return d

    def _msg_dump(exclude_none=False):
        d = {"role": "assistant"}
        if content is not None:
            d["content"] = content
        if tcs:
            d["tool_calls"] = [_tc_dump(tc) for tc in tcs]
        return d

    msg = SimpleNamespace(
        content=content,
        tool_calls=tcs,
        model_dump=_msg_dump,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage or _make_usage())


class FakeClient:
    """Returns scripted responses in order; raises if exhausted."""
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.chat = self          # client.chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        if not self._responses:
            raise RuntimeError("FakeClient: no more scripted responses")
        return self._responses.pop(0)


# ── redirect data files to tmp_path ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    import shared.tools as t
    import shared.memory as m
    import API.harness.history as h
    monkeypatch.setattr(t, "DATA_DIR",       tmp_path)
    monkeypatch.setattr(t, "EXPENSES_FILE",  tmp_path / "expenses.json")
    monkeypatch.setattr(t, "BUDGETS_FILE",   tmp_path / "budgets.json")
    monkeypatch.setattr(m, "MEMORY_FILE",    tmp_path / "memory.json")
    monkeypatch.setattr(h, "_DEFAULT_PATH",  tmp_path / "history.jsonl")


# ═════════════════════════════════════════════════════════════════════════════
# (a) loop handles two parallel tool_calls and returns two tool messages
# ═════════════════════════════════════════════════════════════════════════════

def test_loop_parallel_tool_calls(tmp_path):
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    tc1 = _make_tool_call("id1", "set_budget_limit", {"category": "food", "limit_ils": 1500})
    tc2 = _make_tool_call("id2", "set_budget_limit", {"category": "rent", "limit_ils": 3000})

    responses = [
        _make_response(tool_calls=[tc1, tc2], finish="tool_calls"),
        _make_response(content="Done!", finish="stop"),
    ]
    client  = FakeClient(responses)
    usage   = TurnUsage("gpt-4o-mini")
    system  = {"role": "system", "content": "You are a budget assistant."}

    with patch("API.harness.loop.time.sleep"):
        answer, new_msgs = run_loop(client, "gpt-4o-mini", system, [], usage)

    assert answer == "Done!"
    # new_msgs: assistant(tool_calls) + tool(id1) + tool(id2) + assistant(final)
    tool_msgs = [m for m in new_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    ids = {m["tool_call_id"] for m in tool_msgs}
    assert ids == {"id1", "id2"}
    assert usage.model_calls == 2


# ═════════════════════════════════════════════════════════════════════════════
# (b) tool exception becomes ERROR tool message and loop continues
# ═════════════════════════════════════════════════════════════════════════════

def test_loop_tool_exception_becomes_error_message(tmp_path):
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    # add_expense with negative amount → raises ValueError
    tc = _make_tool_call("id3", "add_expense", {"amount": -50, "category": "food"})

    responses = [
        _make_response(tool_calls=[tc], finish="tool_calls"),
        _make_response(content="I couldn't add that expense.", finish="stop"),
    ]
    client = FakeClient(responses)
    usage  = TurnUsage("gpt-4o-mini")
    system = {"role": "system", "content": "assistant"}

    with patch("API.harness.loop.time.sleep"):
        answer, new_msgs = run_loop(client, "gpt-4o-mini", system, [], usage)

    tool_msgs = [m for m in new_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith("ERROR:")
    assert answer == "I couldn't add that expense."


# ═════════════════════════════════════════════════════════════════════════════
# (c) history append-only and reload
# ═════════════════════════════════════════════════════════════════════════════

def test_history_append_and_reload(tmp_path):
    from API.harness import history

    path = tmp_path / "hist.jsonl"
    m1 = {"role": "user",      "content": "hello"}
    m2 = {"role": "assistant", "content": "hi"}
    m3 = {"role": "user",      "content": "bye"}

    history.append(m1, path)
    history.append(m2, path)
    history.append(m3, path)

    loaded = history.load(path)
    assert loaded == [m1, m2, m3]

    # Overwrite (compaction) then reload
    history.overwrite([m2], path)
    assert history.load(path) == [m2]


# ═════════════════════════════════════════════════════════════════════════════
# (d) context.py excludes irrelevant memory
# ═════════════════════════════════════════════════════════════════════════════

def test_context_excludes_irrelevant_memory(monkeypatch):
    from API.harness import context
    import API.harness.memory as harness_mem

    # Inject a fake get_relevant_for_context that returns controlled data
    # This tests that context.build_messages correctly includes only what
    # get_relevant_for_context returns (i.e. relevant entries only).
    fake_memory = {"food_limit": "1500 ILS"}
    monkeypatch.setattr(harness_mem, "get_relevant_for_context", lambda q: fake_memory)

    system_msg, window = context.build_messages("what is my food budget?", [])
    content = system_msg["content"]

    assert "food_limit" in content
    assert "rent_limit" not in content   # was never in fake_memory


# ═════════════════════════════════════════════════════════════════════════════
# (e) compaction triggers over threshold
# ═════════════════════════════════════════════════════════════════════════════

def test_compaction_triggers_over_threshold(tmp_path, monkeypatch):
    import API.harness.compaction as comp
    monkeypatch.setattr(comp, "COMPACTION_THRESHOLD", 10)  # very low threshold

    big_messages = [
        {"role": "user",      "content": "x" * 30},
        {"role": "assistant", "content": "y" * 30},
    ]
    assert comp.needs_compaction(big_messages) is True

    # compact() should call client once and return a single summary message
    fake_resp = _make_response(content="Summary: user asked about budget.")
    client    = FakeClient([fake_resp])
    with patch("API.harness.loop.time.sleep"):
        result = comp.compact(big_messages, client, "gpt-4o-mini")

    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert "Summary" in result[0]["content"]


def test_compaction_does_not_trigger_under_threshold(monkeypatch):
    import API.harness.compaction as comp
    monkeypatch.setattr(comp, "COMPACTION_THRESHOLD", 100_000)
    msgs = [{"role": "user", "content": "hi"}]
    assert comp.needs_compaction(msgs) is False


# ═════════════════════════════════════════════════════════════════════════════
# (f) API/main.py reconfigures stdout/stderr to utf-8 at startup
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# (g) loop.py logs BadRequestError response body and re-raises
# ═════════════════════════════════════════════════════════════════════════════

def test_loop_bad_request_logs_and_reraises(capsys):
    import openai
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    # openai.BadRequestError requires response.request; build a minimal fake
    fake_request  = SimpleNamespace(method="POST", url="https://api.openai.com")
    fake_response = SimpleNamespace(
        text='{"error":"unsupported"}',
        request=fake_request,
        status_code=400,
        headers={},
    )
    err = openai.BadRequestError(
        message="Bad Request",
        response=fake_response,
        body={"error": "unsupported"},
    )

    class ErrorClient:
        chat = property(lambda self: self)
        completions = property(lambda self: self)
        def create(self, **_):
            raise err

    usage = TurnUsage("gpt-4o-mini")
    with patch("API.harness.loop.time.sleep"):
        with pytest.raises(openai.BadRequestError):
            run_loop(ErrorClient(), "gpt-4o-mini",
                     {"role": "system", "content": "s"}, [], usage)

    captured = capsys.readouterr().out
    assert "[loop.py] BadRequestError" in captured
    assert "unsupported" in captured


# ═════════════════════════════════════════════════════════════════════════════
# (g2) loop.py logs RateLimitError response body and re-raises
# ═════════════════════════════════════════════════════════════════════════════

def test_loop_rate_limit_logs_and_reraises(capsys):
    import openai
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    fake_request  = SimpleNamespace(method="POST", url="https://api.openai.com")
    fake_response = SimpleNamespace(
        text='{"error":"rate_limit_exceeded"}',
        request=fake_request,
        status_code=429,
        headers={},
    )
    err = openai.RateLimitError(
        message="Rate limit exceeded",
        response=fake_response,
        body={"error": "rate_limit_exceeded"},
    )

    class RateLimitClient:
        chat = property(lambda self: self)
        completions = property(lambda self: self)
        def create(self, **_):
            raise err

    usage = TurnUsage("gpt-4o-mini")
    with patch("API.harness.loop.time.sleep"):
        with pytest.raises(openai.RateLimitError):
            run_loop(RateLimitClient(), "gpt-4o-mini",
                     {"role": "system", "content": "s"}, [], usage)

    captured = capsys.readouterr().out
    assert "[loop.py] RateLimitError" in captured
    assert "rate_limit_exceeded" in captured


# ═════════════════════════════════════════════════════════════════════════════
# (h) context.py trims orphan leading messages from the history window
# ═════════════════════════════════════════════════════════════════════════════

def test_context_trims_orphan_tool_message(monkeypatch):
    """Window starting with a tool message must be trimmed to the next user message."""
    from API.harness import context
    import API.harness.memory as harness_mem
    monkeypatch.setattr(harness_mem, "get_relevant_for_context", lambda q: {})
    monkeypatch.setattr(context, "HISTORY_WINDOW", 10)

    history = [
        {"role": "tool",      "tool_call_id": "x", "content": "result"},
        {"role": "user",      "content": "next question"},
        {"role": "assistant", "content": "answer"},
    ]
    _, window = context.build_messages("next question", history)
    assert window[0]["role"] == "user"
    assert len(window) == 2


def test_context_trims_orphan_assistant_tool_calls(monkeypatch):
    """Window starting with an assistant message that has tool_calls must be trimmed."""
    from API.harness import context
    import API.harness.memory as harness_mem
    monkeypatch.setattr(harness_mem, "get_relevant_for_context", lambda q: {})
    monkeypatch.setattr(context, "HISTORY_WINDOW", 10)

    history = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "add_expense", "arguments": "{}"}}]},
        {"role": "tool",      "tool_call_id": "c1", "content": "ok"},
        {"role": "user",      "content": "what next?"},
    ]
    _, window = context.build_messages("what next?", history)
    assert window[0]["role"] == "user"
    assert len(window) == 1


def test_context_clean_window_untouched(monkeypatch):
    """Window already starting with a user message must not be modified."""
    from API.harness import context
    import API.harness.memory as harness_mem
    monkeypatch.setattr(harness_mem, "get_relevant_for_context", lambda q: {})
    monkeypatch.setattr(context, "HISTORY_WINDOW", 10)

    history = [
        {"role": "user",      "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user",      "content": "bye"},
    ]
    _, window = context.build_messages("bye", history)
    assert window == history


# ═════════════════════════════════════════════════════════════════════════════
# (i) runner wipes shared/data files before the scenario starts
# ═════════════════════════════════════════════════════════════════════════════

def test_runner_wipes_data_before_run(tmp_path, monkeypatch):
    """run_scenario must delete history/memory/expenses/budgets before starting."""
    import shared.runner as runner_mod
    import shared.tools as t
    import shared.memory as m

    # Point shared data at tmp_path and pre-create stale files there
    stale_data_dir = tmp_path / "data"
    stale_data_dir.mkdir()
    stale_history  = stale_data_dir / "history.jsonl"
    stale_expenses = stale_data_dir / "expenses.json"
    stale_history.write_text('{"role":"user","content":"stale"}\n', encoding="utf-8")
    stale_expenses.write_text('[{"id":"old"}]', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "_SHARED_DATA", stale_data_dir)
    monkeypatch.setattr(runner_mod, "_DATA_FILES",
                        ("history.jsonl", "expenses.json"))
    monkeypatch.setattr(t, "DATA_DIR",      tmp_path)
    monkeypatch.setattr(t, "EXPENSES_FILE", tmp_path / "expenses.json")
    monkeypatch.setattr(t, "BUDGETS_FILE",  tmp_path / "budgets.json")
    monkeypatch.setattr(m, "MEMORY_FILE",   tmp_path / "memory.json")

    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps({"sessions": [{"session_id": "s1",
                                   "turns": [{"turn": 1, "user": "hi"}]}],
                    "checks": []}),
        encoding="utf-8",
    )
    harness_dir = tmp_path / "h"
    harness_dir.mkdir()
    (harness_dir / "x.py").write_text("x = 1", encoding="utf-8")

    class _FakeAgent:
        def run_turn(self, _): return {"answer": "ok", "metrics": {}}

    runner_mod.run_scenario(scenario_path, _FakeAgent, harness_dir,
                            tmp_path / "out.json")

    assert not stale_history.exists(),  "history.jsonl should have been wiped"
    assert not stale_expenses.exists(), "expenses.json should have been wiped"


# ═════════════════════════════════════════════════════════════════════════════
# (j) thought_signature in tool_call extra_content is preserved end-to-end
# ═════════════════════════════════════════════════════════════════════════════

def test_thought_signature_preserved_in_next_request(tmp_path):
    """
    A tool_call whose extra_content contains a Gemini thought_signature must
    appear unchanged in the messages list of the very next model call.
    """
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    thought_sig = {"google": {"thought_signature": "abc"}}
    tc = _make_tool_call(
        "id_ts", "set_budget_limit",
        {"category": "food", "limit_ils": 1500},
        extra_content=thought_sig,
    )

    # Turn 1: model returns a tool_call with thought_signature
    # Turn 2: model returns a final answer; we capture what messages it received
    received_messages: list = []

    class CapturingClient:
        def __init__(self, responses):
            self._responses = list(responses)
            self.chat = self
            self.completions = self
        def create(self, **kwargs):
            received_messages.append(list(kwargs["messages"]))
            return self._responses.pop(0)

    responses = [
        _make_response(tool_calls=[tc], finish="tool_calls"),
        _make_response(content="Budget set.", finish="stop"),
    ]
    client = CapturingClient(responses)
    usage  = TurnUsage("gpt-4o-mini")
    system = {"role": "system", "content": "assistant"}

    with patch("API.harness.loop.time.sleep"):
        answer, new_msgs = run_loop(client, "gpt-4o-mini", system, [], usage)

    assert answer == "Budget set."
    # The second call's messages must include the assistant record with extra_content
    second_call_msgs = received_messages[1]
    assistant_msgs = [m for m in second_call_msgs if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1
    tcs_in_msg = assistant_msgs[0].get("tool_calls", [])
    assert len(tcs_in_msg) == 1
    assert tcs_in_msg[0].get("extra_content") == thought_sig


# ═════════════════════════════════════════════════════════════════════════════
# (k) loop.py sleeps before every model call
# ═════════════════════════════════════════════════════════════════════════════

def test_loop_sleeps_before_model_call(monkeypatch):
    """run_loop must call time.sleep once per model call."""
    from API.harness.loop   import run_loop
    from API.harness.tokens import TurnUsage

    sleep_calls = []
    monkeypatch.setenv("MODEL_CALL_DELAY", "4")

    responses = [_make_response(content="ok", finish="stop")]
    client = FakeClient(responses)
    usage  = TurnUsage("gpt-4o-mini")
    system = {"role": "system", "content": "s"}

    with patch("API.harness.loop.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        run_loop(client, "gpt-4o-mini", system, [], usage)

    assert sleep_calls == [4.0]


def test_api_main_reconfigures_streams(monkeypatch):
    """main() must call reconfigure(encoding='utf-8', errors='replace') on both streams."""
    import API.main as api_main

    stdout_calls, stderr_calls = [], []
    monkeypatch.setattr(sys.stdout, "reconfigure",
                        lambda **kw: stdout_calls.append(kw), raising=False)
    monkeypatch.setattr(sys.stderr, "reconfigure",
                        lambda **kw: stderr_calls.append(kw), raising=False)
    # Stop main() before it tries to do real work
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(SystemExit):
        # No --scenario and no real agent — will hit sys.exit(1) on missing key
        # but reconfigure runs before the key check, so we just need it called.
        # Patch Agent so __init__ doesn't run.
        with patch("API.main.Agent"):
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)
            api_main.main()

    assert stdout_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr_calls == [{"encoding": "utf-8", "errors": "replace"}]
