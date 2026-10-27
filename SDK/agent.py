"""
SDK/agent.py — OpenAI Agents SDK version of the budget assistant.

SDK replaces hand-written harness files:
  # replaces: loop.py      — Runner.run() drives the tool-call loop automatically
  # replaces: history.py   — SQLiteSession persists conversation to disk; resumes across processes
  # replaces: context.py   — system prompt injected via Agent(instructions=...)
  # replaces: caching.py   — SDK manages message ordering; no manual prefix-cache control needed
                             (OpenAI automatic prefix caching still applies server-side)

SDK does NOT provide:
  # NO equivalent of compaction.py — the SDK has no built-in summarisation/compaction;
    the full session history is sent each turn (token cost grows unbounded without manual work).
  # NO equivalent of caching.py cache_control — OpenAI Chat Completions has no cache_control
    field; prefix caching is automatic and invisible (same as API version).
"""
import asyncio
import os
import time
from pathlib import Path

from agents import Agent, Runner, RunConfig, set_tracing_disabled  # replaces: loop.py
from agents.memory.sqlite_session import SQLiteSession             # replaces: history.py
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from shared.pricing import compute_cost
from SDK.tools import ALL_TOOLS

# ── paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR    = Path(__file__).parent.parent / "shared" / "data"
_DB_PATH     = _DATA_DIR / "sdk_sessions.db"
_MEMORY_FILE = _DATA_DIR / "memory.md"

# ── system prompt (identical text to API/harness/context.py) ─────────────────
_BASE_INSTRUCTIONS = (
    "You are a concise personal budget assistant. "
    "Use the provided tools to record expenses, query summaries, set limits, "
    "convert currencies, and manage memory. "
    "Always reply briefly. Amounts are in ILS unless stated otherwise."
)


def _load_memory_md() -> str:
    """Load the memory markdown file and append it to instructions."""
    if _MEMORY_FILE.exists():
        content = _MEMORY_FILE.read_text(encoding="utf-8").strip()
        if content:
            return f"\n\n## Remembered preferences\n{content}"
    return ""


def _instructions(context, agent) -> str:
    return _BASE_INSTRUCTIONS + _load_memory_md()


def _append_memory_md(key: str, value: str) -> None:
    """Append a key=value line to the memory markdown file."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {key}: {value}\n")
    print(f"[SDK/agent.py] memory.md updated: {key}={value}")


def _extract_metrics(result, model: str, wall_time: float) -> dict:
    """
    Extract token usage from result.context_wrapper.usage (SDK Usage object).
    Cached tokens live in usage.input_tokens_details.cached_tokens.
    Falls back gracefully if fields are absent.
    """
    usage = getattr(result.context_wrapper, "usage", None)
    if usage is None:
        return {"model_calls": 0, "input_tokens": 0, "cached_tokens": 0,
                "output_tokens": 0, "cost_usd": 0.0, "wall_time_s": wall_time}

    input_tok  = getattr(usage, "input_tokens",  0) or 0
    output_tok = getattr(usage, "output_tokens", 0) or 0
    requests   = getattr(usage, "requests",      0) or 0
    details    = getattr(usage, "input_tokens_details", None)
    cached_tok = getattr(details, "cached_tokens", 0) or 0 if details else 0

    print(f"[SDK/agent.py] usage: requests={requests} in={input_tok} "
          f"cached={cached_tok} out={output_tok}")

    return {
        "model_calls":   requests,
        "input_tokens":  input_tok,
        "cached_tokens": cached_tok,
        "output_tokens": output_tok,
        "cost_usd":      round(compute_cost(model, input_tok, cached_tok, output_tok), 6),
        "wall_time_s":   wall_time,
    }


class BudgetAgent:
    def __init__(
        self,
        session_id: str = "default",
        db_path: Path | None = None,
        memory_file: Path | None = None,
    ):
        # Disable tracing export so nothing leaves the process except model calls
        set_tracing_disabled(True)  # replaces: no equivalent in API harness (tracing is SDK-only)

        self.model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self._memory_file = memory_file or _MEMORY_FILE
        self._db_path     = db_path or _DB_PATH

        # One persistent event loop for the lifetime of this agent instance
        self._loop = asyncio.new_event_loop()

        # Reused async OpenAI client — built once, shared across all turns
        self._openai_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),  # None → default OpenAI
            max_retries=6,
        )

        # SQLiteSession persists conversation across processes  # replaces: history.py
        self._session = SQLiteSession(
            session_id=session_id,
            db_path=str(self._db_path),
        )

        # Agent with restricted tool list (no built-in extras)  # replaces: loop.py + context.py
        self._agent = Agent(
            name="BudgetAssistant",
            instructions=_instructions,          # callable → re-evaluated each turn (context, agent)
            tools=ALL_TOOLS,                    # only our tools, no built-ins
            model=OpenAIChatCompletionsModel(
                model=self.model,
                openai_client=self._openai_client,
            ),
        )

        self._run_config = RunConfig(tracing_disabled=True)
        print(f"[SDK/agent.py] initialised model={self.model} session={session_id}")

    def run_turn(self, user_text: str) -> dict:
        print(f"[SDK/agent.py] run_turn: '{user_text[:60]}'")
        start = time.perf_counter()

        # Pause before every model call to avoid rate-limit bursts (identical to API version)
        time.sleep(float(os.getenv("MODEL_CALL_DELAY", "4")))
        result = self._loop.run_until_complete(
            Runner.run(
                self._agent,
                input=user_text,
                session=self._session,      # replaces: history.py (persistence + reload)
                run_config=self._run_config,
            )
        )

        wall = round(time.perf_counter() - start, 3)
        answer  = str(result.final_output or "")
        metrics = _extract_metrics(result, self.model, wall)
        print(f"[SDK/agent.py] turn done — {metrics}")
        return {"answer": answer, "metrics": metrics}

    def close(self) -> None:
        """Close the persistent event loop. Call when the agent is no longer needed."""
        if not self._loop.is_closed():
            self._loop.close()
            print("[SDK/agent.py] event loop closed")
