"""
agent.py — wires the harness together.

Agent.run_turn(user_text) → {"answer": str, "metrics": dict}
"""
import os
import sys
from pathlib import Path
from openai import OpenAI

from API.harness import history, context, compaction
from API.harness.loop   import run_loop
from API.harness.tokens import TurnUsage

_HISTORY_PATH = Path(__file__).parent.parent / "shared" / "data" / "history.jsonl"


class Agent:
    def __init__(self, history_path: Path | None = None):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL"),  # None → default OpenAI
            max_retries=6,
        )
        self.model        = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.history_path = history_path or _HISTORY_PATH
        self._history     = history.load(self.history_path)
        print(f"[agent.py] initialised model={self.model}")

    def run_turn(self, user_text: str) -> dict:
        print(f"[agent.py] run_turn: '{user_text[:60]}'")
        usage = TurnUsage(self.model)

        # 1. Compact if needed (operates on full history)
        if compaction.needs_compaction(self._history):
            self._history = compaction.compact(self._history, self.client, self.model)
            history.overwrite(self._history, self.history_path)

        # 2. Append user message to history
        user_msg = {"role": "user", "content": user_text}
        self._history.append(user_msg)
        history.append(user_msg, self.history_path)

        # 3. Build context (system + relevant memory + windowed history)
        system_msg, window = context.build_messages(user_text, self._history)

        # 4. Run agentic loop
        answer, new_msgs = run_loop(self.client, self.model, system_msg, window, usage)

        # 5. Persist new messages (assistant + tool messages)
        for m in new_msgs:
            self._history.append(m)
        history.append_many(new_msgs, self.history_path)

        metrics = usage.summary()
        print(f"[agent.py] turn done — {metrics}")
        return {"answer": answer, "metrics": metrics}
