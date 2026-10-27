"""
context.py — assembles the messages list sent to the model each turn.

Strategy (token-efficient):
  1. System prompt  (stable → prefix-cacheable)
  2. Relevant memory snippet (only entries matching the user's message)
  3. Recent history window (last HISTORY_WINDOW messages, not the full log)

This keeps the prompt small while still giving the model what it needs.
"""
import os

HISTORY_WINDOW: int = int(os.getenv("HISTORY_WINDOW", "20"))

SYSTEM_PROMPT = (
    "You are a concise personal budget assistant. "
    "Use the provided tools to record expenses, query summaries, set limits, "
    "convert currencies, and manage memory. "
    "Always reply briefly. Amounts are in ILS unless stated otherwise."
)


def build_system_message(memory_snippet: dict) -> dict:
    content = SYSTEM_PROMPT
    if memory_snippet:
        lines = "\n".join(f"- {k}: {v}" for k, v in memory_snippet.items())
        content += f"\n\nRelevant user preferences:\n{lines}"
    return {"role": "system", "content": content}


def build_messages(user_text: str, full_history: list[dict]) -> tuple[dict, list[dict]]:
    """
    Returns (system_msg, windowed_history).
    Caller combines them via caching.ordered_messages().
    """
    from API.harness.memory import get_relevant_for_context
    mem = get_relevant_for_context(user_text)
    print(f"[context.py] relevant memory keys: {list(mem.keys()) or 'none'}")

    window = full_history[-HISTORY_WINDOW:] if len(full_history) > HISTORY_WINDOW else full_history
    if len(full_history) > HISTORY_WINDOW:
        print(f"[context.py] history windowed to last {HISTORY_WINDOW} of {len(full_history)} messages")

    # Drop leading messages until the window starts with a "user" message.
    # This prevents sending orphan tool-reply messages or assistant messages
    # whose tool_calls have no preceding tool replies (rejected by strict APIs).
    original_len = len(window)
    while window and window[0].get("role") != "user":
        window = window[1:]
    if len(window) != original_len:
        print(f"[context.py] trimmed {original_len - len(window)} leading non-user message(s) "
              f"to avoid orphan tool exchange")

    system_msg = build_system_message(mem)
    return system_msg, window
