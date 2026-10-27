"""
caching.py — helpers for OpenAI automatic prefix caching.

OpenAI uses *automatic* prefix caching — there is no cache_control or
cache-write field like Anthropic has.  The cache is hit when the leading
tokens of a request exactly match a previous request.

Strategy: always place stable content FIRST in the messages list so the
longest common prefix is reused:
  1. system prompt  (never changes within a session)
  2. tool definitions are sent via the `tools=` parameter (also stable)
  3. conversation history follows

This module exposes:
  - ordered_messages(system_msg, history)  → stable-first message list
  - cached_tokens_from(response)           → int, 0 if not reported
"""


def ordered_messages(system_msg: dict, history: list[dict]) -> list[dict]:
    """Return [system_msg] + history so the stable prefix comes first."""
    print("[caching.py] building ordered messages (system first for prefix cache)")
    return [system_msg] + history


def cached_tokens_from(response) -> int:
    """Extract cached_tokens from response usage, or 0."""
    try:
        details = response.usage.prompt_tokens_details
        cached = getattr(details, "cached_tokens", 0) or 0
        if cached:
            print(f"[caching.py] prefix cache hit: {cached} cached tokens")
        return cached
    except AttributeError:
        return 0
