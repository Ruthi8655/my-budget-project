"""
compaction.py — replaces old conversation history with a model-generated summary
when the estimated token count exceeds COMPACTION_THRESHOLD.

Token estimation: ~4 chars per token (rough but avoids a tokeniser dependency).
"""
import os

COMPACTION_THRESHOLD: int = int(os.getenv("COMPACTION_THRESHOLD", "6000"))
_CHARS_PER_TOKEN = 4


def _estimate_tokens(messages: list[dict]) -> int:
    total = sum(len(str(m.get("content") or "")) for m in messages)
    return total // _CHARS_PER_TOKEN


def needs_compaction(messages: list[dict]) -> bool:
    est = _estimate_tokens(messages)
    if est >= COMPACTION_THRESHOLD:
        print(f"[compaction.py] estimated {est} tokens ≥ threshold {COMPACTION_THRESHOLD} → compaction needed")
        return True
    return False


def compact(messages: list[dict], client, model: str) -> list[dict]:
    """
    Summarise `messages` with one model call and return a replacement list
    containing a single assistant message with the summary.
    """
    print(f"[compaction.py] compacting {len(messages)} messages via model call")
    summary_prompt = [
        {"role": "system", "content": "Summarise the following conversation compactly, preserving key facts, numbers, and decisions. Be brief."},
        {"role": "user",   "content": "\n".join(
            f"{m['role'].upper()}: {m.get('content') or ''}" for m in messages
        )},
    ]
    response = client.chat.completions.create(model=model, messages=summary_prompt, max_tokens=512)
    summary_text = response.choices[0].message.content.strip()
    print(f"[compaction.py] summary length={len(summary_text)} chars")
    return [{"role": "assistant", "content": f"[Conversation summary]\n{summary_text}"}]
