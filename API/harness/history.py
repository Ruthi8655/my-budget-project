"""
history.py — append-only JSONL conversation history, persisted to disk.
Each line is one message dict (role/content/tool_calls/tool_call_id …).
"""
import json
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "shared" / "data" / "history.jsonl"


def load(path: Path | None = None) -> list[dict]:
    p = path or _DEFAULT_PATH
    if not p.exists():
        print(f"[history.py] no history file at {p}, starting fresh")
        return []
    messages = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[history.py] loaded {len(messages)} messages from {p}")
    return messages


def append(message: dict, path: Path | None = None) -> None:
    p = path or _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")
    print(f"[history.py] appended role={message.get('role')} to {p.name}")


def append_many(messages: list[dict], path: Path | None = None) -> None:
    for m in messages:
        append(m, path)


def overwrite(messages: list[dict], path: Path | None = None) -> None:
    """Replace entire history (used after compaction)."""
    p = path or _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + ("\n" if messages else ""),
        encoding="utf-8",
    )
    print(f"[history.py] overwrote history with {len(messages)} messages (post-compaction)")
