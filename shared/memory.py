import json
from pathlib import Path

MEMORY_FILE = Path(__file__).parent / "data" / "memory.json"


def _load() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {}


def _save(data: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_memory() -> dict:
    return _load()


def write_memory(key: str, value: str) -> None:
    if not key.strip():
        raise ValueError("key must not be empty")
    data = _load()
    data[key.strip()] = value
    _save(data)


def relevant_memory(query: str) -> dict:
    if not query.strip():
        raise ValueError("query must not be empty")
    q = query.strip().lower()
    return {k: v for k, v in _load().items() if q in k.lower() or q in str(v).lower()}
