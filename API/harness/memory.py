"""
memory.py — harness-side memory layer.
Wraps shared.memory and exposes save_memory / recall_memory as agent-callable
tool functions, plus their JSON schemas for the tools list.
"""
from shared.memory import write_memory, relevant_memory, read_memory

MEMORY_TOOL_SCHEMAS = [
    {
        "name": "save_memory",
        "description": "Persist a user preference or decision for future turns (e.g. 'food limit 1500 ILS').",
        "parameters": {
            "type": "object",
            "properties": {
                "key":   {"type": "string", "description": "Short identifier"},
                "value": {"type": "string", "description": "Value to remember"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Search stored preferences/decisions by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword to search"},
            },
            "required": ["query"],
        },
    },
]


def save_memory(key: str, value: str) -> str:
    import shared.memory as _mem
    _mem.write_memory(key, value)
    print(f"[memory.py] saved key={key}")
    return f"Remembered: {key} = {value}"


def recall_memory(query: str) -> str:
    import shared.memory as _mem
    hits = _mem.relevant_memory(query)
    print(f"[memory.py] recall query='{query}' → {len(hits)} hits")
    if not hits:
        return "No matching memory found."
    return "; ".join(f"{k}: {v}" for k, v in hits.items())


def get_relevant_for_context(query: str) -> dict:
    """Used by context.py to inject only relevant memory into the prompt."""
    import shared.memory as _mem
    try:
        return _mem.relevant_memory(query)
    except ValueError:
        return {}
