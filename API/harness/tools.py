"""
tools.py — tool definitions (from shared/schemas.py) and dispatcher.
Exceptions are caught and returned as "ERROR: …" strings so the model can recover.
"""
import json

from shared.schemas import TOOLS as DOMAIN_TOOLS, get_tool_map
from API.harness.memory import MEMORY_TOOL_SCHEMAS, save_memory, recall_memory

# ── build the full tools list (domain + memory) ───────────────────────────────

def get_openai_tools() -> list[dict]:
    """Return tool definitions in OpenAI function-calling format."""
    all_schemas = DOMAIN_TOOLS + MEMORY_TOOL_SCHEMAS
    return [{"type": "function", "function": s} for s in all_schemas]


# ── dispatcher ────────────────────────────────────────────────────────────────

def _get_dispatch_map() -> dict:
    m = get_tool_map()
    m["save_memory"]   = save_memory
    m["recall_memory"] = recall_memory
    return m


def dispatch(tool_name: str, arguments_json: str) -> str:
    """
    Call the named tool with parsed arguments.
    Returns a compact string result, or "ERROR: …" on failure.
    """
    print(f"[tools.py] dispatch {tool_name}")
    dispatch_map = _get_dispatch_map()
    if tool_name not in dispatch_map:
        return f"ERROR: unknown tool '{tool_name}'"
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        result = dispatch_map[tool_name](**args)
        # Compact serialisation: dicts/lists → JSON, primitives → str
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as e:
        print(f"[tools.py] ERROR in {tool_name}: {e}")
        return f"ERROR: {e}"
