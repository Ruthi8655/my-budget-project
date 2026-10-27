"""
SDK/tools.py — wraps shared domain functions with @function_tool.

Each tool uses the same name and description as shared/schemas.py so both
versions present identical tool definitions to the model.

# replaces: API/harness/tools.py  (schema building + dispatch are handled by the SDK)
"""
import json
from agents import function_tool
from shared.schemas import TOOLS as _SCHEMAS
from shared.tools import add_expense, get_summary, set_budget_limit, search_expenses
from shared.currency import convert_currency
import shared.memory as _mem

# Pull descriptions from the single source of truth in shared/schemas.py
_desc = {t["name"]: t["description"] for t in _SCHEMAS}


@function_tool(name_override="add_expense", description_override=_desc["add_expense"])
def sdk_add_expense(amount: float, category: str, note: str = "", currency: str = "ILS") -> str:
    print(f"[SDK/tools.py] add_expense amount={amount} category={category} currency={currency}")
    try:
        return add_expense(amount, category, note, currency)
    except Exception as e:
        return f"ERROR: {e}"


@function_tool(name_override="get_summary", description_override=_desc["get_summary"])
def sdk_get_summary(month: str, category: str = "") -> str:
    print(f"[SDK/tools.py] get_summary month={month} category={category!r}")
    try:
        result = get_summary(month, category or None)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {e}"


@function_tool(name_override="set_budget_limit", description_override=_desc["set_budget_limit"])
def sdk_set_budget_limit(category: str, limit_ils: float) -> str:
    print(f"[SDK/tools.py] set_budget_limit category={category} limit={limit_ils}")
    try:
        return set_budget_limit(category, limit_ils)
    except Exception as e:
        return f"ERROR: {e}"


@function_tool(name_override="convert_currency", description_override=_desc["convert_currency"])
def sdk_convert_currency(amount: float, from_cur: str, to_cur: str) -> str:
    print(f"[SDK/tools.py] convert_currency {amount} {from_cur}->{to_cur}")
    try:
        result = convert_currency(amount, from_cur, to_cur)
        return f"{result} {to_cur.upper()}"
    except Exception as e:
        return f"ERROR: {e}"


@function_tool(name_override="search_expenses", description_override=_desc["search_expenses"])
def sdk_search_expenses(query: str, limit: int = 5) -> str:
    print(f"[SDK/tools.py] search_expenses query={query!r} limit={limit}")
    try:
        results = search_expenses(query, limit)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {e}"


@function_tool(
    name_override="save_memory",
    description_override="Persist a user preference or decision for future turns.",
)
def sdk_save_memory(key: str, value: str) -> str:
    print(f"[SDK/tools.py] save_memory key={key}")
    try:
        _mem.write_memory(key, value)
        return f"Remembered: {key} = {value}"
    except Exception as e:
        return f"ERROR: {e}"


ALL_TOOLS = [
    sdk_add_expense,
    sdk_get_summary,
    sdk_set_budget_limit,
    sdk_convert_currency,
    sdk_search_expenses,
    sdk_save_memory,
]
