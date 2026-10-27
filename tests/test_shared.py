import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── redirect data files to tmp_path for every test ───────────────────────────

@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    import shared.tools as t
    import shared.memory as m
    monkeypatch.setattr(t, "DATA_DIR",       tmp_path)
    monkeypatch.setattr(t, "EXPENSES_FILE",  tmp_path / "expenses.json")
    monkeypatch.setattr(t, "BUDGETS_FILE",   tmp_path / "budgets.json")
    monkeypatch.setattr(m, "MEMORY_FILE",    tmp_path / "memory.json")


def _fx_response(to_cur: str, result: float) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"rates": {to_cur.upper(): result}}
    return resp


# ── convert_currency ──────────────────────────────────────────────────────────

def test_convert_same_currency():
    from shared.currency import convert_currency
    assert convert_currency(100, "ILS", "ILS") == 100.0


def test_convert_usd_to_ils():
    from shared.currency import convert_currency
    with patch("shared.currency.requests.get", return_value=_fx_response("ILS", 185.0)):
        assert convert_currency(50, "USD", "ILS") == 185.0


def test_convert_network_error():
    import requests as req
    from shared.currency import convert_currency
    with patch("shared.currency.requests.get", side_effect=req.RequestException("timeout")):
        with pytest.raises(ValueError, match="Currency conversion failed"):
            convert_currency(10, "EUR", "ILS")


def test_convert_bad_response():
    from shared.currency import convert_currency
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"rates": {}}          # missing key
    with patch("shared.currency.requests.get", return_value=resp):
        with pytest.raises(ValueError, match="Unexpected frankfurter response"):
            convert_currency(10, "USD", "ILS")


# ── add_expense ───────────────────────────────────────────────────────────────

def test_add_expense_ils():
    from shared.tools import add_expense, _load, EXPENSES_FILE
    msg = add_expense(100, "food", "lunch")
    assert "food" in msg and "100" in msg
    assert len(_load(EXPENSES_FILE)) == 1


def test_add_expense_converts_foreign_currency():
    from shared.tools import add_expense, _load, EXPENSES_FILE
    with patch("shared.currency.requests.get", return_value=_fx_response("ILS", 37.0)):
        add_expense(10, "transport", "taxi", currency="USD")
    row = _load(EXPENSES_FILE)[0]
    assert row["amount_ils"] == 37.0
    assert row["original"] == "10 USD"


def test_add_expense_negative_amount():
    from shared.tools import add_expense
    with pytest.raises(ValueError, match="positive"):
        add_expense(-5, "food")


def test_add_expense_empty_category():
    from shared.tools import add_expense
    with pytest.raises(ValueError, match="category"):
        add_expense(10, "  ")


# ── get_summary ───────────────────────────────────────────────────────────────

def test_get_summary_empty():
    from shared.tools import get_summary
    r = get_summary("2025-01")
    assert r == {"month": "2025-01", "total_ils": 0, "categories": {}}


def test_get_summary_totals():
    from shared.tools import add_expense, get_summary
    from datetime import datetime
    month = datetime.now().strftime("%Y-%m")
    add_expense(200, "food")
    add_expense(300, "food")
    add_expense(150, "rent")
    s = get_summary(month)
    assert s["categories"]["food"]["spent"] == 500.0
    assert s["categories"]["rent"]["spent"] == 150.0
    assert s["total_ils"] == 650.0


def test_get_summary_category_filter():
    from shared.tools import add_expense, get_summary
    from datetime import datetime
    month = datetime.now().strftime("%Y-%m")
    add_expense(200, "food")
    add_expense(150, "rent")
    s = get_summary(month, category="food")
    assert "rent" not in s["categories"]
    assert s["categories"]["food"]["spent"] == 200.0


def test_get_summary_shows_limit():
    from shared.tools import add_expense, get_summary, set_budget_limit
    from datetime import datetime
    month = datetime.now().strftime("%Y-%m")
    set_budget_limit("food", 1500)
    add_expense(200, "food")
    s = get_summary(month, "food")
    assert s["categories"]["food"]["limit"] == 1500.0


# ── set_budget_limit ──────────────────────────────────────────────────────────

def test_set_budget_limit_persists():
    from shared.tools import set_budget_limit, _load, BUDGETS_FILE
    set_budget_limit("food", 1500)
    assert _load(BUDGETS_FILE)["food"] == 1500.0


def test_set_budget_limit_zero():
    from shared.tools import set_budget_limit
    with pytest.raises(ValueError, match="positive"):
        set_budget_limit("food", 0)


# ── search_expenses ───────────────────────────────────────────────────────────

def test_search_by_note():
    from shared.tools import add_expense, search_expenses
    add_expense(50, "food", "pizza night")
    add_expense(30, "food", "coffee")
    results = search_expenses("pizza")
    assert len(results) == 1
    assert results[0]["note"] == "pizza night"


def test_search_by_category():
    from shared.tools import add_expense, search_expenses
    add_expense(100, "transport", "bus")
    add_expense(200, "food", "dinner")
    results = search_expenses("transport")
    assert len(results) == 1 and results[0]["category"] == "transport"


def test_search_limit():
    from shared.tools import add_expense, search_expenses
    for i in range(10):
        add_expense(10, "food", f"item {i}")
    assert len(search_expenses("food", limit=3)) == 3


def test_search_empty_query():
    from shared.tools import search_expenses
    with pytest.raises(ValueError, match="query"):
        search_expenses("")


# ── memory ────────────────────────────────────────────────────────────────────

def test_write_and_read_memory():
    from shared.memory import write_memory, read_memory
    write_memory("food_limit", "1500 ILS")
    assert read_memory()["food_limit"] == "1500 ILS"


def test_relevant_memory_hit():
    from shared.memory import write_memory, relevant_memory
    write_memory("food_limit", "1500 ILS")
    write_memory("rent_limit", "3000 ILS")
    write_memory("prefers_short_answers", "yes")
    hits = relevant_memory("food")
    assert "food_limit" in hits and "rent_limit" not in hits


def test_relevant_memory_value_match():
    from shared.memory import write_memory, relevant_memory
    write_memory("pref1", "prefers short answers")
    hits = relevant_memory("short")
    assert "pref1" in hits


def test_relevant_memory_empty_query():
    from shared.memory import relevant_memory
    with pytest.raises(ValueError, match="query"):
        relevant_memory("")


def test_write_memory_empty_key():
    from shared.memory import write_memory
    with pytest.raises(ValueError, match="key"):
        write_memory("  ", "value")


# ── schemas ───────────────────────────────────────────────────────────────────

def test_schema_tool_names():
    from shared.schemas import TOOLS
    names = {t["name"] for t in TOOLS}
    assert names == {"add_expense", "get_summary", "set_budget_limit",
                     "convert_currency", "search_expenses"}


def test_schema_required_fields():
    from shared.schemas import TOOLS
    for tool in TOOLS:
        assert "name" in tool and "description" in tool and "parameters" in tool
        assert "required" in tool["parameters"]


def test_get_tool_map_callable():
    from shared.schemas import get_tool_map
    for name, fn in get_tool_map().items():
        assert callable(fn), f"{name} is not callable"
