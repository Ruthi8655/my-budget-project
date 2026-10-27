import json
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
EXPENSES_FILE = DATA_DIR / "expenses.json"
BUDGETS_FILE  = DATA_DIR / "budgets.json"


def _load(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [] if path == EXPENSES_FILE else {}


def _save(path: Path, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── public API ────────────────────────────────────────────────────────────────

def add_expense(amount: float, category: str, note: str = "", currency: str = "ILS") -> str:
    if amount <= 0:
        raise ValueError("amount must be positive")
    category = category.strip().lower()
    if not category:
        raise ValueError("category is required")

    amount_ils = float(amount)
    if currency.upper() != "ILS":
        from shared.currency import convert_currency
        amount_ils = convert_currency(amount, currency, "ILS")

    row = {
        "id":         str(uuid.uuid4())[:8],
        "date":       datetime.now().strftime("%Y-%m"),
        "amount_ils": round(amount_ils, 2),
        "category":   category,
        "note":       note,
        "original":   f"{amount} {currency.upper()}",
    }
    rows = _load(EXPENSES_FILE)
    rows.append(row)
    _save(EXPENSES_FILE, rows)
    return f"Added {row['original']} ({row['amount_ils']} ILS) to '{category}'."


def get_summary(month: str, category: str | None = None) -> dict:
    rows = [r for r in _load(EXPENSES_FILE) if r["date"] == month]
    if category:
        rows = [r for r in rows if r["category"] == category.strip().lower()]
    if not rows:
        return {"month": month, "total_ils": 0, "categories": {}}

    totals: dict[str, float] = {}
    for r in rows:
        totals[r["category"]] = round(totals.get(r["category"], 0.0) + r["amount_ils"], 2)

    budgets = _load(BUDGETS_FILE)
    categories_out = {
        cat: {"spent": spent, "limit": budgets.get(cat)}
        for cat, spent in sorted(totals.items(), key=lambda x: -x[1])
    }
    return {"month": month, "total_ils": round(sum(totals.values()), 2), "categories": categories_out}


def set_budget_limit(category: str, limit_ils: float) -> str:
    if limit_ils <= 0:
        raise ValueError("limit must be positive")
    category = category.strip().lower()
    budgets = _load(BUDGETS_FILE)
    budgets[category] = round(float(limit_ils), 2)
    _save(BUDGETS_FILE, budgets)
    return f"Budget limit for '{category}' set to {limit_ils} ILS."


def search_expenses(query: str, limit: int = 5) -> list[dict]:
    q = query.strip().lower()
    if not q:
        raise ValueError("query is required")
    matches = [
        {"id": r["id"], "date": r["date"], "amount_ils": r["amount_ils"],
         "category": r["category"], "note": r["note"]}
        for r in _load(EXPENSES_FILE)
        if q in r["category"] or q in r["note"].lower()
    ]
    return matches[:limit]
