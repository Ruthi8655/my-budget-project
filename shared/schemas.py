TOOLS = [
    {
        "name": "add_expense",
        "description": "Record a new expense. Converts non-ILS currencies automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":   {"type": "number", "description": "Expense amount"},
                "category": {"type": "string", "description": "Category, e.g. food, rent"},
                "note":     {"type": "string", "description": "Optional short note"},
                "currency": {"type": "string", "description": "ISO currency code, default ILS"},
            },
            "required": ["amount", "category"],
        },
    },
    {
        "name": "get_summary",
        "description": "Get spending totals per category for a month (YYYY-MM format).",
        "parameters": {
            "type": "object",
            "properties": {
                "month":    {"type": "string", "description": "Month as YYYY-MM"},
                "category": {"type": "string", "description": "Optional: filter to one category"},
            },
            "required": ["month"],
        },
    },
    {
        "name": "set_budget_limit",
        "description": "Set a monthly budget limit in ILS for a category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category":  {"type": "string", "description": "Category name"},
                "limit_ils": {"type": "number", "description": "Limit amount in ILS"},
            },
            "required": ["category", "limit_ils"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies using live rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":   {"type": "number", "description": "Amount to convert"},
                "from_cur": {"type": "string", "description": "Source currency ISO code"},
                "to_cur":   {"type": "string", "description": "Target currency ISO code"},
            },
            "required": ["amount", "from_cur", "to_cur"],
        },
    },
    {
        "name": "search_expenses",
        "description": "Search past expenses by keyword in category or note.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",  "description": "Search keyword"},
                "limit": {"type": "integer", "description": "Max results, default 5"},
            },
            "required": ["query"],
        },
    },
]


def get_tool_map() -> dict:
    from shared.tools import add_expense, get_summary, set_budget_limit, search_expenses
    from shared.currency import convert_currency
    return {
        "add_expense":      add_expense,
        "get_summary":      get_summary,
        "set_budget_limit": set_budget_limit,
        "convert_currency": convert_currency,
        "search_expenses":  search_expenses,
    }
