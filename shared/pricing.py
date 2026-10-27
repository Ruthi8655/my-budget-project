"""
pricing.py — shared token price table (per 1M tokens, USD).
Edit here to update costs for both API/ and SDK/ versions.
"""
PRICE_TABLE: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15,  "cached": 0.075, "output": 0.60},
    "gpt-4o":      {"input": 5.00,  "cached": 2.50,  "output": 15.00},
}
DEFAULT_PRICE: dict[str, float] = {"input": 0.15, "cached": 0.075, "output": 0.60}


def compute_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    p = PRICE_TABLE.get(model, DEFAULT_PRICE)
    non_cached = max(input_tokens - cached_tokens, 0)
    return (
        non_cached     * p["input"]  / 1_000_000
        + cached_tokens * p["cached"] / 1_000_000
        + output_tokens * p["output"] / 1_000_000
    )
