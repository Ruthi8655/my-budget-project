"""
scripts/compare.py — compare two run results JSONs side-by-side.

Usage:
  python scripts/compare.py results/api.json results/sdk.json
"""
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct_diff(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def compare(path_a: str, path_b: str) -> None:
    a, b = _load(path_a), _load(path_b)
    label_a = Path(path_a).stem
    label_b = Path(path_b).stem

    print(f"\n{'='*60}")
    print(f"  Comparing  {label_a}  vs  {label_b}")
    print(f"{'='*60}")

    # ── totals ────────────────────────────────────────────────────────────────
    ta, tb = a.get("totals", {}), b.get("totals", {})
    metrics = [
        ("input_tokens",  "Input tokens"),
        ("cached_tokens", "Cached tokens"),
        ("output_tokens", "Output tokens"),
        ("model_calls",   "Model calls"),
        ("cost_usd",      "Cost (USD)"),
        ("wall_time_s",   "Wall time (s)"),
    ]
    print(f"\n{'Metric':<22} {label_a:>12} {label_b:>12} {'diff':>10}")
    print("-" * 60)
    for key, label in metrics:
        va = ta.get(key, 0)
        vb = tb.get(key, 0)
        diff = _pct_diff(va, vb) if isinstance(va, (int, float)) else "n/a"
        print(f"{label:<22} {va:>12} {vb:>12} {diff:>10}")

    # ── harness lines ─────────────────────────────────────────────────────────
    la, lb = a.get("harness_lines", 0), b.get("harness_lines", 0)
    print(f"\n{'Harness lines':<22} {la:>12} {lb:>12} {_pct_diff(la, lb):>10}")

    # ── checks ────────────────────────────────────────────────────────────────
    def _check_summary(r: dict) -> str:
        return f"{r.get('checks_passed', '?')}/{r.get('checks_total', '?')}"

    print(f"\n{'Checks passed':<22} {_check_summary(a):>12} {_check_summary(b):>12}")

    # ── per-turn comparison ───────────────────────────────────────────────────
    turns_a = {(t["session_id"], t["turn"]): t for t in a.get("turns", [])}
    turns_b = {(t["session_id"], t["turn"]): t for t in b.get("turns", [])}
    all_keys = sorted(set(turns_a) | set(turns_b))

    if all_keys:
        print(f"\n{'Turn':<14} {'in_A':>7} {'in_B':>7} {'out_A':>7} {'out_B':>7} {'cost_A':>9} {'cost_B':>9}")
        print("-" * 60)
        for key in all_keys:
            ta_ = turns_a.get(key, {})
            tb_ = turns_b.get(key, {})
            sid, tnum = key
            label = f"{sid}/t{tnum}"
            print(
                f"{label:<14}"
                f" {ta_.get('input_tokens', 0):>7}"
                f" {tb_.get('input_tokens', 0):>7}"
                f" {ta_.get('output_tokens', 0):>7}"
                f" {tb_.get('output_tokens', 0):>7}"
                f" {ta_.get('cost_usd', 0.0):>9.5f}"
                f" {tb_.get('cost_usd', 0.0):>9.5f}"
            )

    print()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare.py <results_a.json> <results_b.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
