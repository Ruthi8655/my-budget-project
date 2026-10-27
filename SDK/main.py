"""
SDK/main.py — CLI entry point for the SDK version.

Usage:
  python SDK/main.py                                       # interactive chat
  python SDK/main.py --scenario scenario.json --out results/sdk.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from SDK.agent import BudgetAgent       # noqa: E402
from shared.runner import run_scenario  # noqa: E402

_HARNESS_DIR = Path(__file__).parent    # SDK/ itself is the "harness"


def run_interactive(agent: BudgetAgent) -> None:
    print("[SDK/main.py] interactive mode — type 'quit' to exit")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in {"quit", "exit", "q"}:
            break
        if not user_text:
            continue
        result = agent.run_turn(user_text)
        print(f"Agent: {result['answer']}")
        m = result["metrics"]
        print(f"  [tokens in={m['input_tokens']} cached={m['cached_tokens']} "
              f"out={m['output_tokens']} cost=${m['cost_usd']:.5f} time={m['wall_time_s']}s]")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Budget Agent — SDK version")
    parser.add_argument("--scenario", help="Path to scenario.json")
    parser.add_argument("--out",      default="results/sdk.json", help="Output path for results JSON")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.scenario:
        run_scenario(args.scenario, BudgetAgent, _HARNESS_DIR, args.out)
    else:
        run_interactive(BudgetAgent())


if __name__ == "__main__":
    main()
