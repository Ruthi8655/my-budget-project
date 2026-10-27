"""
main.py — CLI entry point for the API/ harness.

Usage:
  python API/main.py                                      # interactive chat
  python API/main.py --scenario scenario.json --out results/api.json
"""
import argparse
import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from API.agent import Agent          # noqa: E402
from shared.runner import run_scenario  # noqa: E402

_HARNESS_DIR = Path(__file__).parent / "harness"


def run_interactive(agent: Agent) -> None:
    print("[main.py] interactive mode — type 'quit' to exit")
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
    parser = argparse.ArgumentParser(description="Budget Agent — API harness")
    parser.add_argument("--scenario", help="Path to scenario.json")
    parser.add_argument("--out",      help="Output path for results JSON", default="results/api.json")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.scenario:
        run_scenario(args.scenario, Agent, _HARNESS_DIR, args.out)
    else:
        run_interactive(Agent())


if __name__ == "__main__":
    main()
