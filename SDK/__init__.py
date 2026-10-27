"""
SDK/main.py — CLI entry point for the SDK version.

Usage:
  python SDK/main.py                                       # interactive chat
  python SDK/main.py --scenario scenario.json --out results/sdk.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from SDK.agent import BudgetAgent  # noqa: E402


def run_scenario(agent: BudgetAgent, scenario_path: str, out_path: str) -> None:
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    turns    = scenario.get("turns", [])
    results  = []
    for i, turn in enumerate(turns, 1):
        user_text = turn["user"]
        print(f"\n{'='*60}\nTurn {i}: {user_text}\n{'='*60}")
        result = agent.run_turn(user_text)
        print(f"Answer: {result['answer']}")
        results.append({"turn": i, "user": user_text, **result})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SDK/main.py] results written to {out_path}")


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
    parser = argparse.ArgumentParser(description="Budget Agent — SDK version")
    parser.add_argument("--scenario", help="Path to scenario.json")
    parser.add_argument("--out",      default="results/sdk.json", help="Output path for results JSON")
    parser.add_argument("--session",  default="default", help="Session ID for SQLite history")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    agent = BudgetAgent(session_id=args.session)
    if args.scenario:
        run_scenario(agent, args.scenario, args.out)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
