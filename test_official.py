"""Official-style local benchmark for CITS3011 Diplomacy agents.

This reproduces the supplied test.py's Scenario 1 and Scenario 2 opponent pools,
while adding optional deterministic seeds and machine-readable output.

It does NOT reproduce the staff-only Scenario 3 Hidden Agent or Scenario 4
student tournament.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from test_utils import (
    SCENARIO_POOLS,
    load_student_agent,
    print_benchmark,
    run_benchmark,
    save_benchmark_csv,
    save_benchmark_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "agent_module",
        nargs="?",
        default="agent_groupnumber",
        help="Python module containing StudentAgent, e.g. agent_09 (default: agent_groupnumber)",
    )
    parser.add_argument("--class-name", default="StudentAgent")
    parser.add_argument("--repeats", type=int, default=10, help="Games per controlled power")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible RNG seed")
    parser.add_argument(
        "--scenario",
        choices=("1", "2", "both"),
        default="both",
        help="Which supplied scenario(s) to run",
    )
    parser.add_argument("--end-year", type=int, default=1920)
    parser.add_argument(
        "--json-prefix",
        type=Path,
        default=None,
        help="Optional output prefix; writes <prefix>_scenario1.json etc.",
    )
    parser.add_argument(
        "--csv-prefix",
        type=Path,
        default=None,
        help="Optional output prefix; writes per-game CSV records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_cls = load_student_agent(args.agent_module, args.class_name)
    scenarios = ("1", "2") if args.scenario == "both" else (args.scenario,)

    for scenario in scenarios:
        result = run_benchmark(
            player_agent=agent_cls,
            opponent_agent_pool=SCENARIO_POOLS[scenario],
            repeats=args.repeats,
            seed=args.seed,
            label=f"Official-style Scenario {scenario} [{args.agent_module}]",
            end_year=args.end_year,
        )
        print_benchmark(result, rubric_scenario=scenario)

        if args.json_prefix is not None:
            path = Path(f"{args.json_prefix}_scenario{scenario}.json")
            save_benchmark_json(result, path)
            print(f"Saved JSON: {path}")
        if args.csv_prefix is not None:
            path = Path(f"{args.csv_prefix}_scenario{scenario}.csv")
            save_benchmark_csv(result, path)
            print(f"Saved CSV: {path}")


if __name__ == "__main__":
    main()
