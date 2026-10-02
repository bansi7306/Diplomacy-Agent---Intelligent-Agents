#!/usr/bin/env python3
"""
CITS3011 Group 09 - Final Report Experiment Tester
===================================================

Purpose
-------
This file contains the experiments that support the quantitative technique
comparisons discussed in the group report. It is NOT intended to reproduce the
teaching team's private marking tests.

The report-level design agreed by the group is:

    BASIC METHOD
        Greedy shortest-path / supply-centre distance heuristic.

    NEW TECHNIQUE 1 - TACTICAL MULTI-UNIT PLANNING
        Short-horizon planning/refinement applied on top of the greedy output.
        In the current agent this report-level technique is represented by the
        short-horizon TECHNIQUES switches listed in TACTICAL_PLANNING_SWITCHES.

    NEW TECHNIQUE 2 - PERSISTENT LONG-TERM STRATEGY
        Keep a strategic target across turns rather than choosing only from the
        immediate greedy score each turn.

    NEW TECHNIQUE 3 - OPPONENT MODELLING
        Use observed opponent aggression, positional pressure, and strength as
        a threat signal. The submitted source currently has
        opponent_multiplier == 0.0, so this tester uses an explicit non-zero
        experimental ON value to actually evaluate the technique.

What the default run does
-------------------------
It runs BOTH locally available project scenarios for every experiment:

    Scenario 1: all opponents are StaticAgent.
    Scenario 2: opponents are sampled from the same weighted pool used by the
                supplied local tester: Random, Attitude, Attitude, Greedy,
                Greedy.

For every configuration, seeds 0,1,2,3,4 are run in fresh Python subprocesses.
Each seed tests the student agent as all seven Diplomacy powers, so by default:

    5 seeds x 7 powers x 2 scenarios = 70 games per configuration.

The default experiment suite has two parts:

A) Sequential development experiment
    01_basic_greedy
    02_plus_tactical_planning
    03_plus_long_term_strategy
    04_plus_opponent_modelling

   This directly shows the report story: start from the basic method, then add
   the three new techniques one by one.

B) One-at-a-time ablation experiment
    05_without_tactical_planning
    06_without_long_term_strategy
    07_without_opponent_modelling

   These all start from the same "all three report techniques ON" condition and
   remove one report technique at a time.

A final reference is also included:
    08_final_agent_reference

   This is the exact current agent source with no test-time changes. It is kept
   separate because the current source has opponent_multiplier == 0.0, whereas
   the report experiment must give opponent modelling a non-zero influence to
   genuinely test it.

Results
-------
The tester automatically creates and updates:

    results/report_technique_evaluation.json
    results/report_technique_summary.csv

Both files are checkpointed after every completed seed, so an interrupted run
can be continued with --resume.

Typical commands
----------------
Full final experiment:
    python test_09.py

Explicit agent module:
    python test_09.py agent_09

Resume an interrupted run:
    python test_09.py agent_09 --resume

Run only the sequential report experiment:
    python test_09.py agent_09 --suite sequential

Run only the one-at-a-time ablations:
    python test_09.py agent_09 --suite ablation

Stronger final validation (3 repeats per power per seed):
    python test_09.py agent_09 --repeats 3

IMPORTANT
---------
If the team changes the exact subcomponents that belong to Technique 1, edit
TACTICAL_PLANNING_SWITCHES below. The rest of the experiment engine does not
need to change.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ===========================================================================
# 1. REPORT EXPERIMENT CONFIGURATION
# ===========================================================================

ALL_POWERS: Tuple[str, ...] = (
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
)

STAGES: Tuple[str, ...] = ("EARLY", "MID", "LATE")

# These are the THREE report-level techniques agreed by the group.
# True means "include this technique in the final report experiment suite".
# The experiment builder below reads this dictionary directly, so changing a
# value here changes which report techniques are added/ablated.
REPORT_TECHNIQUES: Dict[str, bool] = {
    "tactical_planning": True,
    "long_term_strategy": True,
    "opponent_modelling": True,
}

# The order is also the agreed development story in the report.
REPORT_TECHNIQUE_ORDER: Tuple[str, ...] = (
    "tactical_planning",
    "long_term_strategy",
    "opponent_modelling",
)

REPORT_TECHNIQUE_LABELS: Dict[str, str] = {
    "tactical_planning": "Tactical multi-unit planning",
    "long_term_strategy": "Persistent long-term strategy",
    "opponent_modelling": "Opponent modelling",
}

# Technique 1 specifically groups the parts that take independently scored
# greedy unit choices and coordinate them into compatible multi-unit orders.
# This is the same coordination bundle the team previously ablated together.
TACTICAL_PLANNING_SWITCHES: Tuple[str, ...] = (
    "supported_attacks",
    "collision_resolution",
    "self_block_removal",
    "support_reconciliation",
)

LONG_TERM_SWITCH = "strategic_target"
OPPONENT_MODEL_SWITCH = "opponent_modelling"

# The current source agent has opponent_multiplier == 0.0. A zero multiplier
# makes the threat model strategically inert, even if opponent_modelling=True.
# Therefore the report experiment needs a non-zero ON value. Keep this explicit
# so the report/result JSON can state exactly what was tested.
OPPONENT_MODEL_ON_MULTIPLIER = 1.0

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_REPEATS = 1
DEFAULT_END_YEAR = 1920

DEFAULT_JSON = Path("results/report_technique_evaluation.json")
DEFAULT_CSV = Path("results/report_technique_summary.csv")

RESULT_SENTINEL = "__GROUP09_REPORT_TEST_RESULT__="


# ===========================================================================
# 2. SMALL GENERAL HELPERS
# ===========================================================================


def parse_seeds(text: str) -> List[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def clean_module_name(value: str) -> str:
    value = value.strip()
    if value.endswith(".py"):
        value = value[:-3]
    return value.replace("/", ".").replace("\\", ".").strip(".")


def load_agent_class(agent_module: str):
    module_name = clean_module_name(agent_module)
    module = importlib.import_module(module_name)
    if not hasattr(module, "StudentAgent"):
        raise AttributeError(f"{module_name!r} does not expose StudentAgent")
    return module.StudentAgent


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def classify_outcome(sc: int) -> str:
    if sc >= 18:
        return "WIN"
    if sc <= 0:
        return "DEFEAT"
    return "SURVIVE"


# ===========================================================================
# 3. VALIDATE THE LATEST AGENT AGAINST THE REPORT TEST DEFINITION
# ===========================================================================


def validate_agent_structure(agent_cls) -> Dict[str, Any]:
    if not hasattr(agent_cls, "TECHNIQUES"):
        raise AttributeError("StudentAgent.TECHNIQUES is required by this tester")
    if not hasattr(agent_cls, "STAGE_WEIGHTS"):
        raise AttributeError("StudentAgent.STAGE_WEIGHTS is required by this tester")

    techniques = copy.deepcopy(agent_cls.TECHNIQUES)
    weights = copy.deepcopy(agent_cls.STAGE_WEIGHTS)

    required_switches = set(TACTICAL_PLANNING_SWITCHES)
    required_switches.add(LONG_TERM_SWITCH)
    required_switches.add(OPPONENT_MODEL_SWITCH)

    missing = sorted(required_switches - set(techniques))
    if missing:
        raise ValueError(
            "The latest agent is missing report-test technique switch(es): "
            + ", ".join(missing)
        )

    for stage in STAGES:
        if stage not in weights or not isinstance(weights[stage], dict):
            raise ValueError(f"STAGE_WEIGHTS is missing dictionary stage {stage}")
        for key in (
            "distance_cap",
            "support_bonus",
            "contest_penalty",
            "hold_baseline",
            "lookahead_penalty",
            "aggression_weight",
            "pressure_weight",
            "strength_weight",
            "opponent_multiplier",
        ):
            if key not in weights[stage]:
                raise ValueError(f"STAGE_WEIGHTS[{stage!r}] is missing {key!r}")

    return {
        "techniques": techniques,
        "stage_weights": weights,
    }


# ===========================================================================
# 4. BUILD THE EXACT REPORT EXPERIMENT CONFIGURATIONS
# ===========================================================================


def all_switches_off(source_switches: Mapping[str, bool]) -> Dict[str, bool]:
    return {name: False for name in source_switches}


def basic_weight_overrides(source_weights: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Preserve the basic distance heuristic and hold baseline, but neutralise the
    extra tactical/opponent scoring terms. Retreat/build/disband code is left
    intact because those phases still have to produce legal game actions.
    """
    out = copy.deepcopy(source_weights)
    for stage in STAGES:
        out[stage]["support_bonus"] = 0.0
        out[stage]["contest_penalty"] = 0.0
        out[stage]["lookahead_penalty"] = 0.0
        out[stage]["opponent_multiplier"] = 0.0
    return out


def tactical_weight_overrides(source_weights: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Use the source agent's normal tactical weights, but keep opponent influence OFF."""
    out = copy.deepcopy(source_weights)
    for stage in STAGES:
        out[stage]["opponent_multiplier"] = 0.0
    return out


def opponent_on_weight_overrides(source_weights: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Use normal source weights and give opponent modelling a real non-zero influence."""
    out = copy.deepcopy(source_weights)
    for stage in STAGES:
        out[stage]["opponent_multiplier"] = float(OPPONENT_MODEL_ON_MULTIPLIER)
    return out


def make_report_states(
    source_switches: Mapping[str, bool],
    *,
    tactical: bool,
    long_term: bool,
    opponent: bool,
) -> Dict[str, bool]:
    """
    Build a clean report-experiment state from scratch.

    Non-report technique switches stay OFF in the sequential/ablation experiment
    unless they are explicitly part of Technique 1's tactical bundle. This
    prevents unrelated switches from silently confounding the report comparison.
    """
    states = all_switches_off(source_switches)

    if tactical:
        for name in TACTICAL_PLANNING_SWITCHES:
            states[name] = True

    states[LONG_TERM_SWITCH] = bool(long_term)
    states[OPPONENT_MODEL_SWITCH] = bool(opponent)
    return states


def report_state_and_weights(
    source_switches: Mapping[str, bool],
    source_weights: Mapping[str, Mapping[str, Any]],
    active_report_techniques: Iterable[str],
) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]]]:
    """Translate report-level technique names into exact agent switch/weight state."""
    active = set(active_report_techniques)

    states = make_report_states(
        source_switches,
        tactical="tactical_planning" in active,
        long_term="long_term_strategy" in active,
        opponent="opponent_modelling" in active,
    )

    # Without Technique 1 we keep the basic greedy scoring context. With
    # Technique 1, restore the source support/contest tactical scoring weights.
    if "tactical_planning" in active:
        weights = tactical_weight_overrides(source_weights)
    else:
        weights = basic_weight_overrides(source_weights)

    # Technique 3 must have a real non-zero influence when experimentally ON.
    if "opponent_modelling" in active:
        for stage in STAGES:
            weights[stage]["opponent_multiplier"] = float(
                OPPONENT_MODEL_ON_MULTIPLIER
            )
    else:
        for stage in STAGES:
            weights[stage]["opponent_multiplier"] = 0.0

    return states, weights


def build_configurations(
    source_switches: Mapping[str, bool],
    source_weights: Mapping[str, Mapping[str, Any]],
    suite: str,
) -> List[Dict[str, Any]]:
    """
    Build the report experiment from REPORT_TECHNIQUES in one place.

    This is intentionally data-driven: the enabled techniques are added in
    REPORT_TECHNIQUE_ORDER for the sequential study, and each enabled technique
    is then removed one-at-a-time for the ablation study.
    """
    enabled = [
        name
        for name in REPORT_TECHNIQUE_ORDER
        if REPORT_TECHNIQUES.get(name, False)
    ]

    configs: List[Dict[str, Any]] = []
    basic_states, basic_weights = report_state_and_weights(
        source_switches, source_weights, []
    )

    sequential_names: Dict[Tuple[str, ...], str] = {}
    active: List[str] = []

    if suite in ("all", "sequential"):
        basic_name = "01_basic_greedy"
        configs.append(
            {
                "name": basic_name,
                "kind": "sequential",
                "report_stage": "Basic method",
                "active_report_techniques": [],
                "description": (
                    "Greedy shortest-path/supply-centre distance heuristic only; "
                    "the three report techniques are OFF and extra tactical/opponent "
                    "scoring terms are neutralised."
                ),
                "switch_states": basic_states,
                "weight_table": basic_weights,
                "comparison_reference": None,
            }
        )
        previous_name = basic_name

        for index, technique in enumerate(enabled, start=2):
            active.append(technique)
            states, weights = report_state_and_weights(
                source_switches, source_weights, active
            )

            # Stable, human-readable names for the agreed current three.
            if technique == "tactical_planning":
                suffix = "plus_tactical_planning"
            elif technique == "long_term_strategy":
                suffix = "plus_long_term_strategy"
            elif technique == "opponent_modelling":
                suffix = "plus_opponent_modelling"
            else:
                suffix = "plus_" + technique

            name = f"{index:02d}_{suffix}"
            sequential_names[tuple(active)] = name
            configs.append(
                {
                    "name": name,
                    "kind": "sequential",
                    "report_stage": (
                        "Basic + "
                        + " + ".join(
                            f"Technique {REPORT_TECHNIQUE_ORDER.index(t) + 1}"
                            for t in active
                        )
                    ),
                    "active_report_techniques": list(active),
                    "description": (
                        f"Sequentially adds {REPORT_TECHNIQUE_LABELS[technique]} "
                        "to the previous report experiment stage."
                    ),
                    "switch_states": states,
                    "weight_table": weights,
                    "comparison_reference": previous_name,
                }
            )
            previous_name = name

    all_enabled = list(enabled)
    full_states, full_weights = report_state_and_weights(
        source_switches, source_weights, all_enabled
    )

    # The full report-technique reference needs to exist for ablations even if
    # the user runs --suite ablation directly.
    if suite == "ablation":
        full_reference_name = "04_all_report_techniques"
        configs.append(
            {
                "name": full_reference_name,
                "kind": "ablation_reference",
                "report_stage": "All enabled report techniques",
                "active_report_techniques": list(all_enabled),
                "description": (
                    "Common reference for the one-at-a-time ablation study."
                ),
                "switch_states": full_states,
                "weight_table": full_weights,
                "comparison_reference": None,
            }
        )
    else:
        # With the current agreed three enabled, this resolves to
        # 04_plus_opponent_modelling.
        full_reference_name = sequential_names.get(tuple(all_enabled))
        if full_reference_name is None and all_enabled:
            full_reference_name = "04_all_report_techniques"

    if suite in ("all", "ablation") and all_enabled:
        next_number = len(enabled) + 2
        for offset, technique in enumerate(all_enabled):
            active_without = [t for t in all_enabled if t != technique]
            states, weights = report_state_and_weights(
                source_switches, source_weights, active_without
            )
            name = f"{next_number + offset:02d}_without_{technique}"
            configs.append(
                {
                    "name": name,
                    "kind": "ablation",
                    "report_stage": (
                        f"Ablation: remove {REPORT_TECHNIQUE_LABELS[technique]}"
                    ),
                    "active_report_techniques": active_without,
                    "description": (
                        "Starts from the common all-report-techniques condition "
                        f"and removes only {REPORT_TECHNIQUE_LABELS[technique]}."
                    ),
                    "switch_states": states,
                    "weight_table": weights,
                    "comparison_reference": full_reference_name,
                }
            )

    if suite in ("all", "final"):
        final_number = len(enabled) + 2 + (len(enabled) if suite == "all" else 0)
        configs.append(
            {
                "name": f"{final_number:02d}_final_agent_reference",
                "kind": "reference",
                "report_stage": "Exact submitted-agent reference",
                "active_report_techniques": None,
                "description": (
                    "Exact source agent: no test-time switch or weight changes. "
                    "This is kept separate because the current source uses "
                    "opponent_multiplier=0.0, whereas the Technique 3 experiment "
                    "uses an explicit non-zero ON value."
                ),
                "switch_states": None,
                "weight_table": None,
                "comparison_reference": None,
            }
        )

    return configs


# ===========================================================================
# 5. APPLY ONE CONFIGURATION INSIDE A FRESH CHILD PROCESS
# ===========================================================================


def apply_configuration(agent_cls, switch_states, weight_table) -> None:
    if switch_states is not None:
        current = copy.deepcopy(agent_cls.TECHNIQUES)
        unknown = sorted(set(switch_states) - set(current))
        if unknown:
            raise KeyError("Unknown agent TECHNIQUES switch(es): " + ", ".join(unknown))
        current.update({k: bool(v) for k, v in switch_states.items()})
        agent_cls.TECHNIQUES = current

    if weight_table is not None:
        agent_cls.STAGE_WEIGHTS = copy.deepcopy(weight_table)


# ===========================================================================
# 6. SCENARIO 1 / SCENARIO 2 BENCHMARK LOGIC
# ===========================================================================


def scenario_pool(scenario: str, StaticAgent, RandomAgent, AttitudeAgent, GreedyAgent):
    if scenario == "1":
        return (StaticAgent,)
    if scenario == "2":
        # Same weighted structure as the supplied local test: Random is less
        # likely than Attitude and Greedy.
        return (RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent)
    raise ValueError(f"Unsupported local scenario {scenario!r}")


def benchmark_one_scenario(
    *,
    scenario: str,
    seed: int,
    repeats: int,
    end_year: int,
    student_cls,
    run_one_game,
    StaticAgent,
    RandomAgent,
    AttitudeAgent,
    GreedyAgent,
) -> Dict[str, Any]:
    pool = scenario_pool(
        scenario, StaticAgent, RandomAgent, AttitudeAgent, GreedyAgent
    )

    # Separate RNG for choosing opponent classes. This makes the opponent-type
    # lineup identical for the same seed/scenario across every configuration,
    # even if different agent behaviours consume different amounts of global
    # randomness while a game is running.
    lineup_rng = random.Random(seed * 1009 + int(scenario) * 9173)

    records: List[Dict[str, Any]] = []
    scores: Dict[str, List[int]] = defaultdict(list)
    outcomes: Dict[str, List[str]] = defaultdict(list)

    for repeat_idx in range(repeats):
        for controlled_power in ALL_POWERS:
            agents = {}
            opponent_lineup: Dict[str, str] = {}

            for power in ALL_POWERS:
                if power == controlled_power:
                    agents[power] = student_cls()
                else:
                    opponent_cls = lineup_rng.choice(pool)
                    agents[power] = opponent_cls()
                    opponent_lineup[power] = opponent_cls.__name__

            raw_results, actual_end_year = run_one_game(agents, end_year=end_year)
            sc = min(int(raw_results[controlled_power]), 18)
            outcome = classify_outcome(sc)

            scores[controlled_power].append(sc)
            scores["ALL"].append(sc)
            outcomes[controlled_power].append(outcome)
            outcomes["ALL"].append(outcome)

            records.append(
                {
                    "repeat": repeat_idx + 1,
                    "controlled_power": controlled_power,
                    "opponent_lineup": opponent_lineup,
                    "final_sc": sc,
                    "outcome": outcome,
                    "end_year": int(actual_end_year),
                }
            )

    rows: Dict[str, Dict[str, Any]] = {}
    for power in (*ALL_POWERS, "ALL"):
        vals = scores[power]
        outs = outcomes[power]
        n = len(vals)
        rows[power] = {
            "games": n,
            "avg_sc": statistics.fmean(vals) if vals else 0.0,
            "std_sc": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "win_rate": 100.0 * outs.count("WIN") / n if n else 0.0,
            "survive_rate": 100.0 * outs.count("SURVIVE") / n if n else 0.0,
            "defeat_rate": 100.0 * outs.count("DEFEAT") / n if n else 0.0,
        }

    return {
        "scenario": scenario,
        "rows": rows,
        "records": records,
    }


# ===========================================================================
# 7. CHILD WORKER - ONE CONFIGURATION x ONE SEED
# ===========================================================================


def worker_main(args) -> int:
    import numpy as np
    from agent_baselines import AttitudeAgent, GreedyAgent, RandomAgent, StaticAgent
    from game import run_one_game

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    agent_cls = load_agent_class(args.agent)

    switch_states = (
        json.loads(args.switch_states_json) if args.switch_states_json else None
    )
    weight_table = (
        json.loads(args.weight_table_json) if args.weight_table_json else None
    )

    try:
        apply_configuration(agent_cls, switch_states, weight_table)

        scenario_results = {}
        for scenario in ("1", "2"):
            scenario_results[scenario] = benchmark_one_scenario(
                scenario=scenario,
                seed=seed,
                repeats=args.repeats,
                end_year=args.end_year,
                student_cls=agent_cls,
                run_one_game=run_one_game,
                StaticAgent=StaticAgent,
                RandomAgent=RandomAgent,
                AttitudeAgent=AttitudeAgent,
                GreedyAgent=GreedyAgent,
            )

        payload = {
            "ok": True,
            "configuration": args.configuration,
            "seed": seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "switch_states": switch_states,
            "weight_table": weight_table,
            "scenario_results": scenario_results,
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "configuration": args.configuration,
            "seed": seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    print(RESULT_SENTINEL + json.dumps(payload, separators=(",", ":")))
    return 0 if payload["ok"] else 2


def run_worker(
    *,
    agent: str,
    configuration: Mapping[str, Any],
    seed: int,
    repeats: int,
    end_year: int,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--agent",
        agent,
        "--configuration",
        configuration["name"],
        "--seed",
        str(seed),
        "--repeats",
        str(repeats),
        "--end-year",
        str(end_year),
    ]

    if configuration["switch_states"] is not None:
        cmd += [
            "--switch-states-json",
            json.dumps(configuration["switch_states"], separators=(",", ":")),
        ]
    if configuration["weight_table"] is not None:
        cmd += [
            "--weight-table-json",
            json.dumps(configuration["weight_table"], separators=(",", ":")),
        ]

    proc = subprocess.run(
        cmd,
        cwd=os.getcwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            payload = json.loads(line[len(RESULT_SENTINEL) :])

    if payload is None:
        raise RuntimeError(
            f"Worker {configuration['name']}/seed {seed} returned no result.\n"
            f"Last output:\n{proc.stdout[-5000:]}"
        )
    if not payload.get("ok"):
        raise RuntimeError(
            f"{configuration['name']}/seed {seed} failed: "
            f"{payload.get('error_type')}: {payload.get('error')}"
        )
    return payload


# ===========================================================================
# 8. AGGREGATION AND COMPARISONS
# ===========================================================================


def aggregate_scenario(seed_runs: Sequence[Mapping[str, Any]], scenario: str) -> Dict[str, Any]:
    records: List[Mapping[str, Any]] = []
    seed_means: List[float] = []
    seed_win_rates: List[float] = []

    for run in seed_runs:
        result = run["scenario_results"][scenario]
        records.extend(result["records"])
        seed_all = result["rows"]["ALL"]
        seed_means.append(float(seed_all["avg_sc"]))
        seed_win_rates.append(float(seed_all["win_rate"]))

    scores: Dict[str, List[int]] = defaultdict(list)
    outcomes: Dict[str, List[str]] = defaultdict(list)

    for record in records:
        power = record["controlled_power"]
        sc = int(record["final_sc"])
        outcome = str(record["outcome"])

        scores[power].append(sc)
        scores["ALL"].append(sc)
        outcomes[power].append(outcome)
        outcomes["ALL"].append(outcome)

    rows: Dict[str, Dict[str, Any]] = {}
    for power in (*ALL_POWERS, "ALL"):
        vals = scores[power]
        outs = outcomes[power]
        n = len(vals)
        rows[power] = {
            "games": n,
            "avg_sc": statistics.fmean(vals) if vals else 0.0,
            "std_sc": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "win_rate": 100.0 * outs.count("WIN") / n if n else 0.0,
            "survive_rate": 100.0 * outs.count("SURVIVE") / n if n else 0.0,
            "defeat_rate": 100.0 * outs.count("DEFEAT") / n if n else 0.0,
        }

    return {
        "rows": rows,
        "seed_to_seed": {
            "avg_sc_sd": statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0,
            "win_rate_sd": (
                statistics.pstdev(seed_win_rates) if len(seed_win_rates) > 1 else 0.0
            ),
        },
    }


def rebuild_summary(data: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name, result_info in data["results"].items():
        runs = [r for r in result_info.get("seed_runs", []) if r.get("ok", True)]
        if not runs:
            continue
        summary[name] = {
            "scenario1": aggregate_scenario(runs, "1"),
            "scenario2": aggregate_scenario(runs, "2"),
            "completed_seeds": sorted(int(r["seed"]) for r in runs),
        }
    return summary


def compare_summary(
    test_summary: Mapping[str, Any], reference_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scenario_key in ("scenario1", "scenario2"):
        test = test_summary[scenario_key]["rows"]["ALL"]
        ref = reference_summary[scenario_key]["rows"]["ALL"]
        out[scenario_key] = {
            "delta_avg_sc": test["avg_sc"] - ref["avg_sc"],
            "delta_win_rate_percentage_points": test["win_rate"] - ref["win_rate"],
            "delta_survive_rate_percentage_points": (
                test["survive_rate"] - ref["survive_rate"]
            ),
            "delta_defeat_rate_percentage_points": (
                test["defeat_rate"] - ref["defeat_rate"]
            ),
        }
    return out


def rebuild_comparisons(
    data: Mapping[str, Any], config_lookup: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    summary = data["summary"]
    comparisons: Dict[str, Any] = {}

    for name, config in config_lookup.items():
        reference = config.get("comparison_reference")
        if not reference:
            continue
        if name not in summary or reference not in summary:
            continue
        comparisons[name] = {
            "reference": reference,
            "meaning": (
                "Sequential: this configuration minus the previous development stage."
                if config["kind"] == "sequential"
                else "Ablation: configuration-with-technique-removed minus all-three-techniques reference."
            ),
            "delta": compare_summary(summary[name], summary[reference]),
        }

    return comparisons


def summary_csv_rows(
    data: Mapping[str, Any], config_lookup: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for name, config in config_lookup.items():
        if name not in data["summary"]:
            continue

        summary = data["summary"][name]
        s1 = summary["scenario1"]["rows"]["ALL"]
        s2 = summary["scenario2"]["rows"]["ALL"]
        comparison = data.get("comparisons", {}).get(name)

        row: Dict[str, Any] = {
            "configuration": name,
            "kind": config["kind"],
            "report_stage": config["report_stage"],
            "completed_seeds": ",".join(map(str, summary["completed_seeds"])),
            "s1_games": s1["games"],
            "s1_avg_sc": round(s1["avg_sc"], 6),
            "s1_std_sc": round(s1["std_sc"], 6),
            "s1_seed_sd_sc": round(summary["scenario1"]["seed_to_seed"]["avg_sc_sd"], 6),
            "s1_win_rate": round(s1["win_rate"], 6),
            "s1_survive_rate": round(s1["survive_rate"], 6),
            "s1_defeat_rate": round(s1["defeat_rate"], 6),
            "s2_games": s2["games"],
            "s2_avg_sc": round(s2["avg_sc"], 6),
            "s2_std_sc": round(s2["std_sc"], 6),
            "s2_seed_sd_sc": round(summary["scenario2"]["seed_to_seed"]["avg_sc_sd"], 6),
            "s2_win_rate": round(s2["win_rate"], 6),
            "s2_survive_rate": round(s2["survive_rate"], 6),
            "s2_defeat_rate": round(s2["defeat_rate"], 6),
            "comparison_reference": config.get("comparison_reference") or "",
            "s1_delta_sc_vs_reference": "",
            "s1_delta_win_pp_vs_reference": "",
            "s2_delta_sc_vs_reference": "",
            "s2_delta_win_pp_vs_reference": "",
        }

        if comparison:
            delta1 = comparison["delta"]["scenario1"]
            delta2 = comparison["delta"]["scenario2"]
            row["s1_delta_sc_vs_reference"] = round(delta1["delta_avg_sc"], 6)
            row["s1_delta_win_pp_vs_reference"] = round(
                delta1["delta_win_rate_percentage_points"], 6
            )
            row["s2_delta_sc_vs_reference"] = round(delta2["delta_avg_sc"], 6)
            row["s2_delta_win_pp_vs_reference"] = round(
                delta2["delta_win_rate_percentage_points"], 6
            )

        rows.append(row)

    return rows


def checkpoint(
    *,
    data: Dict[str, Any],
    config_lookup: Mapping[str, Mapping[str, Any]],
    json_path: Path,
    csv_path: Path,
) -> None:
    data["summary"] = rebuild_summary(data)
    data["comparisons"] = rebuild_comparisons(data, config_lookup)
    write_json(json_path, data)
    write_csv(csv_path, summary_csv_rows(data, config_lookup))


# ===========================================================================
# 9. HUMAN-READABLE CONSOLE OUTPUT
# ===========================================================================


def print_seed_result(name: str, result: Mapping[str, Any]) -> None:
    s1 = result["scenario_results"]["1"]["rows"]["ALL"]
    s2 = result["scenario_results"]["2"]["rows"]["ALL"]
    print(
        f"    {name:<36} "
        f"S1 {s1['avg_sc']:.2f} SC / {s1['win_rate']:.1f}% win | "
        f"S2 {s2['avg_sc']:.2f} SC / {s2['win_rate']:.1f}% win"
    )


def print_final_summary(data: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 92)
    print("FINAL REPORT EXPERIMENT SUMMARY")
    print("=" * 92)

    for config in configs:
        name = config["name"]
        if name not in data["summary"]:
            continue
        s1 = data["summary"][name]["scenario1"]["rows"]["ALL"]
        s2 = data["summary"][name]["scenario2"]["rows"]["ALL"]
        print(
            f"{name:<36} "
            f"S1 {s1['avg_sc']:>5.2f} SC / {s1['win_rate']:>6.2f}% | "
            f"S2 {s2['avg_sc']:>5.2f} SC / {s2['win_rate']:>6.2f}%"
        )

    if data.get("comparisons"):
        print("\nTechnique deltas:")
        for config in configs:
            name = config["name"]
            comp = data["comparisons"].get(name)
            if not comp:
                continue
            d1 = comp["delta"]["scenario1"]
            d2 = comp["delta"]["scenario2"]
            print(
                f"  {name:<34} vs {comp['reference']:<31} "
                f"S1 ΔSC={d1['delta_avg_sc']:+.2f}, Δwin={d1['delta_win_rate_percentage_points']:+.1f}pp | "
                f"S2 ΔSC={d2['delta_avg_sc']:+.2f}, Δwin={d2['delta_win_rate_percentage_points']:+.1f}pp"
            )


# ===========================================================================
# 10. PARENT EXPERIMENT DRIVER
# ===========================================================================


def parent_main(args) -> int:
    agent_cls = load_agent_class(args.agent)
    structure = validate_agent_structure(agent_cls)
    source_switches = structure["techniques"]
    source_weights = structure["stage_weights"]

    configs = build_configurations(source_switches, source_weights, args.suite)
    config_lookup = {config["name"]: config for config in configs}

    # Make the report-level grouping completely visible before a long run starts.
    print("=" * 92)
    print("GROUP 09 FINAL REPORT EXPERIMENT TESTER")
    print("=" * 92)
    print(f"Agent module              : {args.agent}")
    print(f"Seeds                     : {args.seeds}")
    print(f"Repeats / power / seed    : {args.repeats}")
    print(f"End year                  : {args.end_year}")
    print(f"Suite                     : {args.suite}")
    print(f"Opponent-model ON weight  : {OPPONENT_MODEL_ON_MULTIPLIER}")
    print("\nReport techniques:")
    print("  BASIC       : greedy shortest-path / supply-centre distance heuristic")
    print("  TECHNIQUE 1 : tactical multi-unit planning")
    for switch in TACTICAL_PLANNING_SWITCHES:
        print(f"                - {switch}")
    print(f"  TECHNIQUE 2 : persistent long-term strategy ({LONG_TERM_SWITCH})")
    print(f"  TECHNIQUE 3 : opponent modelling ({OPPONENT_MODEL_SWITCH})")
    print("\nScenarios:")
    print("  Scenario 1  : StaticAgent opponents")
    print("  Scenario 2  : Random/Attitude/Attitude/Greedy/Greedy weighted pool")

    approx_games = len(configs) * len(args.seeds) * args.repeats * len(ALL_POWERS) * 2
    print(f"\nConfigurations            : {len(configs)}")
    print(f"Approx. games             : {approx_games}")
    print(f"Checkpoint JSON           : {args.out}")
    print(f"Summary CSV               : {args.csv}")

    metadata = {
        "purpose": (
            "Experiments supporting the Group 09 report: basic greedy heuristic "
            "+ three agreed report-level techniques, tested sequentially and by ablation."
        ),
        "agent": args.agent,
        "seeds": list(args.seeds),
        "repeats_per_power_per_seed": args.repeats,
        "end_year": args.end_year,
        "suite": args.suite,
        "scenario1_opponents": ["StaticAgent"],
        "scenario2_pool": [
            "RandomAgent",
            "AttitudeAgent",
            "AttitudeAgent",
            "GreedyAgent",
            "GreedyAgent",
        ],
        "report_techniques": {
            "basic": "Greedy shortest-path / supply-centre distance heuristic",
            "technique_1": {
                "name": "Tactical multi-unit planning",
                "agent_switches": list(TACTICAL_PLANNING_SWITCHES),
            },
            "technique_2": {
                "name": "Persistent long-term strategy",
                "agent_switches": [LONG_TERM_SWITCH],
            },
            "technique_3": {
                "name": "Opponent modelling",
                "agent_switches": [OPPONENT_MODEL_SWITCH],
                "experimental_on_multiplier": OPPONENT_MODEL_ON_MULTIPLIER,
                "source_multiplier_note": (
                    "The current source agent uses opponent_multiplier=0.0 in all stages, "
                    "so a non-zero value is required to genuinely evaluate Technique 3."
                ),
            },
        },
        "source_technique_switches": source_switches,
        "source_stage_weights": source_weights,
        "reproducibility": (
            "Each seed runs in a fresh subprocess with PYTHONHASHSEED, random.seed, "
            "and numpy.random.seed set to the same seed. Opponent class lineups use "
            "a separate deterministic RNG so they remain matched across configurations."
        ),
    }

    if args.resume and args.out.exists():
        data = json.loads(args.out.read_text(encoding="utf-8"))
        # Do not silently resume incompatible experiments.
        old_meta = data.get("metadata", {})
        if old_meta.get("agent") != args.agent:
            raise SystemExit(
                f"Cannot resume: existing JSON was created for agent "
                f"{old_meta.get('agent')!r}, not {args.agent!r}."
            )
        if old_meta.get("seeds") != list(args.seeds):
            raise SystemExit(
                "Cannot resume: existing JSON uses different seeds. Delete the old "
                "results file or rerun with matching --seeds."
            )
        if old_meta.get("repeats_per_power_per_seed") != args.repeats:
            raise SystemExit(
                "Cannot resume: existing JSON uses a different --repeats value."
            )
        if old_meta.get("end_year") != args.end_year:
            raise SystemExit(
                "Cannot resume: existing JSON uses a different --end-year value."
            )
        if old_meta.get("suite") != args.suite:
            raise SystemExit(
                "Cannot resume: existing JSON uses a different --suite value."
            )
        print(f"\nResuming from {args.out}")
    else:
        data = {
            "metadata": metadata,
            "configurations": {
                config["name"]: {
                    "name": config["name"],
                    "kind": config["kind"],
                    "report_stage": config["report_stage"],
                    "description": config["description"],
                    "active_report_techniques": config.get("active_report_techniques"),
                    "switch_states": config["switch_states"],
                    "weight_table": config["weight_table"],
                    "comparison_reference": config["comparison_reference"],
                }
                for config in configs
            },
            "results": {
                config["name"]: {"seed_runs": [], "failures": []}
                for config in configs
            },
            "summary": {},
            "comparisons": {},
        }
        checkpoint(
            data=data,
            config_lookup=config_lookup,
            json_path=args.out,
            csv_path=args.csv,
        )

    start = time.monotonic()

    for config in configs:
        name = config["name"]
        print("\n" + "-" * 92)
        print(f"[{name}] {config['report_stage']}")
        print(config["description"])

        result_bucket = data["results"].setdefault(
            name, {"seed_runs": [], "failures": []}
        )
        completed = {
            int(run["seed"])
            for run in result_bucket.get("seed_runs", [])
            if run.get("ok", True)
        }

        for seed in args.seeds:
            if seed in completed:
                print(f"  seed {seed}: already saved, skipping")
                continue

            print(f"  seed {seed}: running Scenario 1 + Scenario 2 ...", flush=True)
            try:
                result = run_worker(
                    agent=args.agent,
                    configuration=config,
                    seed=seed,
                    repeats=args.repeats,
                    end_year=args.end_year,
                )
                result_bucket["seed_runs"].append(result)
                result_bucket["seed_runs"].sort(key=lambda item: int(item["seed"]))

                checkpoint(
                    data=data,
                    config_lookup=config_lookup,
                    json_path=args.out,
                    csv_path=args.csv,
                )
                print_seed_result(name, result)
                print("    results checkpointed")

            except Exception as exc:
                failure = {
                    "configuration": name,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                result_bucket.setdefault("failures", []).append(failure)
                checkpoint(
                    data=data,
                    config_lookup=config_lookup,
                    json_path=args.out,
                    csv_path=args.csv,
                )
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                if not args.continue_on_error:
                    raise

    checkpoint(
        data=data,
        config_lookup=config_lookup,
        json_path=args.out,
        csv_path=args.csv,
    )

    print_final_summary(data, configs)
    print("\nSaved automatically:")
    print(f"  JSON: {args.out}")
    print(f"  CSV : {args.csv}")
    print(f"Elapsed: {(time.monotonic() - start) / 60:.1f} minutes")
    return 0


# ===========================================================================
# 11. COMMAND-LINE INTERFACE
# ===========================================================================


def build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("_worker")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--switch-states-json", default=None)
    parser.add_argument("--weight-table-json", default=None)
    return parser


def build_parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group 09 final report experiment tester: basic greedy heuristic + "
            "three agreed techniques, evaluated in both local scenarios."
        )
    )
    parser.add_argument("agent", nargs="?", default="agent_09")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 0,1,2,3,4",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--suite",
        choices=("all", "sequential", "ablation", "final"),
        default="all",
        help=(
            "all = sequential + ablations + exact final reference; "
            "sequential = add techniques one by one; "
            "ablation = remove one report technique at a time; "
            "final = exact source-agent reference only"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using already checkpointed seeds in the output JSON.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed seed/configuration and continue with the rest.",
    )
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        args = build_worker_parser().parse_args()
        return worker_main(args)

    args = build_parent_parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
