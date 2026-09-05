#!/usr/bin/env python3
"""Run the four-objective discrete Retro* benchmark for seed 2.

Input:
    output/discrete_for_retro_test_5000_seed2.smi

Outputs:
    output/retro_recorded_4d/discrete_retro_results_seed2.csv
    output/retro_recorded_4d/discrete_retro_solved_seed2.smi
    output/retro_recorded_4d/discrete_retro_unsolved_seed2.smi
    output/retro_recorded_4d/discrete_retro_summary.csv
"""

import concurrent.futures
import csv
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# ==================== Configuration ====================

BASE_PATH = "/home/liuyansong/ChemistGA-master/ChemistGA-master"

TASK_PREFIX = "discrete"
SEED = 2

EXPECTED_MOLECULES = 5000
WORKERS = 4
ITERATIONS = 150
EXPANSION_TOPK = 50

INPUT_FILE = os.path.join(
    BASE_PATH,
    "output",
    f"{TASK_PREFIX}_for_retro_test_5000_seed{SEED}.smi",
)

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_4d",
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{TASK_PREFIX}_retro_checkpoint_seed{SEED}.csv",
)

planner = None


# ==================== Retro* ====================

def load_planner():
    global planner

    sys.path.insert(
        0,
        os.path.join(BASE_PATH, "retro_star"),
    )

    from retro_star.api import RSPlanner

    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=ITERATIONS,
        expansion_topk=EXPANSION_TOPK,
    )


def plan_single_smi(smiles):
    """Run Retro* for one molecule."""

    try:
        result = planner.plan(smiles)

        if result is None:
            return 0, ""

        if isinstance(result, dict):
            return int(bool(result.get("succ"))), ""

        if hasattr(result, "succ"):
            return int(bool(result.succ)), ""

        # Compatibility with the previous Retro* API
        return 1, ""

    except Exception as exc:
        return 0, "%s: %s" % (
            type(exc).__name__,
            str(exc),
        )


# ==================== Input and checkpoint ====================

def read_input(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as handle:
        return [
            line.strip().split()[0]
            for line in handle
            if line.strip()
        ]


def load_previous_results(path):
    columns = [
        "input_index",
        "SMILES",
        "solved",
        "error",
    ]

    if (
        not os.path.exists(path)
        or os.path.getsize(path) == 0
    ):
        return pd.DataFrame(columns=columns)

    frame = pd.read_csv(path)

    required = set(columns)
    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Checkpoint is missing columns: "
            f"{sorted(missing)}"
        )

    return frame


def append_record(path, row, write_header):
    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_index",
                "SMILES",
                "solved",
                "error",
            ],
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# ==================== Finalization ====================

def finalize_results(smiles_list):
    frame = pd.read_csv(CHECKPOINT_FILE)

    # If an errored molecule was rerun, retain its latest result.
    frame = frame.drop_duplicates(
        subset=["input_index"],
        keep="last",
    )

    frame = frame.sort_values(
        "input_index"
    ).reset_index(drop=True)

    expected_indices = set(
        range(len(smiles_list))
    )

    observed_indices = set(
        frame["input_index"]
        .astype(int)
        .tolist()
    )

    missing_indices = sorted(
        expected_indices - observed_indices
    )

    if missing_indices:
        raise RuntimeError(
            f"Seed {SEED} is incomplete: "
            f"{len(missing_indices)} molecules have no result. "
            f"Run the script again to resume."
        )

    final_csv = os.path.join(
        OUTPUT_DIR,
        f"{TASK_PREFIX}_retro_results_seed{SEED}.csv",
    )

    solved_file = os.path.join(
        OUTPUT_DIR,
        f"{TASK_PREFIX}_retro_solved_seed{SEED}.smi",
    )

    unsolved_file = os.path.join(
        OUTPUT_DIR,
        f"{TASK_PREFIX}_retro_unsolved_seed{SEED}.smi",
    )

    frame.to_csv(
        final_csv,
        index=False,
    )

    solved_frame = frame[
        frame["solved"].astype(int) == 1
    ]

    unsolved_frame = frame[
        frame["solved"].astype(int) == 0
    ]

    solved_frame["SMILES"].to_csv(
        solved_file,
        index=False,
        header=False,
    )

    unsolved_frame["SMILES"].to_csv(
        unsolved_file,
        index=False,
        header=False,
    )

    error_count = int(
        frame["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
        .gt(0)
        .sum()
    )

    solved_count = len(solved_frame)
    total_count = len(smiles_list)

    synthesizability = (
        100.0 * solved_count / total_count
        if total_count > 0
        else 0.0
    )

    summary = {
        "task": "four_objective_discrete",
        "seed": SEED,
        "input_molecules": total_count,
        "solved_molecules": solved_count,
        "unsolved_molecules": len(unsolved_frame),
        "planner_errors": error_count,
        "synthesizability_percent": synthesizability,
        "result_csv": final_csv,
        "solved_smi": solved_file,
        "unsolved_smi": unsolved_file,
    }

    summary_file = os.path.join(
        OUTPUT_DIR,
        f"{TASK_PREFIX}_retro_summary.csv",
    )

    pd.DataFrame([summary]).to_csv(
        summary_file,
        index=False,
    )

    return summary, summary_file


# ==================== Main ====================

def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    smiles_list = read_input(
        INPUT_FILE
    )

    if len(smiles_list) != EXPECTED_MOLECULES:
        raise RuntimeError(
            f"{INPUT_FILE} contains {len(smiles_list)} molecules; "
            f"the benchmark requires exactly "
            f"{EXPECTED_MOLECULES} entries."
        )

    duplicate_count = (
        len(smiles_list) - len(set(smiles_list))
    )

    print("=" * 72, flush=True)
    print(
        "Four-objective discrete Retro* evaluation | seed 2",
        flush=True,
    )
    print("=" * 72, flush=True)
    print(
        f"Input file: {INPUT_FILE}",
        flush=True,
    )
    print(
        f"Input molecules: {len(smiles_list)}",
        flush=True,
    )

    if duplicate_count:
        print(
            f"WARNING: input contains {duplicate_count} duplicate "
            f"SMILES entries; they will not be removed.",
            flush=True,
        )

    previous = load_previous_results(
        CHECKPOINT_FILE
    )

    if len(previous):
        previous = previous.drop_duplicates(
            subset=["input_index"],
            keep="last",
        )

        error_text = (
            previous["error"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        completed_indices = set(
            previous.loc[
                error_text.str.len() == 0,
                "input_index",
            ]
            .astype(int)
            .tolist()
        )
    else:
        completed_indices = set()

    pending = [
        (index, smiles)
        for index, smiles in enumerate(smiles_list)
        if index not in completed_indices
    ]

    print(
        "Seed %d: %d/%d already recorded; %d pending."
        % (
            SEED,
            len(completed_indices),
            len(smiles_list),
            len(pending),
        ),
        flush=True,
    )

    if pending:
        print("Loading Retro*...", flush=True)
        load_planner()
        print("Retro* loaded.", flush=True)

        start_time = time.time()

        write_header = (
            not os.path.exists(CHECKPOINT_FILE)
            or os.path.getsize(CHECKPOINT_FILE) == 0
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    plan_single_smi,
                    smiles,
                ): (index, smiles)
                for index, smiles in pending
            }

            for number, future in enumerate(
                concurrent.futures.as_completed(
                    future_map
                ),
                start=1,
            ):
                index, smiles = future_map[future]

                try:
                    solved, error = future.result()

                except Exception as exc:
                    solved = 0
                    error = "%s: %s" % (
                        type(exc).__name__,
                        str(exc),
                    )

                append_record(
                    CHECKPOINT_FILE,
                    {
                        "input_index": index,
                        "SMILES": smiles,
                        "solved": int(solved),
                        "error": error,
                    },
                    write_header,
                )

                write_header = False

                if (
                    number % 50 == 0
                    or number == len(pending)
                ):
                    elapsed = time.time() - start_time

                    total_recorded = (
                        len(completed_indices)
                        + number
                    )

                    print(
                        "%d/%d recorded | elapsed: %.1f s"
                        % (
                            total_recorded,
                            len(smiles_list),
                            elapsed,
                        ),
                        flush=True,
                    )

    summary, summary_file = finalize_results(
        smiles_list
    )

    print("\n" + "=" * 72, flush=True)
    print(
        "Four-objective discrete result | seed 2",
        flush=True,
    )
    print("=" * 72, flush=True)

    print(
        "Synthesizability: %d/%d (%.2f%%)"
        % (
            summary["solved_molecules"],
            summary["input_molecules"],
            summary["synthesizability_percent"],
        ),
        flush=True,
    )

    print(
        f"Unsolved molecules: "
        f"{summary['unsolved_molecules']}",
        flush=True,
    )

    print(
        f"Planner errors: "
        f"{summary['planner_errors']}",
        flush=True,
    )

    print(
        f"Results CSV: "
        f"{summary['result_csv']}",
        flush=True,
    )

    print(
        f"Solved SMILES: "
        f"{summary['solved_smi']}",
        flush=True,
    )

    print(
        f"Unsolved SMILES: "
        f"{summary['unsolved_smi']}",
        flush=True,
    )

    print(
        f"Summary: {summary_file}",
        flush=True,
    )

    if summary["planner_errors"] > 0:
        print(
            "WARNING: planner errors were recorded. "
            "Run the script again to retry failed entries.",
            flush=True,
        )


if __name__ == "__main__":
    main()