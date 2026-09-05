#!/usr/bin/env python3
"""Run the DRD2 three-objective Retro* benchmark and retain molecule identities.

The input for each seed must already contain 5,000 generated molecules that
satisfy the DRD2/QED/SA property criteria. Unlike the earlier script, this
version does not truncate the input and records the result for every SMILES.

Edit TASK_PREFIX below or pass --task-prefix on the command line:
    drd_continuous
    drd_discrete
"""

import argparse
import concurrent.futures
import csv
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DEFAULT_BASE_PATH = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
DEFAULT_TASK_PREFIX = "drd_discrete"
EXPECTED_MOLECULES = 5000
WORKERS = 4
ITERATIONS = 150
EXPANSION_TOPK = 50

planner = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument(
        "--task-prefix",
        default=DEFAULT_TASK_PREFIX,
        choices=("drd_continuous", "drd_discrete"),
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    # Seed 2 is used by default for the single-seed re-evaluation requested here.
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    return parser.parse_args()


def load_planner(base_path):
    global planner
    sys.path.insert(0, os.path.join(base_path, "retro_star"))
    from retro_star.api import RSPlanner

    planner = RSPlanner(
        gpu=-1,
        use_value_fn=True,
        iterations=ITERATIONS,
        expansion_topk=EXPANSION_TOPK,
    )


def plan_single_smi(smi):
    """Return a solved flag and preserve planner errors for later auditing."""
    try:
        result = planner.plan(smi)
        if result is None:
            return 0, ""
        if isinstance(result, dict):
            return int(bool(result.get("succ"))), ""
        if hasattr(result, "succ"):
            return int(bool(result.succ)), ""
        # Keep compatibility with the return convention used by the old code.
        return 1, ""
    except Exception as exc:
        return 0, "%s: %s" % (type(exc).__name__, str(exc))


def read_input(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_previous_results(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["input_index", "SMILES", "solved", "error"])
    frame = pd.read_csv(path)
    required = {"input_index", "SMILES", "solved", "error"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Existing checkpoint has incompatible columns: %s" % path)
    return frame


def append_record(path, row, write_header):
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["input_index", "SMILES", "solved", "error"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def finalize_seed(raw_result_path, smiles_list, output_dir, task_prefix, seed):
    frame = pd.read_csv(raw_result_path)
    frame = frame.drop_duplicates(subset=["input_index"], keep="last")
    frame = frame.sort_values("input_index").reset_index(drop=True)

    expected_indices = set(range(len(smiles_list)))
    observed_indices = set(frame["input_index"].astype(int).tolist())
    missing = sorted(expected_indices - observed_indices)
    if missing:
        raise RuntimeError(
            "Seed %d is incomplete: %d molecules have no recorded result. Rerun to resume."
            % (seed, len(missing))
        )

    final_csv = os.path.join(
        output_dir, "%s_retro_results_seed%d.csv" % (task_prefix, seed)
    )
    frame.to_csv(final_csv, index=False)

    solved_frame = frame[frame["solved"].astype(int) == 1]
    unsolved_frame = frame[frame["solved"].astype(int) == 0]
    solved_path = os.path.join(
        output_dir, "%s_retro_solved_seed%d.smi" % (task_prefix, seed)
    )
    unsolved_path = os.path.join(
        output_dir, "%s_retro_unsolved_seed%d.smi" % (task_prefix, seed)
    )
    solved_frame["SMILES"].to_csv(solved_path, index=False, header=False)
    unsolved_frame["SMILES"].to_csv(unsolved_path, index=False, header=False)

    error_count = int(frame["error"].fillna("").astype(str).str.len().gt(0).sum())
    solved_count = len(solved_frame)
    rate = 100.0 * solved_count / len(smiles_list)
    return {
        "seed": seed,
        "input_molecules": len(smiles_list),
        "solved_molecules": solved_count,
        "unsolved_molecules": len(unsolved_frame),
        "planner_errors": error_count,
        "synthesizability_percent": rate,
        "result_csv": final_csv,
        "solved_smi": solved_path,
    }


def run_seed(base_path, output_dir, task_prefix, seed, workers):
    input_path = os.path.join(
        base_path,
        "output",
        "%s_for_retro_test_5000_seed%d.smi" % (task_prefix, seed),
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    smiles_list = read_input(input_path)
    if len(smiles_list) != EXPECTED_MOLECULES:
        raise RuntimeError(
            "%s contains %d molecules; the benchmark requires exactly %d."
            % (input_path, len(smiles_list), EXPECTED_MOLECULES)
        )

    duplicate_count = len(smiles_list) - len(set(smiles_list))
    if duplicate_count:
        print(
            "WARNING: seed %d input contains %d duplicate SMILES strings."
            % (seed, duplicate_count),
            flush=True,
        )

    raw_result_path = os.path.join(
        output_dir, "%s_retro_checkpoint_seed%d.csv" % (task_prefix, seed)
    )
    previous = load_previous_results(raw_result_path)
    if len(previous):
        previous = previous.drop_duplicates(subset=["input_index"], keep="last")
        error_text = previous["error"].fillna("").astype(str)
        completed = set(
            previous.loc[error_text.str.len() == 0, "input_index"].astype(int).tolist()
        )
    else:
        completed = set()

    pending = [
        (index, smi)
        for index, smi in enumerate(smiles_list)
        if index not in completed
    ]
    print(
        "Seed %d: %d/%d already recorded; %d pending."
        % (seed, len(completed), len(smiles_list), len(pending)),
        flush=True,
    )

    start = time.time()
    write_header = not os.path.exists(raw_result_path) or os.path.getsize(raw_result_path) == 0
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(plan_single_smi, smi): (index, smi)
                for index, smi in pending
            }
            for number, future in enumerate(
                concurrent.futures.as_completed(future_map), start=1
            ):
                index, smi = future_map[future]
                try:
                    solved, error = future.result()
                except Exception as exc:
                    solved = 0
                    error = "%s: %s" % (type(exc).__name__, str(exc))
                append_record(
                    raw_result_path,
                    {
                        "input_index": index,
                        "SMILES": smi,
                        "solved": int(solved),
                        "error": error,
                    },
                    write_header,
                )
                write_header = False

                if number % 50 == 0 or number == len(pending):
                    elapsed = time.time() - start
                    total_done = len(completed) + number
                    print(
                        "  %d/%d recorded | %.1f s"
                        % (total_done, len(smiles_list), elapsed),
                        flush=True,
                    )

    return finalize_seed(
        raw_result_path, smiles_list, output_dir, task_prefix, seed
    )


def main():
    args = parse_args()
    output_dir = os.path.join(args.base_path, "output", "retro_recorded_3d")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Retro*...", flush=True)
    load_planner(args.base_path)
    print("Retro* loaded.", flush=True)

    summaries = []
    for seed in args.seeds:
        print("\n" + "=" * 72, flush=True)
        print("%s | seed %d" % (args.task_prefix, seed), flush=True)
        print("=" * 72, flush=True)
        summaries.append(
            run_seed(
                args.base_path,
                output_dir,
                args.task_prefix,
                seed,
                args.workers,
            )
        )
        print(
            "Seed %d: %d/%d solved (%.2f%%), planner errors=%d"
            % (
                summaries[-1]["seed"],
                summaries[-1]["solved_molecules"],
                summaries[-1]["input_molecules"],
                summaries[-1]["synthesizability_percent"],
                summaries[-1]["planner_errors"],
            ),
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = os.path.join(
        output_dir, "%s_retro_summary.csv" % args.task_prefix
    )
    summary_frame.to_csv(summary_path, index=False)

    rates = summary_frame["synthesizability_percent"].to_numpy(dtype=float)
    print("\n" + "=" * 72)
    if len(summary_frame) == 1:
        row = summary_frame.iloc[0]
        print(
            "Seed %d synthesizability: %d/%d (%.2f%%)"
            % (
                int(row["seed"]),
                int(row["solved_molecules"]),
                int(row["input_molecules"]),
                float(row["synthesizability_percent"]),
            )
        )
    else:
        print("Mean synthesizability: %.2f +/- %.2f%%" % (np.mean(rates), np.std(rates)))
    print("Summary: %s" % summary_path)
    if int(summary_frame["planner_errors"].sum()) > 0:
        print("WARNING: planner errors were recorded. Inspect and rerun them before reporting.")


if __name__ == "__main__":
    main()
