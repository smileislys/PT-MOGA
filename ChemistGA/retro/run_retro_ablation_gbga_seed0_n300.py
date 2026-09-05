#!/usr/bin/env python3
"""Run the GB-GA crossover ablation Retro* experiment (seed 0, n=300).

The script samples 300 entries directly from the stored candidate file. It
does not canonicalize or deduplicate the source entries before sampling. A
fixed sampling seed makes the tested subset reproducible. Checkpoints allow
an interrupted Retro* run to resume without repeating completed entries.
"""

from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import os
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ================================ Controls ================================
RUN_ABLATION_GBGA = True
PREPARE_SAMPLE_ONLY = False

# =============================== Parameters ===============================
BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
EVOLUTION_SEED = 0
SAMPLE_SIZE = 300
SAMPLING_SEED = 2026
WORKERS = 4
RETRO_ITERATIONS = 150
EXPANSION_TOPK = 50
MAX_ATTEMPTS_PER_MOLECULE = 2

EXPERIMENT = "ablation_gbga"
METHOD = "GB-GA crossover ablation"
TASK = "JNK3/GSK3beta six-objective"
SOURCE_FILE = (
    BASE_PATH
    / "output"
    / f"ablation_gbga_6d_mpo_for_retro_test_500_seed{EVOLUTION_SEED}.smi"
)
OUTPUT_DIR = (
    BASE_PATH
    / "output"
    / f"retro_recorded_ablation_gbga_seed{EVOLUTION_SEED}_n{SAMPLE_SIZE}"
)
PREFIX = f"{EXPERIMENT}_seed{EVOLUTION_SEED}_n{SAMPLE_SIZE}"

SAMPLE_FILE = OUTPUT_DIR / f"{PREFIX}_sampled.smi"
CHECKPOINT_FILE = OUTPUT_DIR / f"{PREFIX}_retro_checkpoint.csv"
RESULTS_FILE = OUTPUT_DIR / f"{PREFIX}_retro_results.csv"
SOLVED_FILE = OUTPUT_DIR / f"{PREFIX}_retro_solved.smi"
UNSOLVED_FILE = OUTPUT_DIR / f"{PREFIX}_retro_unsolved.smi"
SUMMARY_FILE = OUTPUT_DIR / f"{PREFIX}_retro_summary.csv"
METADATA_FILE = OUTPUT_DIR / f"{PREFIX}_input_metadata.txt"

FIELDNAMES = ["input_index", "smiles", "solved", "runtime_seconds", "error"]

# =============================== Retro* setup ==============================
sys.path.insert(0, str(BASE_PATH / "retro_star"))
try:
    from retro_star.api import RSPlanner
except ImportError as exc:
    raise ImportError(f"Cannot import Retro* from {BASE_PATH / 'retro_star'}") from exc

PLANNER = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=RETRO_ITERATIONS,
    expansion_topk=EXPANSION_TOPK,
)


def result_is_solved(result) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        return bool(result.get("succ", False))
    if hasattr(result, "succ"):
        return bool(result.succ)
    return True


def plan_single_entry(item: tuple[int, str]) -> dict[str, object]:
    input_index, smiles = item
    started = time.time()
    solved = False
    error = ""

    for attempt in range(1, MAX_ATTEMPTS_PER_MOLECULE + 1):
        try:
            solved = result_is_solved(PLANNER.plan(smiles))
            error = ""
            break
        except Exception as exc:
            error = (
                f"attempt {attempt}/{MAX_ATTEMPTS_PER_MOLECULE}: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "input_index": input_index,
        "smiles": smiles,
        "solved": int(solved),
        "runtime_seconds": round(time.time() - started, 4),
        "error": error,
    }


# ============================= Input sampling ==============================
def read_smiles(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return [line.strip().split()[0] for line in handle if line.strip()]


def digest(smiles_list: list[str]) -> str:
    return hashlib.sha256(("\n".join(smiles_list) + "\n").encode("utf-8")).hexdigest()


def prepare_sample() -> tuple[list[str], int]:
    raw = read_smiles(SOURCE_FILE)
    if len(raw) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(raw)} source entries are available; {SAMPLE_SIZE} are required."
        )

    sampled = random.Random(SAMPLING_SEED).sample(raw, SAMPLE_SIZE)
    sample_text = "\n".join(sampled) + "\n"

    if SAMPLE_FILE.exists():
        if SAMPLE_FILE.read_text(encoding="utf-8") != sample_text:
            raise RuntimeError(
                f"Existing deterministic sample differs: {SAMPLE_FILE}. "
                "Move the output directory before changing the source pool or seeds."
            )
    else:
        SAMPLE_FILE.write_text(sample_text, encoding="utf-8")

    metadata = (
        f"experiment={EXPERIMENT}\n"
        f"evolution_seed={EVOLUTION_SEED}\n"
        f"sampling_seed={SAMPLING_SEED}\n"
        f"sample_size={SAMPLE_SIZE}\n"
        f"source_file={SOURCE_FILE}\n"
        f"source_entries={len(raw)}\n"
        f"sample_sha256={digest(sampled)}\n"
    )
    if METADATA_FILE.exists():
        if METADATA_FILE.read_text(encoding="utf-8") != metadata:
            raise RuntimeError(f"Input metadata mismatch: {METADATA_FILE}")
    else:
        METADATA_FILE.write_text(metadata, encoding="utf-8")

    return sampled, len(raw)


# ================================ Checkpoint ===============================
def load_checkpoint(smiles_list: list[str]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    if not CHECKPOINT_FILE.exists():
        return records

    with CHECKPOINT_FILE.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["input_index"])
            if index < 0 or index >= len(smiles_list):
                raise RuntimeError(f"Invalid checkpoint index: {index}")
            if row["smiles"] != smiles_list[index]:
                raise RuntimeError(f"Checkpoint/input mismatch at index {index}")
            if not row.get("error", ""):
                records[index] = {
                    "input_index": index,
                    "smiles": row["smiles"],
                    "solved": int(row["solved"]),
                    "runtime_seconds": float(row["runtime_seconds"]),
                    "error": "",
                }
    return records


def append_checkpoint(record: dict[str, object]) -> None:
    write_header = not CHECKPOINT_FILE.exists() or CHECKPOINT_FILE.stat().st_size == 0
    with CHECKPOINT_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


# ================================= Export ==================================
def export_results(
    source_entries: int,
    records_by_index: dict[int, dict[str, object]],
) -> None:
    ordered = [records_by_index[index] for index in range(SAMPLE_SIZE)]

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)

    solved = [str(row["smiles"]) for row in ordered if int(row["solved"]) == 1]
    unsolved = [str(row["smiles"]) for row in ordered if int(row["solved"]) == 0]
    SOLVED_FILE.write_text("\n".join(solved) + ("\n" if solved else ""), encoding="utf-8")
    UNSOLVED_FILE.write_text(
        "\n".join(unsolved) + ("\n" if unsolved else ""), encoding="utf-8"
    )

    rate = 100.0 * len(solved) / SAMPLE_SIZE
    summary = {
        "experiment": EXPERIMENT,
        "method": METHOD,
        "task": TASK,
        "evolution_seed": EVOLUTION_SEED,
        "sampling_seed": SAMPLING_SEED,
        "source_entries": source_entries,
        "evaluated": SAMPLE_SIZE,
        "retro_solved": len(solved),
        "retro_unsolved": len(unsolved),
        "retro_solved_rate_percent": f"{rate:.4f}",
        "iterations": RETRO_ITERATIONS,
        "expansion_topk": EXPANSION_TOPK,
    }
    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "=" * 76)
    print(f"{METHOD} | {TASK} | seed {EVOLUTION_SEED}")
    print(f"Source entries        : {source_entries}")
    print(f"Random sampled input  : {SAMPLE_SIZE}")
    print(f"Retro*-solved         : {len(solved)}")
    print(f"Retro*-unsolved       : {len(unsolved)}")
    print(f"Retro*-solved rate    : {rate:.2f}%")
    print(f"Solved molecule file  : {SOLVED_FILE}")
    print(f"Summary               : {SUMMARY_FILE}")
    print("=" * 76)


def main() -> None:
    if not RUN_ABLATION_GBGA:
        print("RUN_ABLATION_GBGA is False; nothing was executed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled, source_entries = prepare_sample()

    print("\n" + "#" * 76)
    print(f"Experiment : {EXPERIMENT}")
    print(f"Source     : {SOURCE_FILE}")
    print(f"Seed       : {EVOLUTION_SEED}")
    print(f"Sample     : {SAMPLE_SIZE} raw entries (sampling seed {SAMPLING_SEED})")
    print("#" * 76)

    if PREPARE_SAMPLE_ONLY:
        print(f"Sample prepared: {SAMPLE_FILE}")
        return

    records_by_index = load_checkpoint(sampled)
    pending = [
        (index, smiles)
        for index, smiles in enumerate(sampled)
        if index not in records_by_index
    ]
    print(f"Recorded {len(records_by_index)}/{SAMPLE_SIZE}; pending {len(pending)}.")

    started = time.time()
    newly_completed = 0
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(plan_single_entry, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                append_checkpoint(record)
                newly_completed += 1

                index = int(record["input_index"])
                if not record["error"]:
                    records_by_index[index] = record
                else:
                    print(
                        f"Entry {index} remains pending after retries: {record['error']}",
                        flush=True,
                    )

                if newly_completed % 10 == 0 or newly_completed == len(pending):
                    solved_so_far = sum(int(row["solved"]) for row in records_by_index.values())
                    print(
                        f"Completed {len(records_by_index)}/{SAMPLE_SIZE} | "
                        f"solved {solved_so_far} | elapsed {time.time() - started:.1f}s",
                        flush=True,
                    )

    if len(records_by_index) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Evaluation incomplete: {len(records_by_index)}/{SAMPLE_SIZE}. "
            "Run the script again to retry failed entries."
        )

    export_results(source_entries, records_by_index)


if __name__ == "__main__":
    main()
