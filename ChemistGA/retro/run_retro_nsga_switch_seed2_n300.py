#!/usr/bin/env python3
"""Run fixed-seed, fixed-size Retro* comparisons for NSGA-II/NSGA-III.

All enabled experiments use evolution seed 2 and a deterministic sample of
300 entries from the corresponding target-qualified candidate file. The input
entries are sampled as stored, without canonicalization or deduplication.
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

# ============================ 1. Switch buttons ============================
# Set one or more entries to True. Enabled experiments run sequentially.
RUN_SWITCHES = {
    "nsga2_3d": False,
    "nsga3_3d": False,
    "nsga2_4d": False,
    "nsga3_4d": False,
    "nsga2_6d": False,
    "nsga3_6d": True,
}

# True: create/verify the 300-molecule sample files without running Retro*.
PREPARE_SAMPLES_ONLY = False

# ============================== 2. Parameters ==============================
BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
EVOLUTION_SEED = 2
SAMPLE_SIZE = 300
SAMPLING_SEED = 2026
WORKERS = 4
RETRO_ITERATIONS = 150
EXPANSION_TOPK = 50
MAX_ATTEMPTS_PER_MOLECULE = 2

EXPERIMENTS = {
    "nsga2_3d": {
        "method": "NSGA-II",
        "task": "DRD2 three-objective",
        "input_name": "nsga2_drd_3d_for_retro_test_500_seed{seed}.smi",
    },
    "nsga3_3d": {
        "method": "NSGA-III",
        "task": "DRD2 three-objective",
        "input_name": "drd_nsga3_3d_for_retro_test_500_seed{seed}.smi",
    },
    "nsga2_4d": {
        "method": "NSGA-II",
        "task": "JNK3/GSK3beta four-objective",
        "input_name": "nsga2_4d_for_retro_test_500_seed{seed}.smi",
    },
    "nsga3_4d": {
        "method": "NSGA-III",
        "task": "JNK3/GSK3beta four-objective",
        "input_name": "nsga3_4d_for_retro_test_500_seed{seed}.smi",
    },
    "nsga2_6d": {
        "method": "NSGA-II",
        "task": "JNK3/GSK3beta six-objective",
        "input_name": "nsga2_6d_for_retro_test_500_seed{seed}.smi",
    },
    "nsga3_6d": {
        "method": "NSGA-III",
        "task": "JNK3/GSK3beta six-objective",
        "input_name": "6d_mpo_ALL_for_retro_test_500_seed{seed}.smi",
    },
}

OUTPUT_ROOT = (
    BASE_PATH
    / "output"
    / f"retro_recorded_nsga_seed{EVOLUTION_SEED}_n{SAMPLE_SIZE}"
)

# ============================== 3. Retro* setup =============================
sys.path.insert(0, str(BASE_PATH / "retro_star"))
try:
    from retro_star.api import RSPlanner
except ImportError as exc:
    raise ImportError(
        f"Cannot import Retro* from {BASE_PATH / 'retro_star'}"
    ) from exc


PLANNER = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=RETRO_ITERATIONS,
    expansion_topk=EXPANSION_TOPK,
)


def result_is_solved(result) -> bool:
    """Interpret results using the behavior of the local Retro* API."""
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


# ========================== 4. Candidate preparation ========================
def read_smiles(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return [line.strip().split()[0] for line in handle if line.strip()]


def digest(smiles_list: list[str]) -> str:
    payload = ("\n".join(smiles_list) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_sample(
    key: str,
    config: dict[str, str],
    output_dir: Path,
) -> tuple[list[str], dict[str, int], Path]:
    source_file = BASE_PATH / "output" / config["input_name"].format(
        seed=EVOLUTION_SEED
    )
    raw = read_smiles(source_file)

    if len(raw) < SAMPLE_SIZE:
        raise RuntimeError(
            f"{key}: only {len(raw)} source entries are available; "
            f"{SAMPLE_SIZE} are required. Do not sample with replacement."
        )

    rng = random.Random(SAMPLING_SEED)
    sampled = rng.sample(raw, SAMPLE_SIZE)
    sample_file = output_dir / (
        f"{key}_seed{EVOLUTION_SEED}_sampled_{SAMPLE_SIZE}.smi"
    )

    sample_text = "\n".join(sampled) + "\n"
    if sample_file.exists():
        existing = sample_file.read_text(encoding="utf-8")
        if existing != sample_text:
            raise RuntimeError(
                f"Existing sample differs from deterministic resampling: {sample_file}. "
                "Move the experiment output directory before changing its source pool."
            )
    else:
        sample_file.write_text(sample_text, encoding="utf-8")

    counts = {
        "source_entries": len(raw),
        "sampled_entries": len(sampled),
    }
    return sampled, counts, source_file


# =============================== 5. Checkpoint ===============================
FIELDNAMES = [
    "input_index",
    "smiles",
    "solved",
    "runtime_seconds",
    "error",
]


def experiment_paths(key: str) -> dict[str, Path]:
    output_dir = OUTPUT_ROOT / key
    prefix = f"{key}_seed{EVOLUTION_SEED}_n{SAMPLE_SIZE}"
    return {
        "dir": output_dir,
        "checkpoint": output_dir / f"{prefix}_retro_checkpoint.csv",
        "results": output_dir / f"{prefix}_retro_results.csv",
        "solved": output_dir / f"{prefix}_retro_solved.smi",
        "unsolved": output_dir / f"{prefix}_retro_unsolved.smi",
        "summary": output_dir / f"{prefix}_retro_summary.csv",
        "metadata": output_dir / f"{prefix}_input_metadata.txt",
    }


def validate_input_identity(
    paths: dict[str, Path],
    source_file: Path,
    sampled: list[str],
) -> None:
    metadata = (
        f"experiment_seed={EVOLUTION_SEED}\n"
        f"sampling_seed={SAMPLING_SEED}\n"
        f"sample_size={SAMPLE_SIZE}\n"
        f"source_file={source_file}\n"
        f"sample_sha256={digest(sampled)}\n"
    )
    metadata_file = paths["metadata"]
    if metadata_file.exists():
        if metadata_file.read_text(encoding="utf-8") != metadata:
            raise RuntimeError(
                "The sampled input differs from the existing checkpoint metadata. "
                f"Move or delete {paths['dir']} before starting a different run."
            )
    else:
        metadata_file.write_text(metadata, encoding="utf-8")


def load_checkpoint(
    checkpoint_file: Path,
    smiles_list: list[str],
) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    if not checkpoint_file.exists():
        return records

    with checkpoint_file.open("r", newline="", encoding="utf-8") as handle:
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


def append_checkpoint(
    checkpoint_file: Path,
    record: dict[str, object],
) -> None:
    write_header = (
        not checkpoint_file.exists() or checkpoint_file.stat().st_size == 0
    )
    with checkpoint_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


# ================================ 6. Export ==================================
def export_results(
    key: str,
    config: dict[str, str],
    paths: dict[str, Path],
    counts: dict[str, int],
    records_by_index: dict[int, dict[str, object]],
) -> None:
    ordered = [records_by_index[index] for index in range(SAMPLE_SIZE)]

    with paths["results"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(ordered)

    solved = [
        str(record["smiles"])
        for record in ordered
        if int(record["solved"]) == 1
    ]
    unsolved = [
        str(record["smiles"])
        for record in ordered
        if int(record["solved"]) == 0
    ]
    paths["solved"].write_text("\n".join(solved) + "\n", encoding="utf-8")
    paths["unsolved"].write_text(
        "\n".join(unsolved) + "\n",
        encoding="utf-8",
    )

    rate = 100.0 * len(solved) / SAMPLE_SIZE
    summary_fields = [
        "experiment",
        "method",
        "task",
        "evolution_seed",
        "sampling_seed",
        "source_entries",
        "evaluated",
        "retro_solved",
        "retro_unsolved",
        "retro_solved_rate_percent",
        "iterations",
        "expansion_topk",
    ]
    with paths["summary"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(
            {
                "experiment": key,
                "method": config["method"],
                "task": config["task"],
                "evolution_seed": EVOLUTION_SEED,
                "sampling_seed": SAMPLING_SEED,
                "source_entries": counts["source_entries"],
                "evaluated": SAMPLE_SIZE,
                "retro_solved": len(solved),
                "retro_unsolved": len(unsolved),
                "retro_solved_rate_percent": f"{rate:.4f}",
                "iterations": RETRO_ITERATIONS,
                "expansion_topk": EXPANSION_TOPK,
            }
        )

    print("\n" + "=" * 76)
    print(f"{config['method']} | {config['task']} | seed {EVOLUTION_SEED}")
    print(f"Source entries               : {counts['source_entries']}")
    print(f"Fixed sampled input          : {SAMPLE_SIZE}")
    print(f"Retro*-solved                : {len(solved)}")
    print(f"Retro*-unsolved              : {len(unsolved)}")
    print(f"Retro*-solved rate           : {rate:.2f}%")
    print(f"Perfect-molecule file        : {paths['solved']}")
    print(f"Summary                      : {paths['summary']}")
    print("=" * 76)


# ============================== 7. Experiment ================================
def run_experiment(key: str) -> None:
    config = EXPERIMENTS[key]
    paths = experiment_paths(key)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    sampled, counts, source_file = prepare_sample(key, config, paths["dir"])
    validate_input_identity(paths, source_file, sampled)

    print("\n" + "#" * 76)
    print(f"Experiment : {key}")
    print(f"Method     : {config['method']}")
    print(f"Task       : {config['task']}")
    print(f"Source     : {source_file}")
    print(f"Seed       : {EVOLUTION_SEED}")
    print(f"Pool       : {counts['source_entries']} source entries")
    print(f"Sample     : {SAMPLE_SIZE} candidates (sampling seed {SAMPLING_SEED})")
    print("#" * 76)

    if PREPARE_SAMPLES_ONLY:
        print("Sample preparation completed; Retro* was not started.")
        return

    records_by_index = load_checkpoint(paths["checkpoint"], sampled)
    pending = [
        (index, smiles)
        for index, smiles in enumerate(sampled)
        if index not in records_by_index
    ]
    print(
        f"Recorded {len(records_by_index)}/{SAMPLE_SIZE}; "
        f"pending {len(pending)}."
    )

    started = time.time()
    newly_completed = 0
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:
            futures = {
                executor.submit(plan_single_entry, item): item[0]
                for item in pending
            }
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                append_checkpoint(paths["checkpoint"], record)
                newly_completed += 1

                index = int(record["input_index"])
                if not record["error"]:
                    records_by_index[index] = record
                else:
                    print(
                        f"Entry {index} failed after retries and remains pending: "
                        f"{record['error']}",
                        flush=True,
                    )

                if newly_completed % 10 == 0 or newly_completed == len(pending):
                    solved_so_far = sum(
                        int(row["solved"])
                        for row in records_by_index.values()
                    )
                    elapsed = time.time() - started
                    print(
                        f"Completed {len(records_by_index)}/{SAMPLE_SIZE} | "
                        f"solved {solved_so_far} | elapsed {elapsed:.1f}s",
                        flush=True,
                    )

    if len(records_by_index) != SAMPLE_SIZE:
        raise RuntimeError(
            f"{key}: evaluation incomplete: {len(records_by_index)}/{SAMPLE_SIZE} "
            "valid outcomes recorded. Run the script again to retry errors."
        )

    export_results(key, config, paths, counts, records_by_index)


def main() -> None:
    enabled = [key for key, enabled in RUN_SWITCHES.items() if enabled]
    if not enabled:
        raise RuntimeError("No experiment is enabled in RUN_SWITCHES")

    unknown = [key for key in enabled if key not in EXPERIMENTS]
    if unknown:
        raise KeyError(f"Unknown experiment switch(es): {unknown}")

    print(f"Enabled experiments: {', '.join(enabled)}")
    for key in enabled:
        run_experiment(key)


if __name__ == "__main__":
    main()
