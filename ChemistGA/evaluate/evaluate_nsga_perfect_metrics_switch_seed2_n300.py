#!/usr/bin/env python3
"""Evaluate Retro*-solved molecules from the fixed seed-2 NSGA comparison.

The switch names and output paths match run_retro_nsga_switch_seed2_n300.py.
Novelty, diversity, and Bemis-Murcko scaffold count are calculated only for
Retro*-solved entries. Solved entries are retained without deduplication.
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ============================ 1. Switch buttons ============================
# Use the same experiment switch as the corresponding Retro* run.
RUN_SWITCHES = {
    "nsga2_3d": False,
    "nsga3_3d": True,
    "nsga2_4d": False,
    "nsga3_4d": False,
    "nsga2_6d": False,
    "nsga3_6d": False,
}

# ============================== 2. Parameters ==============================
BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
EVOLUTION_SEED = 2
EXPECTED_TESTED = 300

FP_RADIUS = 3
FP_BITS = 2048
NOVELTY_THRESHOLD = 0.4

OUTPUT_ROOT = (
    BASE_PATH
    / "output"
    / f"retro_recorded_nsga_seed{EVOLUTION_SEED}_n{EXPECTED_TESTED}"
)
MASTER_OUTPUT = OUTPUT_ROOT / (
    f"nsga_seed{EVOLUTION_SEED}_n{EXPECTED_TESTED}_perfect_metrics.csv"
)

EXPERIMENTS = {
    "nsga2_3d": {
        "method": "NSGA-II",
        "task": "DRD2 three-objective",
        "reference": "drd_succ_250.csv",
    },
    "nsga3_3d": {
        "method": "NSGA-III",
        "task": "DRD2 three-objective",
        "reference": "drd_succ_250.csv",
    },
    "nsga2_4d": {
        "method": "NSGA-II",
        "task": "JNK3/GSK3beta four-objective",
        "reference": "jnk_gsk.csv",
    },
    "nsga3_4d": {
        "method": "NSGA-III",
        "task": "JNK3/GSK3beta four-objective",
        "reference": "jnk_gsk.csv",
    },
    "nsga2_6d": {
        "method": "NSGA-II",
        "task": "JNK3/GSK3beta six-objective",
        "reference": "jnk_gsk.csv",
    },
    "nsga3_6d": {
        "method": "NSGA-III",
        "task": "JNK3/GSK3beta six-objective",
        "reference": "jnk_gsk.csv",
    },
}


# =========================== 3. Molecular utilities ========================
def parse_molecule(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
    return canonical, mol


def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
    )


def experiment_paths(key: str) -> dict[str, Path]:
    output_dir = OUTPUT_ROOT / key
    prefix = f"{key}_seed{EVOLUTION_SEED}_n{EXPECTED_TESTED}"
    return {
        "dir": output_dir,
        "results": output_dir / f"{prefix}_retro_results.csv",
        "solved": output_dir / f"{prefix}_retro_solved.smi",
        "metrics": output_dir / f"{prefix}_perfect_metrics.csv",
    }


# ============================== 4. Input loading ============================
def load_reference_fingerprints(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")

    frame = pd.read_csv(path, header=None)
    seen: set[str] = set()
    fingerprints = []

    for smiles in frame.iloc[:, 0].dropna():
        canonical, mol = parse_molecule(str(smiles))
        if mol is not None and canonical not in seen:
            seen.add(canonical)
            fingerprints.append(fingerprint(mol))

    if not fingerprints:
        raise RuntimeError(f"No valid reference molecules were loaded from {path}")
    return fingerprints, len(seen)


def load_retro_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Retro* result file not found: {path}")

    frame = pd.read_csv(path)
    if "smiles" not in frame.columns and "SMILES" in frame.columns:
        frame = frame.rename(columns={"SMILES": "smiles"})

    required = {"input_index", "smiles", "solved", "error"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Retro* result file is missing columns: {sorted(missing)}"
        )

    frame = frame.drop_duplicates(subset=["input_index"], keep="last")
    frame = frame.sort_values("input_index").reset_index(drop=True)

    if len(frame) != EXPECTED_TESTED:
        raise RuntimeError(
            f"Expected {EXPECTED_TESTED} Retro* records, but found {len(frame)}"
        )

    expected_indices = list(range(EXPECTED_TESTED))
    if frame["input_index"].astype(int).tolist() != expected_indices:
        raise RuntimeError("Retro* input indices are incomplete or non-contiguous")

    errors = frame["error"].fillna("").astype(str).str.strip()
    if errors.str.len().gt(0).any():
        raise RuntimeError(
            f"{int(errors.str.len().gt(0).sum())} Retro* records contain planner "
            "errors. Rerun the Retro* script before metric evaluation."
        )

    frame["solved"] = frame["solved"].astype(int)
    if not frame["solved"].isin([0, 1]).all():
        raise RuntimeError("The solved column must contain only 0 or 1")
    return frame


def load_solved_molecules(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Retro* solved SMI file not found: {path}")

    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        smiles_list = [
            line.strip().split()[0]
            for line in handle
            if line.strip()
        ]

    molecules = []
    canonical_smiles = []
    for smiles in smiles_list:
        canonical, mol = parse_molecule(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid Retro*-solved SMILES: {smiles}")
        molecules.append(mol)
        canonical_smiles.append(canonical)

    duplicate_count = len(canonical_smiles) - len(set(canonical_smiles))
    return molecules, duplicate_count, len(set(canonical_smiles))


# ============================ 5. Evaluation metrics =========================
def calculate_novelty(predicted_fps, reference_fps) -> float:
    if not predicted_fps:
        return 0.0

    novel = 0
    for fp in predicted_fps:
        maximum_similarity = max(
            DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        )
        if maximum_similarity < NOVELTY_THRESHOLD:
            novel += 1
    return 100.0 * novel / len(predicted_fps)


def calculate_diversity(predicted_fps) -> float:
    count = len(predicted_fps)
    if count < 2:
        return 0.0

    similarity_sum = 0.0
    for index, fp in enumerate(predicted_fps):
        similarity_sum += sum(
            DataStructs.BulkTanimotoSimilarity(fp, predicted_fps[:index])
        )
    pair_count = count * (count - 1) / 2
    return 1.0 - similarity_sum / pair_count


def count_scaffolds(molecules) -> int:
    scaffolds = set()
    for mol in molecules:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        if scaffold:
            scaffolds.add(scaffold)
    return len(scaffolds)


# ================================ 6. Evaluate ================================
def evaluate_experiment(key: str) -> dict[str, object]:
    config = EXPERIMENTS[key]
    paths = experiment_paths(key)
    reference_file = BASE_PATH / "data" / "inh" / config["reference"]

    reference_fps, reference_count = load_reference_fingerprints(reference_file)
    retro_results = load_retro_results(paths["results"])
    molecules, duplicate_count, unique_count = load_solved_molecules(
        paths["solved"]
    )

    solved_csv = int(retro_results["solved"].sum())
    solved_smi = len(molecules)
    if solved_csv != solved_smi:
        raise RuntimeError(
            f"{key}: solved SMI has {solved_smi} entries, but Retro* CSV "
            f"records {solved_csv} solved entries"
        )
    if solved_smi == 0:
        raise RuntimeError(f"{key}: no Retro*-solved molecules are available")

    predicted_fps = [fingerprint(mol) for mol in molecules]
    metrics = {
        "experiment": key,
        "method": config["method"],
        "task": config["task"],
        "seed": EVOLUTION_SEED,
        "reference_molecules": reference_count,
        "evaluated_entries": EXPECTED_TESTED,
        "retro_solved_entries": solved_smi,
        "retro_unsolved_entries": EXPECTED_TESTED - solved_smi,
        "synthesizability_percent": 100.0 * solved_smi / EXPECTED_TESTED,
        "canonical_unique_solved_information_only": unique_count,
        "canonical_duplicate_solved_entries_retained": duplicate_count,
        "novelty_percent_solved_entries": calculate_novelty(
            predicted_fps,
            reference_fps,
        ),
        "diversity_solved_entries": calculate_diversity(predicted_fps),
        "unique_bemis_murcko_scaffolds": count_scaffolds(molecules),
        "fingerprint_radius": FP_RADIUS,
        "fingerprint_bits": FP_BITS,
        "novelty_similarity_threshold": NOVELTY_THRESHOLD,
    }

    pd.DataFrame([metrics]).to_csv(paths["metrics"], index=False)

    print("\n" + "=" * 76)
    print(f"{config['method']} | {config['task']} | seed {EVOLUTION_SEED}")
    print(f"Evaluated input entries          : {EXPECTED_TESTED}")
    print(f"Retro*-solved entries            : {solved_smi}")
    print(f"Synthesizability                 : {metrics['synthesizability_percent']:.2f}%")
    print(f"Canonical duplicates retained    : {duplicate_count}")
    print(f"Canonical unique solved entries  : {unique_count}")
    print(f"Novelty of solved entries        : {metrics['novelty_percent_solved_entries']:.2f}%")
    print(f"Diversity of solved entries      : {metrics['diversity_solved_entries']:.4f}")
    print(f"Unique Bemis-Murcko scaffolds    : {metrics['unique_bemis_murcko_scaffolds']}")
    print(f"Saved                            : {paths['metrics']}")
    print("=" * 76)
    return metrics


def update_master_output(new_rows: list[dict[str, object]]) -> None:
    new_frame = pd.DataFrame(new_rows)
    if MASTER_OUTPUT.exists():
        existing = pd.read_csv(MASTER_OUTPUT)
        existing = existing[
            ~existing["experiment"].isin(new_frame["experiment"])
        ]
        combined = pd.concat([existing, new_frame], ignore_index=True)
    else:
        combined = new_frame

    order = list(EXPERIMENTS)
    combined["_order"] = combined["experiment"].map(
        {key: index for index, key in enumerate(order)}
    )
    combined = combined.sort_values("_order").drop(columns="_order")
    combined.to_csv(MASTER_OUTPUT, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\nUpdated combined metrics: {MASTER_OUTPUT}")


def main() -> None:
    enabled = [key for key, enabled in RUN_SWITCHES.items() if enabled]
    if not enabled:
        raise RuntimeError("No experiment is enabled in RUN_SWITCHES")

    unknown = [key for key in enabled if key not in EXPERIMENTS]
    if unknown:
        raise KeyError(f"Unknown experiment switch(es): {unknown}")

    rows = [evaluate_experiment(key) for key in enabled]
    update_master_output(rows)


if __name__ == "__main__":
    main()
