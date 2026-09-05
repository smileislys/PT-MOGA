#!/usr/bin/env python3
"""Evaluate Retro*-solved GB-GA ablation molecules (seed 0, n=300).

Novelty, diversity, and Bemis-Murcko scaffold count are computed only for
Retro*-solved entries. Solved entries are retained as sampled, including any
canonical duplicate entries, to keep the metric denominator aligned with the
Retro* experiment. Duplicate counts are reported for transparency.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ================================ Controls ================================
RUN_ABLATION_GBGA = True

# =============================== Parameters ===============================
BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
EVOLUTION_SEED = 0
EXPECTED_TESTED = 300
EXPERIMENT = "ablation_gbga"
METHOD = "GB-GA crossover ablation"
TASK = "JNK3/GSK3beta six-objective"

FP_RADIUS = 3
FP_BITS = 2048
NOVELTY_THRESHOLD = 0.4

REFERENCE_FILE = BASE_PATH / "data" / "inh" / "jnk_gsk.csv"
OUTPUT_DIR = (
    BASE_PATH
    / "output"
    / f"retro_recorded_ablation_gbga_seed{EVOLUTION_SEED}_n{EXPECTED_TESTED}"
)
PREFIX = f"{EXPERIMENT}_seed{EVOLUTION_SEED}_n{EXPECTED_TESTED}"
RESULTS_FILE = OUTPUT_DIR / f"{PREFIX}_retro_results.csv"
SOLVED_FILE = OUTPUT_DIR / f"{PREFIX}_retro_solved.smi"
METRICS_FILE = OUTPUT_DIR / f"{PREFIX}_perfect_metrics.csv"


def parse_molecule(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, mol


def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)


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
        raise FileNotFoundError(f"Retro* results not found: {path}")

    frame = pd.read_csv(path)
    required = {"input_index", "smiles", "solved", "error"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Retro* result file is missing columns: {sorted(missing)}")

    frame = frame.drop_duplicates(subset=["input_index"], keep="last")
    frame = frame.sort_values("input_index").reset_index(drop=True)
    if len(frame) != EXPECTED_TESTED:
        raise RuntimeError(
            f"Expected {EXPECTED_TESTED} Retro* records, but found {len(frame)}"
        )
    if frame["input_index"].astype(int).tolist() != list(range(EXPECTED_TESTED)):
        raise RuntimeError("Retro* input indices are incomplete or non-contiguous")

    errors = frame["error"].fillna("").astype(str).str.strip()
    if errors.str.len().gt(0).any():
        raise RuntimeError(
            f"{int(errors.str.len().gt(0).sum())} records contain planner errors. "
            "Rerun the Retro* script before metric evaluation."
        )

    frame["solved"] = frame["solved"].astype(int)
    if not frame["solved"].isin([0, 1]).all():
        raise RuntimeError("The solved column must contain only 0 or 1")
    return frame


def load_solved_molecules(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Solved molecule file not found: {path}")

    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        smiles_list = [line.strip().split()[0] for line in handle if line.strip()]

    molecules = []
    canonical_smiles = []
    for smiles in smiles_list:
        canonical, mol = parse_molecule(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid Retro*-solved SMILES: {smiles}")
        molecules.append(mol)
        canonical_smiles.append(canonical)

    unique_count = len(set(canonical_smiles))
    duplicate_count = len(canonical_smiles) - unique_count
    return molecules, duplicate_count, unique_count


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
            mol=mol, includeChirality=False
        )
        if scaffold:
            scaffolds.add(scaffold)
    return len(scaffolds)


def main() -> None:
    if not RUN_ABLATION_GBGA:
        print("RUN_ABLATION_GBGA is False; nothing was evaluated.")
        return

    reference_fps, reference_count = load_reference_fingerprints(REFERENCE_FILE)
    retro_results = load_retro_results(RESULTS_FILE)
    molecules, duplicate_count, unique_count = load_solved_molecules(SOLVED_FILE)

    solved_from_csv = int(retro_results["solved"].sum())
    if solved_from_csv != len(molecules):
        raise RuntimeError(
            f"Solved SMI contains {len(molecules)} entries, but results CSV "
            f"records {solved_from_csv}."
        )
    if not molecules:
        raise RuntimeError("No Retro*-solved molecules are available")

    predicted_fps = [fingerprint(mol) for mol in molecules]
    metrics = {
        "experiment": EXPERIMENT,
        "method": METHOD,
        "task": TASK,
        "seed": EVOLUTION_SEED,
        "reference_molecules": reference_count,
        "evaluated_entries": EXPECTED_TESTED,
        "retro_solved_entries": len(molecules),
        "retro_unsolved_entries": EXPECTED_TESTED - len(molecules),
        "synthesizability_percent": 100.0 * len(molecules) / EXPECTED_TESTED,
        "canonical_unique_solved_information_only": unique_count,
        "canonical_duplicate_solved_entries_retained": duplicate_count,
        "novelty_percent_solved_entries": calculate_novelty(
            predicted_fps, reference_fps
        ),
        "diversity_solved_entries": calculate_diversity(predicted_fps),
        "unique_bemis_murcko_scaffolds": count_scaffolds(molecules),
        "fingerprint_radius": FP_RADIUS,
        "fingerprint_bits": FP_BITS,
        "novelty_similarity_threshold": NOVELTY_THRESHOLD,
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_FILE, index=False, quoting=csv.QUOTE_MINIMAL
    )

    print("\n" + "=" * 76)
    print(f"{METHOD} | {TASK} | seed {EVOLUTION_SEED}")
    print(f"Evaluated input entries          : {EXPECTED_TESTED}")
    print(f"Retro*-solved entries            : {len(molecules)}")
    print(f"Synthesizability                 : {metrics['synthesizability_percent']:.2f}%")
    print(f"Canonical duplicates retained    : {duplicate_count}")
    print(f"Canonical unique solved entries  : {unique_count}")
    print(f"Novelty of solved entries        : {metrics['novelty_percent_solved_entries']:.2f}%")
    print(f"Diversity of solved entries      : {metrics['diversity_solved_entries']:.4f}")
    print(f"Unique Bemis-Murcko scaffolds    : {metrics['unique_bemis_murcko_scaffolds']}")
    print(f"Saved                            : {METRICS_FILE}")
    print("=" * 76)


if __name__ == "__main__":
    main()
