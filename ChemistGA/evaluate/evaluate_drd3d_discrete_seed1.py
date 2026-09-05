#!/usr/bin/env python3
"""Evaluate DRD2 three-objective discrete Retro*-solved molecules, seed 1.

This SMI-only evaluator reads the recorded Retro*-solved molecules directly.
Novelty, diversity, and Bemis-Murcko scaffold count are calculated over all
solved entries without deduplication. Synthesizability uses the fixed 5,000
tested candidate entries as its denominator.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ============================== Configuration =============================
BASE_PATH = "/home/liuyansong/ChemistGA-master/ChemistGA-master"
SEED = 1
EXPECTED_TESTED = 5000

FP_RADIUS = 3
FP_BITS = 2048
NOVELTY_THRESHOLD = 0.4

REFERENCE_FILE = os.path.join(
    BASE_PATH,
    "data",
    "inh",
    "drd_succ_250.csv",
)

SOLVED_SMI_FILE = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_3d",
    f"drd_discrete_retro_solved_seed{SEED}.smi",
)

OUTPUT_FILE = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_3d",
    f"drd_discrete_perfect_metrics_seed{SEED}.csv",
)


def parse_molecule(smiles):
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


def load_reference_fingerprints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference file not found: {path}")

    frame = pd.read_csv(path, header=None)
    seen = set()
    fingerprints = []

    for smiles in frame.iloc[:, 0].dropna():
        canonical, mol = parse_molecule(smiles)
        if mol is not None and canonical not in seen:
            seen.add(canonical)
            fingerprints.append(fingerprint(mol))

    if not fingerprints:
        raise RuntimeError("No valid DRD2 reference molecules were loaded.")
    return fingerprints, len(seen)


def load_solved_molecules(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Retro* solved SMI file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
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
            raise RuntimeError(f"Invalid solved SMILES encountered: {smiles}")
        molecules.append(mol)
        canonical_smiles.append(canonical)

    unique_count = len(set(canonical_smiles))
    duplicate_count = len(canonical_smiles) - unique_count
    return molecules, duplicate_count, unique_count


def calculate_novelty(predicted_fps, reference_fps):
    if not predicted_fps:
        return 0.0

    novel_count = 0
    for fp in predicted_fps:
        maximum_similarity = max(
            DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        )
        if maximum_similarity < NOVELTY_THRESHOLD:
            novel_count += 1
    return 100.0 * novel_count / len(predicted_fps)


def calculate_diversity(predicted_fps):
    molecule_count = len(predicted_fps)
    if molecule_count < 2:
        return 0.0

    similarity_sum = 0.0
    for index, fp in enumerate(predicted_fps):
        similarity_sum += sum(
            DataStructs.BulkTanimotoSimilarity(fp, predicted_fps[:index])
        )
    pair_count = molecule_count * (molecule_count - 1) / 2
    return 1.0 - similarity_sum / pair_count


def count_scaffolds(molecules):
    scaffolds = set()
    for mol in molecules:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        if scaffold:
            scaffolds.add(scaffold)
    return len(scaffolds)


def main():
    reference_fps, reference_count = load_reference_fingerprints(
        REFERENCE_FILE
    )
    perfect_molecules, duplicate_count, unique_count = load_solved_molecules(
        SOLVED_SMI_FILE
    )

    if not perfect_molecules:
        raise RuntimeError("The solved SMI file contains no valid molecules.")

    solved_count = len(perfect_molecules)
    if solved_count > EXPECTED_TESTED:
        raise RuntimeError(
            f"Solved file contains {solved_count} entries, exceeding the fixed "
            f"tested total of {EXPECTED_TESTED}."
        )

    predicted_fps = [fingerprint(mol) for mol in perfect_molecules]
    metrics = {
        "task": "drd_discrete",
        "seed": SEED,
        "reference_molecules": reference_count,
        "tested_success_molecule_entries": EXPECTED_TESTED,
        "retro_solved_perfect_entries": solved_count,
        "retro_unsolved_or_unresolved_entries": EXPECTED_TESTED - solved_count,
        "synthesizability_percent": 100.0 * solved_count / EXPECTED_TESTED,
        "canonical_unique_perfect_molecules_information_only": unique_count,
        "canonical_duplicate_entries_not_removed": duplicate_count,
        "novelty_percent_all_perfect_entries": calculate_novelty(
            predicted_fps,
            reference_fps,
        ),
        "diversity_all_perfect_entries": calculate_diversity(predicted_fps),
        "unique_bemis_murcko_scaffolds": count_scaffolds(perfect_molecules),
        "fingerprint_radius": FP_RADIUS,
        "fingerprint_bits": FP_BITS,
        "novelty_similarity_threshold": NOVELTY_THRESHOLD,
    }

    pd.DataFrame([metrics]).to_csv(OUTPUT_FILE, index=False)

    print("=" * 72)
    print(f"PT-MOGA DRD2 three-objective discrete evaluation | seed {SEED}")
    print("=" * 72)
    print(
        "Synthesizability: %d/%d (%.2f%%)"
        % (
            solved_count,
            EXPECTED_TESTED,
            metrics["synthesizability_percent"],
        )
    )
    print(f"Solved entries evaluated without deduplication: {solved_count}")
    print(
        "Canonical duplicates retained: %d (canonical unique molecules: %d)"
        % (duplicate_count, unique_count)
    )
    print(
        "Novelty (all perfect entries): %.2f%%"
        % metrics["novelty_percent_all_perfect_entries"]
    )
    print(
        "Diversity (all perfect entries): %.4f"
        % metrics["diversity_all_perfect_entries"]
    )
    print(
        "Unique Bemis-Murcko scaffolds: %d"
        % metrics["unique_bemis_murcko_scaffolds"]
    )
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
