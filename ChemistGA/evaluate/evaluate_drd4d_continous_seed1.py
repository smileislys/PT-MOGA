#!/usr/bin/env python3
"""Evaluate four-objective continuous Retro*-solved molecules, seed 1.

The 5,000 input molecules have already satisfied the JNK3, GSK3β,
QED and SA criteria. Synthesizability is calculated over all tested
entries. Novelty, diversity and scaffold count are calculated on the
Retro*-solved perfect-molecule subset without removing duplicates.
"""

import os
import warnings

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ==================== Configuration ====================

BASE_PATH = "/home/liuyansong/ChemistGA-master/ChemistGA-master"

TASK_PREFIX = "continuous"
SEED = 1
EXPECTED_TESTED = 5000

FP_RADIUS = 3
FP_BITS = 2048
NOVELTY_THRESHOLD = 0.4

# JNK3/GSK3β dual-target reference molecules
REFERENCE_FILE = os.path.join(
    BASE_PATH,
    "data",
    "inh",
    "jnk_gsk.csv",
)

RETRO_RESULT_FILE = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_4d",
    f"{TASK_PREFIX}_retro_results_seed{SEED}.csv",
)

SOLVED_SMI_FILE = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_4d",
    f"{TASK_PREFIX}_retro_solved_seed{SEED}.smi",
)

OUTPUT_FILE = os.path.join(
    BASE_PATH,
    "output",
    "retro_recorded_4d",
    f"{TASK_PREFIX}_perfect_metrics_seed{SEED}.csv",
)


# ==================== Molecular utilities ====================

def parse_molecule(smiles):
    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None, None

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )

    return canonical_smiles, mol


def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        FP_RADIUS,
        nBits=FP_BITS,
    )


# ==================== Input loading ====================

def load_reference_fingerprints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference file not found: {path}"
        )

    frame = pd.read_csv(
        path,
        header=None,
    )

    seen = set()
    reference_fps = []

    for smiles in frame.iloc[:, 0].dropna():
        canonical_smiles, mol = parse_molecule(
            smiles
        )

        if (
            mol is not None
            and canonical_smiles not in seen
        ):
            seen.add(canonical_smiles)
            reference_fps.append(
                fingerprint(mol)
            )

    if not reference_fps:
        raise RuntimeError(
            "No valid JNK3/GSK3β reference molecules were loaded."
        )

    print(
        f"Loaded {len(reference_fps)} canonical-unique "
        f"JNK3/GSK3β reference molecules."
    )

    return reference_fps


def load_retro_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Retro* result file not found: {path}"
        )

    frame = pd.read_csv(path)

    required_columns = {
        "input_index",
        "SMILES",
        "solved",
        "error",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Retro* result file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Keep the most recent result when a failed entry was rerun.
    frame = frame.drop_duplicates(
        subset=["input_index"],
        keep="last",
    )

    frame = frame.sort_values(
        "input_index"
    ).reset_index(drop=True)

    if len(frame) != EXPECTED_TESTED:
        raise RuntimeError(
            f"Expected {EXPECTED_TESTED} Retro* records, "
            f"but found {len(frame)}."
        )

    error_mask = (
        frame["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
        .gt(0)
    )

    if error_mask.any():
        raise RuntimeError(
            f"{int(error_mask.sum())} Retro* records contain "
            f"planner errors. Run the Retro* script again "
            f"before calculating the final metrics."
        )

    frame["solved"] = frame[
        "solved"
    ].astype(int)

    if not frame["solved"].isin([0, 1]).all():
        raise RuntimeError(
            "The solved column must contain only 0 or 1."
        )

    return frame


def load_solved_molecules(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Retro* solved SMI file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        smiles_list = [
            line.strip().split()[0]
            for line in handle
            if line.strip()
        ]

    molecules = []
    canonical_smiles_list = []

    for smiles in smiles_list:
        canonical_smiles, mol = parse_molecule(
            smiles
        )

        if mol is None:
            raise RuntimeError(
                f"Invalid solved SMILES encountered: "
                f"{smiles}"
            )

        # Retain every solved entry without deduplication.
        molecules.append(mol)
        canonical_smiles_list.append(
            canonical_smiles
        )

    duplicate_count = (
        len(canonical_smiles_list)
        - len(set(canonical_smiles_list))
    )

    unique_molecule_count = len(
        set(canonical_smiles_list)
    )

    return (
        molecules,
        duplicate_count,
        unique_molecule_count,
    )


# ==================== Evaluation metrics ====================

def calculate_novelty(
    predicted_fps,
    reference_fps,
):
    if not predicted_fps:
        return 0.0

    novel_count = 0

    for fp in predicted_fps:
        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                fp,
                reference_fps,
            )
        )

        maximum_similarity = max(
            similarities
        )

        if (
            maximum_similarity
            < NOVELTY_THRESHOLD
        ):
            novel_count += 1

    return (
        100.0
        * novel_count
        / len(predicted_fps)
    )


def calculate_diversity(predicted_fps):
    molecule_count = len(
        predicted_fps
    )

    if molecule_count < 2:
        return 0.0

    similarity_sum = 0.0

    for index, fp in enumerate(
        predicted_fps
    ):
        similarities = (
            DataStructs.BulkTanimotoSimilarity(
                fp,
                predicted_fps[:index],
            )
        )

        similarity_sum += sum(
            similarities
        )

    pair_count = (
        molecule_count
        * (molecule_count - 1)
        / 2
    )

    return (
        1.0
        - similarity_sum / pair_count
    )


def count_scaffolds(molecules):
    scaffolds = set()

    for mol in molecules:
        scaffold = (
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
        )

        if scaffold:
            scaffolds.add(scaffold)

    return len(scaffolds)


# ==================== Main evaluation ====================

def main():
    reference_fps = (
        load_reference_fingerprints(
            REFERENCE_FILE
        )
    )

    retro_results = (
        load_retro_results(
            RETRO_RESULT_FILE
        )
    )

    (
        perfect_molecules,
        duplicate_count,
        unique_molecule_count,
    ) = load_solved_molecules(
        SOLVED_SMI_FILE
    )

    if not perfect_molecules:
        raise RuntimeError(
            "The solved SMI file contains no valid molecules."
        )

    tested_count = len(
        retro_results
    )

    solved_count_csv = int(
        retro_results["solved"].sum()
    )

    solved_count_smi = len(
        perfect_molecules
    )

    if solved_count_smi != solved_count_csv:
        raise RuntimeError(
            f"The solved SMI file contains "
            f"{solved_count_smi} entries, but the "
            f"Retro* CSV records {solved_count_csv} "
            f"solved entries."
        )

    predicted_fps = [
        fingerprint(mol)
        for mol in perfect_molecules
    ]

    metrics = {
        "task": "four_objective_continuous",
        "seed": SEED,
        "tested_success_molecule_entries":
            tested_count,
        "retro_solved_perfect_entries":
            solved_count_smi,
        "synthesizability_percent":
            100.0
            * solved_count_smi
            / tested_count,
        "canonical_unique_perfect_molecules_information_only":
            unique_molecule_count,
        "canonical_duplicate_entries_not_removed":
            duplicate_count,
        "novelty_percent_all_perfect_entries":
            calculate_novelty(
                predicted_fps,
                reference_fps,
            ),
        "diversity_all_perfect_entries":
            calculate_diversity(
                predicted_fps
            ),
        "unique_bemis_murcko_scaffolds":
            count_scaffolds(
                perfect_molecules
            ),
    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 72)
    print(
        "PT-MOGA four-objective continuous evaluation | seed 1"
    )
    print("=" * 72)

    print(
        "Synthesizability: %d/%d (%.2f%%)"
        % (
            metrics[
                "retro_solved_perfect_entries"
            ],
            metrics[
                "tested_success_molecule_entries"
            ],
            metrics[
                "synthesizability_percent"
            ],
        )
    )

    print(
        "Solved entries evaluated without "
        "deduplication: %d"
        % metrics[
            "retro_solved_perfect_entries"
        ]
    )

    print(
        "Canonical duplicates retained: %d "
        "(canonical unique molecules: %d)"
        % (
            metrics[
                "canonical_duplicate_entries_not_removed"
            ],
            metrics[
                "canonical_unique_perfect_molecules_information_only"
            ],
        )
    )

    print(
        "Novelty (all perfect entries): %.2f%%"
        % metrics[
            "novelty_percent_all_perfect_entries"
        ]
    )

    print(
        "Diversity (all perfect entries): %.4f"
        % metrics[
            "diversity_all_perfect_entries"
        ]
    )

    print(
        "Unique Bemis-Murcko scaffolds: %d"
        % metrics[
            "unique_bemis_murcko_scaffolds"
        ]
    )

    print(
        "Saved:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()