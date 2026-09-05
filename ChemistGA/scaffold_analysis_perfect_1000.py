#!/usr/bin/env python3
"""Scaffold-level novelty analysis for Retro*-solved perfect molecules.

The workflow follows the Scaffold Analysis section of ChemistGA:
1. randomly sample 1,000 perfect molecules from task 2;
2. extract unique Bemis-Murcko scaffolds;
3. compare every generated scaffold with the real-active scaffold set;
4. record the nearest-reference (maximum) Tanimoto similarity;
5. report counts at <=0.1, <=0.2, <=0.3 and <=0.4 and the mean similarity.

Change RUN_CONTINUOUS and RUN_DISCRETE in the configuration block, then
run this file directly from PyCharm. No command-line arguments are required.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

# ============================================================================
# RUN SWITCHES: only change this block, then click Run in PyCharm.
# ============================================================================
RUN_CONTINUOUS = False  # Continuous: True = run; False = skip
RUN_DISCRETE = True     # Discrete:   True = run; False = skip

CONTINUOUS_SEED = 1
DISCRETE_SEED = 2

SAMPLE_SIZE = 1000
SAMPLE_SEED = 2026      # Change this integer for another reproducible sample.
MORGAN_RADIUS = 3
FINGERPRINT_BITS = 2048

BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
REFERENCE_FILE = BASE_PATH / "data" / "inh" / "jnk_gsk.csv"
OUTPUT_DIR = BASE_PATH / "output" / "scaffold_analysis_4d"

THRESHOLDS = (0.1, 0.2, 0.3, 0.4)


def selected_jobs() -> list[SimpleNamespace]:
    jobs: list[SimpleNamespace] = []
    retro_dir = BASE_PATH / "output" / "retro_recorded_4d"

    if RUN_CONTINUOUS:
        jobs.append(
            SimpleNamespace(
                perfect_file=retro_dir
                / f"continuous_retro_solved_seed{CONTINUOUS_SEED}.smi",
                reference_file=REFERENCE_FILE,
                output_dir=OUTPUT_DIR,
                label=f"PT-MOGA-continuous-seed{CONTINUOUS_SEED}",
                sample_size=SAMPLE_SIZE,
                sample_seed=SAMPLE_SEED,
                radius=MORGAN_RADIUS,
                n_bits=FINGERPRINT_BITS,
            )
        )

    if RUN_DISCRETE:
        jobs.append(
            SimpleNamespace(
                perfect_file=retro_dir
                / f"discrete_retro_solved_seed{DISCRETE_SEED}.smi",
                reference_file=REFERENCE_FILE,
                output_dir=OUTPUT_DIR,
                label=f"PT-MOGA-discrete-seed{DISCRETE_SEED}",
                sample_size=SAMPLE_SIZE,
                sample_seed=SAMPLE_SEED,
                radius=MORGAN_RADIUS,
                n_bits=FINGERPRINT_BITS,
            )
        )

    if not jobs:
        raise RuntimeError(
            "No task selected. Set RUN_CONTINUOUS or RUN_DISCRETE to True."
        )
    return jobs


def read_smiles(path: Path) -> list[str]:
    """Read the first field from SMI, CSV or TSV-like files."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    values: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            # Files in this project store SMILES in the first comma/tab/space field.
            first = text.split(",", 1)[0].split("\t", 1)[0].split()[0]
            values.append(first.strip().strip('"').strip("'"))
    return values


def canonicalize_unique(smiles: list[str]) -> tuple[list[str], int]:
    unique: dict[str, None] = {}
    invalid = 0
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        unique.setdefault(canonical, None)
    return list(unique), invalid


def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def extract_unique_scaffolds(smiles: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    scaffolds: dict[str, None] = {}
    molecule_records: list[dict[str, str]] = []
    for value in smiles:
        scaffold = murcko_scaffold_smiles(value)
        molecule_records.append(
            {"smiles": value, "murcko_scaffold": scaffold or ""}
        )
        if scaffold:
            scaffolds.setdefault(scaffold, None)
    return list(scaffolds), molecule_records


def fingerprints(scaffolds: list[str], radius: int, n_bits: int):
    fps = []
    valid_scaffolds = []
    for scaffold in scaffolds:
        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        valid_scaffolds.append(scaffold)
    return valid_scaffolds, fps


def nearest_reference_similarities(
    generated_scaffolds: list[str],
    generated_fps,
    reference_scaffolds: list[str],
    reference_fps,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scaffold, fp in zip(generated_scaffolds, generated_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        nearest_index = int(np.argmax(similarities))
        records.append(
            {
                "generated_scaffold": scaffold,
                "nearest_reference_scaffold": reference_scaffolds[nearest_index],
                "max_tanimoto_similarity": float(similarities[nearest_index]),
            }
        )
    return records


def safe_label(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def analyze_job(args: SimpleNamespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = safe_label(args.label)

    perfect_raw = read_smiles(args.perfect_file)
    perfect_unique, perfect_invalid = canonicalize_unique(perfect_raw)
    if len(perfect_unique) < args.sample_size:
        raise RuntimeError(
            f"Only {len(perfect_unique)} canonical-unique perfect molecules are available; "
            f"cannot sample {args.sample_size} without replacement."
        )

    rng = random.Random(args.sample_seed)
    sampled_smiles = rng.sample(perfect_unique, args.sample_size)

    reference_raw = read_smiles(args.reference_file)
    reference_unique, reference_invalid = canonicalize_unique(reference_raw)
    if not reference_unique:
        raise RuntimeError("No valid reference-active molecules were loaded.")

    generated_scaffolds, molecule_records = extract_unique_scaffolds(sampled_smiles)
    reference_scaffolds, _ = extract_unique_scaffolds(reference_unique)
    if not generated_scaffolds:
        raise RuntimeError("No non-empty Murcko scaffolds were extracted from the sample.")
    if not reference_scaffolds:
        raise RuntimeError("No non-empty Murcko scaffolds were extracted from the reference set.")

    generated_scaffolds, generated_fps = fingerprints(
        generated_scaffolds, args.radius, args.n_bits
    )
    reference_scaffolds, reference_fps = fingerprints(
        reference_scaffolds, args.radius, args.n_bits
    )
    scaffold_records = nearest_reference_similarities(
        generated_scaffolds,
        generated_fps,
        reference_scaffolds,
        reference_fps,
    )

    similarities = np.asarray(
        [row["max_tanimoto_similarity"] for row in scaffold_records], dtype=float
    )
    summary = {
        "label": args.label,
        "perfect_input_entries": len(perfect_raw),
        "perfect_invalid_entries": perfect_invalid,
        "perfect_canonical_unique": len(perfect_unique),
        "sample_size": len(sampled_smiles),
        "sample_seed": args.sample_seed,
        "sampled_unique_murcko_scaffolds": len(generated_scaffolds),
        "reference_valid_unique_molecules": len(reference_unique),
        "reference_invalid_entries": reference_invalid,
        "reference_unique_murcko_scaffolds": len(reference_scaffolds),
        "morgan_radius": args.radius,
        "fingerprint_bits": args.n_bits,
        "count_similarity_le_0.1": int(np.sum(similarities <= 0.1)),
        "count_similarity_le_0.2": int(np.sum(similarities <= 0.2)),
        "count_similarity_le_0.3": int(np.sum(similarities <= 0.3)),
        "count_similarity_le_0.4": int(np.sum(similarities <= 0.4)),
        "mean_max_similarity": float(np.mean(similarities)),
        "median_max_similarity": float(np.median(similarities)),
        "std_max_similarity": float(np.std(similarities)),
    }

    sampled_path = args.output_dir / f"{label}_sampled_perfect_{args.sample_size}.smi"
    molecule_path = args.output_dir / f"{label}_sampled_molecules_scaffolds.csv"
    scaffold_path = args.output_dir / f"{label}_unique_scaffold_similarity.csv"
    summary_path = args.output_dir / f"{label}_scaffold_summary.csv"
    json_path = args.output_dir / f"{label}_scaffold_summary.json"

    sampled_path.write_text("\n".join(sampled_smiles) + "\n", encoding="utf-8")
    pd.DataFrame(molecule_records).to_csv(molecule_path, index=False)
    pd.DataFrame(scaffold_records).sort_values(
        "max_tanimoto_similarity"
    ).to_csv(scaffold_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"Scaffold analysis: {args.label}")
    print("=" * 68)
    print(f"Perfect input entries             : {len(perfect_raw)}")
    print(f"Canonical-unique perfect molecules: {len(perfect_unique)}")
    print(f"Random sample size                : {len(sampled_smiles)}")
    print(f"Unique generated Murcko scaffolds : {len(generated_scaffolds)}")
    print(f"Unique reference Murcko scaffolds : {len(reference_scaffolds)}")
    for threshold in THRESHOLDS:
        count = int(np.sum(similarities <= threshold))
        print(f"Scaffolds with similarity <= {threshold:.1f}: {count}")
    print(f"Mean maximum similarity           : {np.mean(similarities):.3f}")
    print(f"Median maximum similarity         : {np.median(similarities):.3f}")
    print(f"SD of maximum similarity          : {np.std(similarities):.3f}")
    print("-" * 68)
    print(f"Saved sample  : {sampled_path}")
    print(f"Saved details : {scaffold_path}")
    print(f"Saved summary : {summary_path}")


def main() -> None:
    jobs = selected_jobs()
    print(f"Selected scaffold-analysis jobs: {len(jobs)}")
    for index, job in enumerate(jobs, start=1):
        print(f"\nRunning job {index}/{len(jobs)}: {job.label}")
        analyze_job(job)


if __name__ == "__main__":
    main()
