import os
import sys
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Draw, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


BASE_PATH = Path(__file__).resolve().parents[1]
SCORING_DIR = BASE_PATH / "scoring"
CHEMISTGA_DIR = BASE_PATH / "ChemistGA"
JNK_GSK_DIR = CHEMISTGA_DIR / "high_score" / "high_score_jnk_gsk"
OUTPUT_DIR = BASE_PATH / "output"

INPUT_SMI = OUTPUT_DIR / "nsga3_6d_perfect_set_ALL.smi"
OUT_PREFIX = OUTPUT_DIR / "nsga3_6d_top5_scored_molecules"

NUM_DISPLAY = 5
TOP_POOL_SIZE = int(os.environ.get("TOP_POOL_SIZE", 30))
RANDOM_SEED = int(os.environ.get("FIGURE_RANDOM_SEED", int.from_bytes(os.urandom(4), "little")))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(SCORING_DIR))
sys.path.insert(0, str(CHEMISTGA_DIR))
sys.path.insert(0, str(JNK_GSK_DIR))


def load_scorers():
    """
    Load the same activity/QED/SA scorers used by the original PT-MOGA code.
    The activity model paths in high_score_properties_jnk_gsk.py are relative,
    so scorer initialization is performed under JNK_GSK_DIR.
    """
    old_cwd = os.getcwd()
    os.chdir(JNK_GSK_DIR)
    try:
        from high_score_properties_jnk_gsk import get_scoring_function

        scorers = {
            "jnk3": get_scoring_function("jnk3"),
            "gsk3": get_scoring_function("gsk3"),
            "qed": get_scoring_function("qed"),
            "sa": get_scoring_function("sa"),
        }
    finally:
        os.chdir(old_cwd)
    return scorers


def read_smiles(path):
    smiles = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            smi = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            smiles.append(Chem.MolToSmiles(mol, canonical=True))

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for smi in smiles:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)
    return unique


def score_6d(smiles):
    scorers = load_scorers()

    jnk3_scores = np.asarray(scorers["jnk3"](smiles), dtype=float)
    gsk3_scores = np.asarray(scorers["gsk3"](smiles), dtype=float)
    qed_scores = np.asarray(scorers["qed"](smiles), dtype=float)
    sa_scores = np.asarray(scorers["sa"](smiles), dtype=float)

    records = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        jnk3 = float(jnk3_scores[i])
        gsk3 = float(gsk3_scores[i])
        qed = float(qed_scores[i])
        sa = float(sa_scores[i])

        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        logp = float(Crippen.MolLogP(mol))

        # Keep exactly the same normalization idea as run_nsga3_6d.py.
        sa_norm = max(0.0, (10.0 - sa) / 9.0)
        tpsa_norm = math.exp(-0.5 * ((tpsa - 60.0) / 30.0) ** 2)
        logp_norm = math.exp(-0.5 * ((logp - 3.0) / 1.5) ** 2)

        pass_6d = (
            jnk3 >= 0.5
            and gsk3 >= 0.5
            and qed >= 0.6
            and sa <= 4.0
            and tpsa <= 90.0
            and 1.0 <= logp <= 5.0
        )

        norm_values = np.array([jnk3, gsk3, qed, sa_norm, tpsa_norm, logp_norm], dtype=float)
        six_dim_mean = float(np.mean(norm_values))
        six_dim_min = float(np.min(norm_values))
        six_dim_std = float(np.std(norm_values))

        # This ranking score is only for selecting representative molecules for display.
        # It rewards high average performance, penalizes poor weakest dimension, and
        # avoids molecules that are excellent in one target but weak in another.
        display_score = six_dim_mean + 0.30 * six_dim_min - 0.10 * six_dim_std
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

        records.append(
            {
                "SMILES": smi,
                "JNK3": jnk3,
                "GSK3b": gsk3,
                "QED": qed,
                "SA": sa,
                "SA_norm": sa_norm,
                "TPSA": tpsa,
                "TPSA_norm": tpsa_norm,
                "LogP": logp,
                "LogP_norm": logp_norm,
                "six_dim_mean": six_dim_mean,
                "six_dim_min": six_dim_min,
                "six_dim_std": six_dim_std,
                "display_score": display_score,
                "pass_6d_threshold": pass_6d,
                "Murcko_scaffold": scaffold,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values(
        by=["pass_6d_threshold", "display_score", "six_dim_min", "six_dim_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return df


def make_legend(row):
    return (
        f"JNK3={row.JNK3:.2f}  GSK3\u03b2={row.GSK3b:.2f}\n"
        f"QED={row.QED:.2f}  SA={row.SA:.2f}\n"
        f"TPSA={row.TPSA:.1f}  LogP={row.LogP:.2f}"
    )


def random_best_five(df, n=NUM_DISPLAY, pool_size=TOP_POOL_SIZE):
    """
    Randomly select molecules from the best-scored candidate pool.

    Logic:
    1. Prefer molecules passing all six thresholds.
    2. Define the "best pool" as the top-ranked subset by display_score.
    3. Randomly sample 5 molecules from this high-quality pool.
    4. Prefer different Bemis-Murcko scaffolds to avoid showing near-duplicates.
    """
    if df.empty:
        return df

    source = df[df["pass_6d_threshold"]].copy()
    if len(source) < n:
        print("WARNING: fewer than 5 molecules pass all 6D thresholds; using highest-scored molecules as fallback.")
        source = df.copy()

    source = source.sort_values(
        by=["display_score", "six_dim_min", "six_dim_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    pool = source.head(min(pool_size, len(source))).copy()

    # Shuffle only inside the best pool, so every run differs while quality remains high.
    pool = pool.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    selected_rows = []
    used_scaffolds = set()
    for _, row in pool.iterrows():
        scaffold = row.get("Murcko_scaffold", "")
        if scaffold and scaffold in used_scaffolds:
            continue
        selected_rows.append(row)
        if scaffold:
            used_scaffolds.add(scaffold)
        if len(selected_rows) == n:
            break

    # If scaffold diversity is insufficient, fill remaining slots from the same best pool.
    if len(selected_rows) < n:
        selected_smiles = {row["SMILES"] for row in selected_rows}
        for _, row in pool.iterrows():
            if row["SMILES"] in selected_smiles:
                continue
            selected_rows.append(row)
            selected_smiles.add(row["SMILES"])
            if len(selected_rows) == n:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def draw_top5(top_df):
    mols = [Chem.MolFromSmiles(s) for s in top_df["SMILES"]]
    legends = [make_legend(row) for row in top_df.itertuples()]

    png_path = OUT_PREFIX.with_suffix(".png")
    svg_path = OUT_PREFIX.with_suffix(".svg")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(430, 360),
        legends=legends,
        useSVG=False,
    )
    img.save(png_path)

    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=5,
        subImgSize=(430, 360),
        legends=legends,
        useSVG=True,
    )
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return png_path, svg_path


def main():
    if not INPUT_SMI.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_SMI}")

    smiles = read_smiles(INPUT_SMI)
    print(f"Loaded valid unique molecules: {len(smiles)}")

    df = score_6d(smiles)
    if df.empty:
        raise RuntimeError("No valid molecule could be scored.")

    all_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_all_scores.csv")
    top_csv = OUT_PREFIX.with_name(OUT_PREFIX.name + "_top5.csv")
    top_smi = OUT_PREFIX.with_name(OUT_PREFIX.name + "_top5.smi")

    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    perfect_df = df[df["pass_6d_threshold"]].copy()
    print(f"6D threshold-passing molecules: {len(perfect_df)} / {len(df)}")

    print(f"Random seed for this run: {RANDOM_SEED}")
    print(f"Randomly sampling {NUM_DISPLAY} molecules from the top {TOP_POOL_SIZE} high-quality candidates.")
    top5 = random_best_five(df, n=NUM_DISPLAY, pool_size=TOP_POOL_SIZE)

    top5.to_csv(top_csv, index=False, encoding="utf-8-sig")
    with open(top_smi, "w", encoding="utf-8") as f:
        for smi in top5["SMILES"]:
            f.write(smi + "\n")

    png_path, svg_path = draw_top5(top5)

    print("\nTop 5 molecules for display:")
    cols = ["SMILES", "JNK3", "GSK3b", "QED", "SA", "TPSA", "LogP", "display_score"]
    print(top5[cols].to_string(index=False))

    print("\nSaved files:")
    print(f"All scored table: {all_csv}")
    print(f"Top 5 table:      {top_csv}")
    print(f"Top 5 SMILES:     {top_smi}")
    print(f"Top 5 PNG figure: {png_path}")
    print(f"Top 5 SVG figure: {svg_path}")


if __name__ == "__main__":
    main()
