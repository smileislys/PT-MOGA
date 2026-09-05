from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


POSSIBLE_BASE_PATHS = [
    Path("/home/liuyansong/ChemistGA-master/ChemistGA-master"),
    Path(r"D:\study1\liuyansong\ChemistGA-master\ChemistGA-master"),
]

BASE_PATH = next((path for path in POSSIBLE_BASE_PATHS if path.exists()), POSSIBLE_BASE_PATHS[0])
OUT_DIR = BASE_PATH / "output" / "chemical_space_tsne_6d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GBGA_SEED = 0
TRANSFORMER_SEED = 2
EXPECTED_INITIAL_MOLECULES = 287
EXPECTED_CANDIDATES_PER_CROSSOVER = 300
TSNE_RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
TSNE_PERPLEXITY = 35


DATASETS = {
    "Initial molecules": [
        BASE_PATH / "data" / "inh" / "jnk_gsk.csv",
    ],
    "GB-GA crossover variant": [
        BASE_PATH
        / "output"
        / f"retro_recorded_ablation_gbga_seed{GBGA_SEED}_n300"
        / f"ablation_gbga_seed{GBGA_SEED}_n300_sampled.smi",
    ],
    "PT-MOGA": [
        BASE_PATH
        / "output"
        / f"retro_recorded_nsga_seed{TRANSFORMER_SEED}_n300"
        / "nsga3_6d"
        / f"nsga3_6d_seed{TRANSFORMER_SEED}_sampled_300.smi",
    ],
}


COLORS = {
    "Initial molecules": "#9AA4B2",
    "GB-GA crossover variant": "#E07A2F",
    "PT-MOGA": "#2166AC",
}

# Publication-facing labels. Internal dataset keys are kept unchanged so that
# data loading, styling, and t-SNE calculations remain identical.
DISPLAY_LABELS = {
    "Initial molecules": "Initial molecules",
    "GB-GA crossover variant": "GB-GA crossover",
    "PT-MOGA": "Transformer crossover",
}


def read_smiles_file(path: Path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return []

    smiles = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Supports .smi, one-column csv, or csv with score columns.
            first = line.replace("\t", ",").split(",")[0].strip()
            if first and first.lower() not in {"smiles", "smile"}:
                smiles.append(first)
    return smiles


def canonicalize_smiles(smiles):
    """Validate and canonicalize entries without removing duplicate candidates."""
    valid = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        valid.append(can)
    return valid


def morgan_fp_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_BITS
    )


def tanimoto_distance_matrix(fps):
    n = len(fps)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dist[i, :] = 1.0 - np.asarray(sims, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def collect_data():
    rows = []
    for label, paths in DATASETS.items():
        all_smiles = []
        for path in paths:
            all_smiles.extend(read_smiles_file(path))
        all_smiles = canonicalize_smiles(all_smiles)

        # Keep the complete initial/reference set and directly use the two
        # pre-sampled 300-molecule candidate files. No Retro* solvability filter
        # or second random sampling is applied before joint t-SNE fitting.
        if label == "Initial molecules" and len(all_smiles) != EXPECTED_INITIAL_MOLECULES:
            raise RuntimeError(
                f"Expected {EXPECTED_INITIAL_MOLECULES} initial molecules, but loaded "
                f"{len(all_smiles)}. Check the reference input before plotting."
            )
        if (
            label != "Initial molecules"
            and len(all_smiles) != EXPECTED_CANDIDATES_PER_CROSSOVER
        ):
            raise RuntimeError(
                f"Expected {EXPECTED_CANDIDATES_PER_CROSSOVER} molecules for {label}, "
                f"but loaded {len(all_smiles)}. Check the sampled input file."
            )

        print(f"{label}: {len(all_smiles)} valid candidate entries")
        for smi in all_smiles:
            rows.append({"model": label, "smiles": smi})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid SMILES were loaded. Please check input paths.")
    return df


def make_tsne(df):
    fps = []
    keep_rows = []
    for _, row in df.iterrows():
        fp = morgan_fp_from_smiles(row["smiles"])
        if fp is not None:
            fps.append(fp)
            keep_rows.append(row)

    df = pd.DataFrame(keep_rows).reset_index(drop=True)
    print(f"Total molecules for t-SNE: {len(df)}")

    dist = tanimoto_distance_matrix(fps)
    perplexity = min(TSNE_PERPLEXITY, max(5, (len(df) - 1) // 3))
    print(f"t-SNE perplexity: {perplexity}")

    tsne_kwargs = {
        "n_components": 2,
        "metric": "precomputed",
        "init": "random",
        "perplexity": perplexity,
        "random_state": TSNE_RANDOM_STATE,
    }

    # Different scikit-learn versions use either max_iter or n_iter.
    try:
        tsne = TSNE(**tsne_kwargs, learning_rate="auto", max_iter=1500)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, learning_rate=200.0, n_iter=1500)

    coords = tsne.fit_transform(dist)
    df["tSNE-1"] = coords[:, 0]
    df["tSNE-2"] = coords[:, 1]
    return df


def plot_tsne(df):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(6.9, 5.0), dpi=300)

    order = ["Initial molecules", "GB-GA crossover variant", "PT-MOGA"]
    for label in order:
        sub = df[df["model"] == label]
        if sub.empty:
            continue
        legend_label = DISPLAY_LABELS[label]
        if label == "Initial molecules":
            ax.scatter(
                sub["tSNE-1"], sub["tSNE-2"],
                s=10, c=COLORS[label], alpha=0.32,
                edgecolors="none", label=legend_label, rasterized=True
            )
        elif label == "GB-GA crossover variant":
            ax.scatter(
                sub["tSNE-1"], sub["tSNE-2"],
                s=15, c=COLORS[label], alpha=0.55,
                edgecolors="white", linewidths=0.15,
                label=legend_label, rasterized=True
            )
        else:
            ax.scatter(
                sub["tSNE-1"], sub["tSNE-2"],
                s=18, c=COLORS[label], alpha=0.72,
                edgecolors="white", linewidths=0.18,
                label=legend_label, rasterized=True
            )

    ax.set_title(
        "Chemical space visualization",
        pad=7,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA4B2")
    ax.spines["bottom"].set_color("#9AA4B2")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.margins(x=0.018, y=0.022)
    ax.legend(
        frameon=False,
        loc="upper right",
        markerscale=1.8,
        borderaxespad=0.4,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.925)

    run_tag = (
        f"gbga_seed{GBGA_SEED}_transformer_seed{TRANSFORMER_SEED}"
        f"_n{EXPECTED_CANDIDATES_PER_CROSSOVER}"
    )
    stem = f"Figure_chemical_space_tSNE_initial_GBGA_Transformer_{run_tag}"
    png = OUT_DIR / f"{stem}.png"
    tif = OUT_DIR / f"{stem}_600dpi.tiff"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"

    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(png, dpi=600, **save_kwargs)
    fig.savefig(tif, dpi=600, **save_kwargs)
    fig.savefig(pdf, **save_kwargs)
    fig.savefig(svg, **save_kwargs)
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {tif}")
    print(f"Saved: {pdf}")
    print(f"Saved: {svg}")


def main():
    df = collect_data()
    df_tsne = make_tsne(df)

    run_tag = (
        f"gbga_seed{GBGA_SEED}_transformer_seed{TRANSFORMER_SEED}"
        f"_n{EXPECTED_CANDIDATES_PER_CROSSOVER}"
    )
    csv_path = OUT_DIR / f"chemical_space_tSNE_coordinates_{run_tag}.csv"
    df_tsne.to_csv(csv_path, index=False)
    print(f"Saved coordinates: {csv_path}")

    counts = df_tsne.groupby("model").size().reset_index(name="n")
    counts.to_csv(OUT_DIR / f"chemical_space_tSNE_group_counts_{run_tag}.csv", index=False)
    print(counts)

    plot_tsne(df_tsne)


if __name__ == "__main__":
    main()
