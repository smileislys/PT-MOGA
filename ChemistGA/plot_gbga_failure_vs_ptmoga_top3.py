import os
import re
import sys
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Draw, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

RANDOM_SEED = int(os.environ.get("FIGURE_RANDOM_SEED", int.from_bytes(os.urandom(4), "little")))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. Paths: designed for your remote server
# ============================================================
BASE_PATH = Path("/home/liuyansong/ChemistGA-master/ChemistGA-master")
CHEMISTGA_DIR = BASE_PATH / "ChemistGA"
OUTPUT_DIR = BASE_PATH / "output"
SCORING_DIR = BASE_PATH / "scoring"
JNK_GSK_DIR = CHEMISTGA_DIR / "high_score" / "high_score_jnk_gsk"

GBGA_RETRO_LOG = CHEMISTGA_DIR / "retro_ablation_gbga.log"
GBGA_UNSOLVED_SMI = OUTPUT_DIR / "contrast_gbga_unsolved_from_retro_log.smi"

PTMOGA_SCORED_CSV = OUTPUT_DIR / "nsga3_6d_top5_scored_molecules_all_scores.csv"

OUT_PREFIX = OUTPUT_DIR / "figure_gbga_unsolved_vs_ptmoga_top3"

sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(SCORING_DIR))
sys.path.insert(0, str(CHEMISTGA_DIR))
sys.path.insert(0, str(JNK_GSK_DIR))


# ============================================================
# 2. Structural alert SMARTS for highlighting GB-GA failures
#    These are heuristic alerts for visual annotation, not
#    definitive proof of chemical impossibility.
# ============================================================
ALERT_SMARTS = [
    ("peroxide / O-O linkage", "[OX2]-[OX2]"),
    ("azide-like motif", "[$([N-]=[N+]=N),$([N]=[N+]=[N-])]"),
    ("thiocarbonyl-sulfide motif", "[CX3](=S)[SX2]"),
    ("thioester / acyl sulfide motif", "[CX3](=O)[SX2]"),
    ("sulfonyl peroxide motif", "S(=O)(=O)O[O]"),
    ("highly chlorinated phosphorus", "[P](Cl)(Cl)Cl"),
    ("diazo / cumulene-like N motif", "[N-]=[N+]=[*]"),
    ("strained cyclopropyl substituent", "[C;R3]1[C;R3][C;R3]1"),
    ("strained small ring", "[r3,r4]"),
    ("highly substituted amide", "[NX3][CX3](=O)[#6]"),
    ("sulfonamide / sulfone motif", "S(=O)(=O)N"),
    ("reactive acyl halide-like motif", "[CX3](=O)[F,Cl,Br,I]"),
]

ALERT_PATTERNS = []
for label, smarts in ALERT_SMARTS:
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        ALERT_PATTERNS.append((label, patt))


def load_scorers():
    """
    Load exactly the same JNK3/GSK3/QED/SA scorers used in your PT-MOGA code.
    The original model file paths are relative, so initialization happens inside
    high_score_jnk_gsk.
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


def canonicalize(smi):
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def read_smi(path):
    smiles = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            can = canonicalize(line.strip().split()[0])
            if can:
                smiles.append(can)
    return list(dict.fromkeys(smiles))


def extract_gbga_unsolved_from_log():
    """
    Retro* failure lines look like:
    Synthesis path for <SMILES> not found. Please try increasing the number of iterations.
    """
    if not GBGA_RETRO_LOG.exists():
        raise FileNotFoundError(f"Cannot find GB-GA retro log: {GBGA_RETRO_LOG}")

    text = GBGA_RETRO_LOG.read_text(encoding="utf-8", errors="ignore")
    raw = re.findall(r"Synthesis path for\s+(.+?)\s+not found", text)

    smiles = []
    for smi in raw:
        can = canonicalize(smi)
        if can:
            smiles.append(can)

    smiles = list(dict.fromkeys(smiles))
    with open(GBGA_UNSOLVED_SMI, "w", encoding="utf-8") as f:
        for smi in smiles:
            f.write(smi + "\n")
    return smiles


def get_gbga_unsolved_smiles():
    if GBGA_UNSOLVED_SMI.exists():
        smiles = read_smi(GBGA_UNSOLVED_SMI)
        if smiles:
            return smiles
    return extract_gbga_unsolved_from_log()


def score_6d(smiles, method, retro_status):
    scorers = load_scorers()
    jnk3 = np.asarray(scorers["jnk3"](smiles), dtype=float)
    gsk3 = np.asarray(scorers["gsk3"](smiles), dtype=float)
    qed = np.asarray(scorers["qed"](smiles), dtype=float)
    sa = np.asarray(scorers["sa"](smiles), dtype=float)

    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        logp = float(Crippen.MolLogP(mol))

        sa_norm = max(0.0, (10.0 - float(sa[i])) / 9.0)
        tpsa_norm = math.exp(-0.5 * ((tpsa - 60.0) / 30.0) ** 2)
        logp_norm = math.exp(-0.5 * ((logp - 3.0) / 1.5) ** 2)

        norm = np.array([jnk3[i], gsk3[i], qed[i], sa_norm, tpsa_norm, logp_norm], dtype=float)
        pass_6d = (
            float(jnk3[i]) >= 0.5
            and float(gsk3[i]) >= 0.5
            and float(qed[i]) >= 0.6
            and float(sa[i]) <= 4.0
            and tpsa <= 90.0
            and 1.0 <= logp <= 5.0
        )

        alert_labels, alert_atoms, alert_bonds = find_alerts(mol)

        rows.append(
            {
                "method": method,
                "retro_status": retro_status,
                "SMILES": smi,
                "JNK3": float(jnk3[i]),
                "GSK3b": float(gsk3[i]),
                "QED": float(qed[i]),
                "SA": float(sa[i]),
                "SA_norm": sa_norm,
                "TPSA": tpsa,
                "TPSA_norm": tpsa_norm,
                "LogP": logp,
                "LogP_norm": logp_norm,
                "six_dim_mean": float(norm.mean()),
                "six_dim_min": float(norm.min()),
                "six_dim_std": float(norm.std()),
                "display_score": float(norm.mean() + 0.30 * norm.min() - 0.10 * norm.std()),
                "pass_6d_threshold": pass_6d,
                "alert_count": len(alert_atoms),
                "alert_labels": "; ".join(alert_labels) if alert_labels else "Retro* unsolved; no predefined alert matched",
            }
        )
    return pd.DataFrame(rows)


def find_alerts(mol):
    labels = []
    atoms = set()
    bonds = set()
    for label, patt in ALERT_PATTERNS:
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        labels.append(label)
        for match in matches:
            for a in match:
                atoms.add(a)
            for a1, a2 in zip(match[:-1], match[1:]):
                bond = mol.GetBondBetweenAtoms(int(a1), int(a2))
                if bond is not None:
                    bonds.add(bond.GetIdx())
    return labels, sorted(atoms), sorted(bonds)


def fallback_core_region(mol):
    """
    If no predefined structural alert is matched, highlight the largest ring
    system as a visual cue for a Retro*-unsolved molecule. This is only a
    figure annotation and should be described as an unsolved core region.
    """
    ring_atoms = mol.GetRingInfo().AtomRings()
    if not ring_atoms:
        heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        atoms = heavy[: min(6, len(heavy))]
    else:
        atoms = list(max(ring_atoms, key=len))

    atom_set = set(atoms)
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            bonds.append(bond.GetIdx())
    return sorted(atom_set), sorted(bonds)


def prepare_ptmoga_top3():
    if not PTMOGA_SCORED_CSV.exists():
        raise FileNotFoundError(f"Cannot find PT-MOGA scored CSV: {PTMOGA_SCORED_CSV}")

    df = pd.read_csv(PTMOGA_SCORED_CSV)
    if "SMILES" not in df.columns:
        raise ValueError("PT-MOGA CSV must contain a SMILES column.")

    for col in ["JNK3", "GSK3b", "QED", "SA", "TPSA", "LogP"]:
        if col not in df.columns:
            raise ValueError(f"PT-MOGA CSV is missing column: {col}")

    if "display_score" not in df.columns:
        norm_cols = []
        if "SA_norm" not in df.columns:
            df["SA_norm"] = df["SA"].apply(lambda x: max(0.0, (10.0 - float(x)) / 9.0))
        if "TPSA_norm" not in df.columns:
            df["TPSA_norm"] = df["TPSA"].apply(lambda x: math.exp(-0.5 * ((float(x) - 60.0) / 30.0) ** 2))
        if "LogP_norm" not in df.columns:
            df["LogP_norm"] = df["LogP"].apply(lambda x: math.exp(-0.5 * ((float(x) - 3.0) / 1.5) ** 2))
        norm_cols = ["JNK3", "GSK3b", "QED", "SA_norm", "TPSA_norm", "LogP_norm"]
        df["six_dim_mean"] = df[norm_cols].mean(axis=1)
        df["six_dim_min"] = df[norm_cols].min(axis=1)
        df["six_dim_std"] = df[norm_cols].std(axis=1)
        df["display_score"] = df["six_dim_mean"] + 0.30 * df["six_dim_min"] - 0.10 * df["six_dim_std"]

    if "pass_6d_threshold" in df.columns:
        df = df[df["pass_6d_threshold"].astype(str).str.lower().isin(["true", "1", "yes"])]

    df = df.sort_values(["display_score", "six_dim_min", "six_dim_mean"], ascending=[False, False, False]).copy()
    pool = df.head(min(30, len(df)))
    df = pool.sample(n=min(3, len(pool)), random_state=RANDOM_SEED).copy()
    df["method"] = "PT-MOGA"
    df["retro_status"] = "solved"
    df["alert_count"] = 0
    df["alert_labels"] = "Retro* solved"
    return df


def select_gbga_top3():
    smiles = get_gbga_unsolved_smiles()
    df = score_6d(smiles, "GB-GA crossover", "unsolved")
    if df.empty:
        raise RuntimeError("No GB-GA unsolved molecule could be scored.")

    # Prefer representative failed molecules that are still reasonably strong by
    # predicted objectives, while giving priority to structural-alert examples.
    # This avoids unfairly selecting random low-quality molecules.
    df = df.sort_values(
        ["pass_6d_threshold", "alert_count", "SA", "display_score"],
        ascending=[False, False, False, False],
    )

    alert_pool = df[df["alert_count"] > 0].head(min(80, len(df[df["alert_count"] > 0])))
    general_pool = df.head(min(120, len(df)))
    pool = pd.concat([alert_pool, general_pool], ignore_index=True).drop_duplicates("SMILES")

    if len(pool) < 3:
        pool = df

    selected = pool.sample(n=min(3, len(pool)), random_state=RANDOM_SEED).copy()
    return selected, df


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(52, True)
FONT_SUB = load_font(28, False)
FONT_CARD = load_font(31, True)
FONT_TEXT = load_font(24, False)
FONT_SMALL = load_font(21, False)
FONT_METRIC = load_font(22, True)
FONT_METRIC_VALUE = load_font(23, False)


def draw_mol_image(smi, highlight=True, size=(500, 310)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return Image.new("RGB", size, "white")

    atoms, bonds = [], []
    fallback_highlight = False
    if highlight:
        _, atoms, bonds = find_alerts(mol)
        if not atoms:
            atoms, bonds = fallback_core_region(mol)
            fallback_highlight = True

    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2.8
    opts.padding = 0.06
    opts.minFontSize = 16
    opts.maxFontSize = 26
    opts.continuousHighlight = True

    atom_color = (1.0, 0.20, 0.12) if not fallback_highlight else (1.0, 0.42, 0.28)
    bond_color = (0.95, 0.06, 0.02) if not fallback_highlight else (0.95, 0.22, 0.10)
    highlight_atom_colors = {idx: atom_color for idx in atoms}
    highlight_bond_colors = {idx: bond_color for idx in bonds}
    highlight_atom_radii = {idx: 0.30 for idx in atoms}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors=highlight_atom_colors,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_atom_radii,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    from io import BytesIO

    return Image.open(BytesIO(png)).convert("RGB")


def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def write_multiline(draw, x, y, lines, font, fill, line_gap=6):
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill)
        yy += font.size + line_gap
    return yy


def draw_metric_chip(draw, x, y, label, value, accent):
    chip_w, chip_h = 245, 54
    draw.rounded_rectangle(
        (x, y, x + chip_w, y + chip_h),
        radius=13,
        fill=(250, 252, 253),
        outline=(216, 224, 231),
        width=1,
    )
    draw.text((x + 14, y + 8), label, font=FONT_METRIC, fill=accent)
    draw.text((x + 118, y + 8), value, font=FONT_METRIC_VALUE, fill=(28, 34, 40))
    return chip_w


def draw_card(canvas, x, y, w, h, row, is_gbga):
    draw = ImageDraw.Draw(canvas)
    border = (215, 80, 72) if is_gbga else (62, 150, 95)
    head_bg = (255, 244, 242) if is_gbga else (238, 249, 242)
    status_color = (190, 45, 40) if is_gbga else (36, 125, 70)
    accent = (190, 45, 40) if is_gbga else (36, 125, 70)

    rounded_rect(draw, (x, y, x + w, y + h), 22, (255, 255, 255), border, 3)
    rounded_rect(draw, (x, y, x + w, y + 66), 20, head_bg, border, 2)
    draw.text((x + 24, y + 18), row["method"], font=FONT_CARD, fill=(25, 30, 35))
    status_text = f"Retro*: {str(row['retro_status']).upper()}" if is_gbga else f"Retro*: {row['retro_status']}"
    draw.text((x + w - 260, y + 20), status_text, font=FONT_TEXT, fill=status_color)

    mol_img = draw_mol_image(row["SMILES"], highlight=is_gbga, size=(w - 70, 345))
    canvas.paste(mol_img, (x + 35, y + 78))

    if is_gbga:
        alert = str(row.get("alert_labels", "Retro* unsolved"))
        has_alert = int(row.get("alert_count", 0)) > 0
        title = "Potential structural alert:" if has_alert else "Retro* unsolved core region:"
        if not has_alert:
            alert = "No predefined SMARTS alert; core region highlighted"
        if len(alert) > 72:
            alert = alert[:69] + "..."
        draw.rounded_rectangle((x + 30, y + 426, x + w - 30, y + 498), radius=14, fill=(255, 247, 245), outline=(232, 129, 118), width=2)
        draw.text((x + 48, y + 438), title, font=FONT_SMALL, fill=(190, 45, 40))
        draw.text((x + 48, y + 466), alert, font=FONT_SMALL, fill=(190, 45, 40))
    else:
        draw.rounded_rectangle((x + 30, y + 430, x + 480, y + 484), radius=14, fill=(239, 249, 243), outline=(112, 181, 139), width=1)
        draw.text((x + 48, y + 446), "High-scoring solved candidate", font=FONT_SMALL, fill=(36, 125, 70))

    metric_y1 = y + h - 122
    metric_y2 = y + h - 62
    start_x = x + 34
    gap = 14
    chips = [
        ("JNK3", f"{row['JNK3']:.2f}"),
        ("GSK3β", f"{row['GSK3b']:.2f}"),
        ("QED", f"{row['QED']:.2f}"),
        ("SA", f"{row['SA']:.2f}"),
        ("TPSA", f"{row['TPSA']:.1f}"),
        ("LogP", f"{row['LogP']:.2f}"),
    ]
    for idx, (label, value) in enumerate(chips[:3]):
        draw_metric_chip(draw, start_x + idx * (245 + gap), metric_y1, label, value, accent)
    for idx, (label, value) in enumerate(chips[3:]):
        draw_metric_chip(draw, start_x + idx * (245 + gap), metric_y2, label, value, accent)


def draw_contrast_figure(gbga3, ptmoga3):
    W, H = 3840, 2160
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((110, 54), "Retrosynthetic accessibility contrast under six-objective optimization", font=FONT_TITLE, fill=(20, 25, 30))
    draw.text(
        (110, 118),
        "Randomly sampled examples from GB-GA Retro* failures and PT-MOGA high-scoring solved candidates",
        font=FONT_SUB,
        fill=(95, 105, 115),
    )
    draw.line((110, 175, W - 110, 175), fill=(220, 226, 232), width=3)

    draw.text((185, 203), "Graph-based crossover", font=FONT_CARD, fill=(190, 45, 40))
    draw.text((185, 241), "Retro*: unsolved examples", font=FONT_SUB, fill=(190, 45, 40))
    draw.text((2055, 203), "Transformer-guided crossover", font=FONT_CARD, fill=(36, 125, 70))
    draw.text((2055, 241), "Retro*: solved high-score examples", font=FONT_SUB, fill=(36, 125, 70))

    card_w, card_h = 1740, 590
    left_x, right_x = 110, 1990
    top_y, gap_y = 300, 35
    for i in range(3):
        y = top_y + i * (card_h + gap_y)
        draw_card(canvas, left_x, y, card_w, card_h, gbga3.iloc[i], is_gbga=True)
        draw_card(canvas, right_x, y, card_w, card_h, ptmoga3.iloc[i], is_gbga=False)

    note = (
        "Red highlights mark predefined structural-alert matches when present. "
        "Unsolved/solved labels are based on Retro* search within 150 iterations."
    )
    draw.text((110, H - 42), note, font=FONT_SMALL, fill=(95, 105, 115))

    png = OUT_PREFIX.with_suffix(".png")
    tiff = OUT_PREFIX.with_suffix(".tiff")
    canvas.save(png, dpi=(600, 600), optimize=True)
    canvas.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    return png, tiff


def main():
    gbga3, gbga_all = select_gbga_top3()
    ptmoga3 = prepare_ptmoga_top3()

    selected = pd.concat([gbga3, ptmoga3], ignore_index=True)

    gbga_all.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_gbga_unsolved_all_scored.csv"), index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_PREFIX.with_name(OUT_PREFIX.name + "_selected6.csv"), index=False, encoding="utf-8-sig")

    png, tiff = draw_contrast_figure(gbga3, ptmoga3)

    print("Selected GB-GA unsolved molecules:")
    print(f"Random seed for this run: {RANDOM_SEED}")
    print(gbga3[["SMILES", "JNK3", "GSK3b", "QED", "SA", "TPSA", "LogP", "alert_labels"]].to_string(index=False))
    print("\nSelected PT-MOGA solved high-score molecules:")
    print(ptmoga3[["SMILES", "JNK3", "GSK3b", "QED", "SA", "TPSA", "LogP", "display_score"]].to_string(index=False))
    print("\nSaved files:")
    print(OUT_PREFIX.with_name(OUT_PREFIX.name + "_gbga_unsolved_all_scored.csv"))
    print(OUT_PREFIX.with_name(OUT_PREFIX.name + "_selected6.csv"))
    print(png)
    print(tiff)


if __name__ == "__main__":
    main()
