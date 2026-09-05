#!/usr/bin/env python3
"""Shared utilities for the controlled DRD2 three-objective HV experiment."""

from __future__ import annotations

import csv
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rdkit import Chem


PathLike = Union[str, Path]


def configure_project(base_path: PathLike) -> Dict[str, Path]:
    base = Path(base_path).expanduser().resolve()
    paths = {
        "base": base,
        "chemistga": base / "ChemistGA",
        "drd": base / "ChemistGA" / "high_score" / "high_score_drd",
        "scoring": base / "scoring",
        "transformer": base / "transformer_model",
        "initial_library": base / "data" / "inh" / "drd_succ_250.csv",
        "output": base / "output" / "hv_comparison_3d",
    }
    required = [
        paths["chemistga"],
        paths["drd"],
        paths["scoring"],
        paths["transformer"],
        paths["initial_library"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required project paths are missing:\n" + "\n".join(missing))

    for path in [paths["drd"], paths["chemistga"], paths["scoring"], paths["transformer"], base]:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    paths["output"].mkdir(parents=True, exist_ok=True)
    return paths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_smiles(path: PathLike) -> List[str]:
    path = Path(path)
    values = pd.read_csv(path, header=None).values.flatten().tolist()
    valid = []
    for value in values:
        if pd.isna(value):
            continue
        smi = str(value).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol, canonical=True))
    if not valid:
        raise RuntimeError(f"No valid SMILES were loaded from {path}")
    return valid


def load_or_create_initial_population(
    library_file: PathLike,
    shared_file: PathLike,
    population_size: int,
    seed: int,
) -> List[str]:
    shared_file = Path(shared_file)
    if shared_file.exists():
        population = [
            line.strip()
            for line in shared_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(population) != population_size:
            raise RuntimeError(
                f"Shared initial population has {len(population)} entries; "
                f"expected {population_size}: {shared_file}"
            )
        return population

    library = read_smiles(library_file)
    rng = random.Random(seed)
    population = [rng.choice(library) for _ in range(population_size)]
    shared_file.parent.mkdir(parents=True, exist_ok=True)
    shared_file.write_text("\n".join(population) + "\n", encoding="utf-8")
    return population


class ThreeObjectiveScorer:
    """DRD2, QED and normalized synthetic-accessibility scores."""

    def __init__(self) -> None:
        from high_score_properties_drd2 import get_scoring_function

        try:
            self.drd2 = get_scoring_function("drd2")
        except ValueError as error:
            raise RuntimeError(
                "The DRD2 random-forest pickle was created with scikit-learn "
                "0.23.1 and is incompatible with the active scikit-learn build. "
                "Run this experiment in the same remote environment that already "
                "executes the original ChemistGA scoring code."
            ) from error
        self.qed = get_scoring_function("qed")
        self.sa = get_scoring_function("sa")

    def __call__(self, smiles: Sequence[str]) -> pd.DataFrame:
        if not smiles:
            return pd.DataFrame(
                columns=["smiles", "drd2", "qed", "sa_raw", "sa_norm", "feasible"]
            )

        drd2 = np.asarray(self.drd2(list(smiles)), dtype=float)
        qed = np.asarray(self.qed(list(smiles)), dtype=float)
        sa_raw = np.asarray(self.sa(list(smiles)), dtype=float)
        if not (len(drd2) == len(qed) == len(sa_raw) == len(smiles)):
            raise RuntimeError("Scoring functions returned inconsistent lengths.")

        sa_norm = np.clip((10.0 - sa_raw) / 9.0, 0.0, 1.0)
        drd2 = np.clip(drd2, 0.0, 1.0)
        qed = np.clip(qed, 0.0, 1.0)
        feasible = (drd2 >= 0.5) & (qed >= 0.6) & (sa_raw <= 4.0)

        return pd.DataFrame(
            {
                "smiles": list(smiles),
                "drd2": drd2,
                "qed": qed,
                "sa_raw": sa_raw,
                "sa_norm": sa_norm,
                "feasible": feasible.astype(int),
            }
        )


def objective_matrix(score_frame: pd.DataFrame) -> np.ndarray:
    return score_frame[["drd2", "qed", "sa_norm"]].to_numpy(dtype=float)


def non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Return the nondominated mask for a maximization problem."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated_by_other = np.all(points >= points[i], axis=1) & np.any(
            points > points[i], axis=1
        )
        mask[i] = not np.any(dominated_by_other)
    return mask


def crowding_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n, n_objectives = points.shape
    distances = np.zeros(n, dtype=float)
    if n <= 2:
        distances[:] = np.inf
        return distances

    for objective in range(n_objectives):
        order = np.argsort(points[:, objective], kind="mergesort")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        minimum = points[order[0], objective]
        maximum = points[order[-1], objective]
        if maximum <= minimum:
            continue
        for position in range(1, n - 1):
            index = order[position]
            if np.isinf(distances[index]):
                continue
            previous_value = points[order[position - 1], objective]
            next_value = points[order[position + 1], objective]
            distances[index] += (next_value - previous_value) / (maximum - minimum)
    return distances


def fixed_size_pareto_subset(points: np.ndarray, size: int) -> np.ndarray:
    """Select a common-size evaluation subset using fronts and crowding distance."""
    points = np.asarray(points, dtype=float)
    if len(points) <= size:
        return points

    remaining = np.arange(len(points))
    selected = []
    while len(selected) < size and len(remaining):
        mask = non_dominated_mask(points[remaining])
        front = remaining[mask]
        slots = size - len(selected)
        if len(front) <= slots:
            selected.extend(front.tolist())
        else:
            distances = crowding_distance(points[front])
            order = np.argsort(-distances, kind="mergesort")
            selected.extend(front[order[:slots]].tolist())
        remaining = remaining[~mask]
    return points[np.asarray(selected, dtype=int)]


def _hypervolume_2d_origin(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    points = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    area = 0.0
    previous_y = 0.0
    for y in sorted(set(points[:, 0].tolist())):
        active = points[points[:, 0] >= y]
        max_z = float(np.max(active[:, 1])) if len(active) else 0.0
        area += (float(y) - previous_y) * max_z
        previous_y = float(y)
    return area


def hypervolume_3d_origin(points: np.ndarray) -> float:
    """Exact 3D HV for maximization with the fixed reference point (0, 0, 0)."""
    if len(points) == 0:
        return 0.0
    points = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    points = points[non_dominated_mask(points)]
    volume = 0.0
    previous_x = 0.0
    for x in sorted(set(points[:, 0].tolist())):
        active = points[points[:, 0] >= x]
        area = _hypervolume_2d_origin(active[:, 1:3])
        volume += (float(x) - previous_x) * area
        previous_x = float(x)
    return float(volume)


def original_continuous_fitness(score_frame: pd.DataFrame) -> np.ndarray:
    """Author fitness: DRD2 + QED + I(SA <= 4)."""
    return (
        score_frame["drd2"].to_numpy(dtype=float)
        + score_frame["qed"].to_numpy(dtype=float)
        + (score_frame["sa_raw"].to_numpy(dtype=float) <= 4.0).astype(float)
    )


def safe_probabilities(fitness: Sequence[float]) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    values[~np.isfinite(values)] = 0.0
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return values / total


def make_unique_parent_pairs(
    choose_parent,
    number_of_pairs: int,
    require_different: bool,
    max_attempts: int = 100000,
) -> List[str]:
    pairs: List[str] = []
    seen = set()
    attempts = 0
    while len(pairs) < number_of_pairs:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not construct {number_of_pairs} unique parent pairs."
            )
        parent_a = choose_parent()
        parent_b = choose_parent()
        if require_different and parent_a == parent_b:
            continue
        pair = ".".join(sorted([parent_a, parent_b]))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def record_checkpoint(
    model_name: str,
    generation: int,
    population: Sequence[str],
    scorer: ThreeObjectiveScorer,
    output_dir: PathLike,
    hv_evaluation_size: int = 100,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    population_dir = output_dir / "populations"
    population_dir.mkdir(parents=True, exist_ok=True)

    scores = scorer(population)
    points = objective_matrix(scores)
    nd_mask = non_dominated_mask(points)
    scores["nondominated"] = nd_mask.astype(int)
    scores.insert(0, "generation", generation)
    scores.insert(0, "model", model_name)
    scores.to_csv(
        population_dir / f"generation_{generation:03d}.csv",
        index=False,
    )

    hv_points = fixed_size_pareto_subset(points, hv_evaluation_size)
    metrics: Dict[str, object] = {
        "model": model_name,
        "generation": int(generation),
        "population_size": int(len(scores)),
        "unique_smiles": int(scores["smiles"].nunique()),
        "hv_evaluation_size": int(len(hv_points)),
        "hypervolume": hypervolume_3d_origin(hv_points),
        "hypervolume_full_population": hypervolume_3d_origin(points),
        "nondominated_count": int(nd_mask.sum()),
        "feasible_count": int(scores["feasible"].sum()),
        "feasible_rate": float(scores["feasible"].mean() * 100.0),
        "drd2_mean": float(scores["drd2"].mean()),
        "drd2_median": float(scores["drd2"].median()),
        "drd2_max": float(scores["drd2"].max()),
        "qed_mean": float(scores["qed"].mean()),
        "qed_median": float(scores["qed"].median()),
        "qed_max": float(scores["qed"].max()),
        "sa_raw_mean": float(scores["sa_raw"].mean()),
        "sa_raw_median": float(scores["sa_raw"].median()),
        "sa_raw_min": float(scores["sa_raw"].min()),
        "sa_norm_mean": float(scores["sa_norm"].mean()),
        "sa_norm_median": float(scores["sa_norm"].median()),
        "sa_norm_max": float(scores["sa_norm"].max()),
    }
    upsert_metrics(output_dir / "generation_metrics.csv", metrics)
    return metrics


def upsert_metrics(path: PathLike, row: Dict[str, object]) -> None:
    path = Path(path)
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[frame["generation"] != row["generation"]]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("generation")
    frame.to_csv(path, index=False)


def save_state(path: PathLike, state: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_state(path: PathLike) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def restore_random_state(state: Dict) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    try:
        import torch

        if state.get("torch_random_state") is not None:
            torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and state.get("torch_cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])
    except Exception:
        pass


def capture_random_state() -> Dict:
    result = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": None,
        "torch_cuda_random_state": None,
    }
    try:
        import torch

        result["torch_random_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            result["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return result


def print_metrics(metrics: Dict[str, object]) -> None:
    print(
        f"[generation {int(metrics['generation']):02d}] "
        f"HV={metrics['hypervolume']:.6f} | "
        f"feasible={metrics['feasible_rate']:.1f}% | "
        f"ND={int(metrics['nondominated_count'])} | "
        f"DRD2={metrics['drd2_mean']:.3f} | "
        f"QED={metrics['qed_mean']:.3f} | "
        f"SA_norm={metrics['sa_norm_mean']:.3f}",
        flush=True,
    )
