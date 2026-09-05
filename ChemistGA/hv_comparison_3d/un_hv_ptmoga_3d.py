#!/usr/bin/env python3
"""Run the legacy PT-MOGA experiment logic and record 3D HV checkpoints."""

from __future__ import annotations

import argparse
import random
import traceback

from rdkit import Chem

from hv_common_3d import (
    ThreeObjectiveScorer,
    capture_random_state,
    configure_project,
    load_or_create_initial_population,
    load_state,
    make_unique_parent_pairs,
    objective_matrix,
    print_metrics,
    record_checkpoint,
    restore_random_state,
    save_state,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path",
        default="/home/liuyansong/ChemistGA-master/ChemistGA-master",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--mutation-rate", type=float, default=0.01)
    parser.add_argument("--elite-fraction", type=float, default=0.5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tournament_parent(population, scalar_scores, k=2):
    indices = random.sample(range(len(population)), k)
    winner = max(indices, key=lambda index: scalar_scores[index])
    return population[winner]


def reproduce_legacy(
    population,
    scores_3d,
    number_of_parent_pairs,
    mutation_rate,
    get_synthesis_molecules,
    mutate_module,
):
    scalar_scores = [float(sum(values)) for values in scores_3d]
    parent_pairs = make_unique_parent_pairs(
        choose_parent=lambda: tournament_parent(population, scalar_scores, k=2),
        number_of_pairs=number_of_parent_pairs,
        require_different=True,
    )
    _, selected_children, all_transformer_candidates, _ = get_synthesis_molecules(
        parent_pairs
    )

    mutated_offspring = []
    for child in selected_children:
        mol = Chem.MolFromSmiles(child)
        if mol is None:
            continue
        mutated = mutate_module.mutate(mol, mutation_rate)
        if mutated is not None:
            mutated_offspring.append(Chem.MolToSmiles(mutated, canonical=True))

    # Match the previously executed PT-MOGA script: mutation is called, but the
    # effective environmental-selection pool is the unmutated Transformer
    # candidate pool returned as all_population.
    if not all_transformer_candidates:
        raise RuntimeError("PT-MOGA produced no valid Transformer candidates.")
    return mutated_offspring, list(all_transformer_candidates)


def legacy_mixed_pareto_environmental_selection(
    population,
    scores_3d,
    population_size,
    elite_fraction,
    pareto_selection,
):
    elite_size = max(1, min(population_size, int(round(population_size * elite_fraction))))
    elites, elite_scores = pareto_selection(
        population,
        scores_3d,
        min(elite_size, len(population)),
    )

    # Preserve the previous implementation exactly: once a SMILES is selected
    # as an elite, all identical SMILES entries are removed from the non-elite pool.
    elite_set = set(elites)
    remaining = [
        (smi, score)
        for smi, score in zip(population, scores_3d)
        if smi not in elite_set
    ]

    slots = population_size - len(elites)
    if len(remaining) >= slots:
        exploration = random.sample(remaining, slots)
    else:
        exploration = list(remaining)

    return (
        list(elites) + [item[0] for item in exploration],
        list(elite_scores) + [item[1] for item in exploration],
    )


def main():
    args = parse_args()
    paths = configure_project(args.base_path)

    scorer = ThreeObjectiveScorer()
    if args.dry_run:
        test_population = load_or_create_initial_population(
            paths["initial_library"],
            paths["output"]
            / "shared"
            / f"initial_population_seed{args.seed}_n{args.population_size}.smi",
            args.population_size,
            args.seed,
        )
        print(scorer(test_population[:5]))
        print("Dry run passed: paths, initial population and scoring model are available.")
        return

    import mutate as mutate_module
    from high_score_crossover_first_model_drd import get_synthesis_molecules
    from pareto_ranking import pareto_selection

    run_dir = (
        paths["output"]
        / f"ptmoga_legacy_revised_seed{args.seed}_mut{args.mutation_rate:g}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "resume_state.pkl"
    config = {
        "seed": args.seed,
        "population_size": args.population_size,
        "generations": args.generations,
        "checkpoint_every": args.checkpoint_every,
        "mutation_rate": args.mutation_rate,
        "elite_fraction": args.elite_fraction,
        "offspring_flow": "legacy_unmutated_transformer_candidate_pool",
    }

    state = None if args.no_resume else load_state(state_file)
    if state is not None:
        if state["config"] != config:
            raise RuntimeError(
                "Resume-state configuration differs from the current command. "
                "Use --no-resume or restore the original arguments."
            )
        population = state["population"]
        scores_3d = state["scores_3d"]
        completed_generation = int(state["completed_generation"])
        restore_random_state(state)
        print(f"Resuming PT-MOGA from generation {completed_generation}.")
    else:
        population = load_or_create_initial_population(
            paths["initial_library"],
            paths["output"]
            / "shared"
            / f"initial_population_seed{args.seed}_n{args.population_size}.smi",
            args.population_size,
            args.seed,
        )
        seed_everything(args.seed)
        initial_frame = scorer(population)
        scores_3d = objective_matrix(initial_frame).tolist()
        completed_generation = 0
        metrics = record_checkpoint(
            "PT-MOGA",
            0,
            population,
            scorer,
            run_dir,
        )
        print_metrics(metrics)

    for generation in range(completed_generation + 1, args.generations + 1):
        try:
            mutated_offspring, transformer_candidates = reproduce_legacy(
                population,
                scores_3d,
                args.population_size,
                args.mutation_rate,
                get_synthesis_molecules,
                mutate_module,
            )
            # Intentionally unused for population updating, matching the previous
            # experiment implementation. The count remains available for logs.
            print(
                f"Generation {generation}: mutated children={len(mutated_offspring)}, "
                f"Transformer candidates={len(transformer_candidates)}",
                flush=True,
            )
            candidate_frame = scorer(transformer_candidates)
            candidate_scores = objective_matrix(candidate_frame).tolist()
            population, scores_3d = legacy_mixed_pareto_environmental_selection(
                transformer_candidates,
                candidate_scores,
                args.population_size,
                args.elite_fraction,
                pareto_selection,
            )
        except Exception:
            traceback.print_exc()
            raise

        state = {
            "config": config,
            "completed_generation": generation,
            "population": population,
            "scores_3d": scores_3d,
            **capture_random_state(),
        }
        save_state(state_file, state)

        if generation % args.checkpoint_every == 0 or generation == args.generations:
            metrics = record_checkpoint(
                "PT-MOGA",
                generation,
                population,
                scorer,
                run_dir,
            )
            print_metrics(metrics)

    print(f"PT-MOGA HV experiment finished: {run_dir}")


if __name__ == "__main__":
    main()
