# PT-MOGA

Code accompanying **A Transformer-Guided Genetic Algorithm for Pareto-Based Multi-Objective Molecular Optimization**.

PT-MOGA combines reaction-aware molecular generation with Pareto-based selection for multi-objective molecular optimization. This project includes continuous and discrete scoring experiments, NSGA-II/NSGA-III selection variants, GB-GA crossover comparisons, hypervolume analysis, and computational synthetic feasibility evaluation.

## Code location

The source directory is the repository root. All paths below are relative to that directory.

| Directory | Contents |
|---|---|
| `ChemistGA/high_score/` | Molecular generation, scoring, and selection |
| `ChemistGA/extract/` | Candidate filtering and sampling |
| `ChemistGA/retro/` | Retro* evaluation |
| `ChemistGA/evaluate/` | Novelty, diversity, scaffold counts, and result summaries |
| `ChemistGA/hv_comparison_3d/` | Hypervolume experiments and plotting |
| `data/`, `scoring/` | Input data and scoring components |
| `output/` | Experiment data and evaluation outputs |

## Running the experiments

The scripts were written for Linux and use Python, NumPy, pandas, RDKit, PyTorch, pymoo, and a compatible Transformer implementation. The `transformer_model/` implementation directory and pretrained Transformer weights are not included in this upload. The generation scripts require `transformer_model.onmt.opts_translate.OPT_TRANSLATE` and the corresponding translation modules; these must be supplied and configured before generation can run. Retro* evaluation additionally requires Retro* and its model, reaction-template, and starting-molecule files; Retro* is not bundled in this directory. Refer to the upstream model documentation for setup. A complete pinned environment has not yet been supplied.

### Molecular Transformer setup

Configure Molecular Transformer according to the [official repository](https://github.com/pschwllr/MolecularTransformer). Download `MIT_mixed_augm_model_average_20.pt` from the [pretrained model archive](https://ibm.ent.box.com/v/MolecularTransformerModels), and place it at:

```text
transformer_model/experiments/checkpoints/all/MIT_mixed_augm_model_average_20.pt
```

The model path above is relative to the repository root. Downloading the weights alone does not supply the required Transformer implementation.

1. Update `base_path` / `BASE_PATH`, model paths, and input/output paths for your installation. Several scripts retain absolute paths from the original server.
2. Select the task, run identifier, and experiment switches in the relevant scripts.
3. Run generation, candidate extraction, Retro* evaluation, and metric calculation in that order. To analyze existing results, use their matching evaluation scripts directly.

For example, after configuring paths and dependencies, the following commands evaluate the existing discrete three-objective candidate file for run 1 and then calculate its structural metrics. Run them from the source directory:

```bash
python ChemistGA/retro/run_retro_drd3d_sum_recorded_5000.py --task-prefix drd_discrete --seeds 1
python ChemistGA/evaluate/evaluate_drd3d_discrete_seed1.py
```

Keep the actual sampled SMILES and corresponding per-molecule results. Some older sampling scripts do not set a random seed. Match each evaluation to its recorded input; filenames alone do not establish the algorithm or sample size. Computational synthetic feasibility does not establish experimental synthesis.

## Attribution and license

This implementation builds on [ChemistGA](https://github.com/jkwang93/ChemistGA), with extensions for Pareto selection, additional objective configurations, and experimental evaluation. Reaction-aware crossover originates from the ChemistGA/Molecular Transformer approach and is not claimed as an original component of this project.

Please cite the PT-MOGA manuscript when using these extensions and acknowledge the underlying work:

- Wang et al. *ChemistGA: A Chemical Synthesizable Accessible Molecular Generation Algorithm for Real-World Drug Discovery*. J. Med. Chem. 2022, 65, 12482-12496. DOI: [10.1021/acs.jmedchem.2c01179](https://doi.org/10.1021/acs.jmedchem.2c01179).
- [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).
- [Retro*](https://github.com/binghong-ml/retro_star).

The project extensions and inherited ChemistGA code are provided under the MIT License; see `LICENSE.md`. Original copyright notices are retained. Third-party software, datasets, and pretrained models remain subject to their respective licenses.
