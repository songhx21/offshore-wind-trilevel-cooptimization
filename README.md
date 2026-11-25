# Offshore wind farm tri-level co-optimization

Source code for **Tri-Level Co-optimization of Offshore Wind Farm Layout, Collector Topology, and Yaw Control Considering Operating Exposure and Collector Reliability**.

This repository publishes source code only. Experimental result files, learned checkpoints, run-level data, and supplementary Word/PDF documents are not included.

## Layout

- `experiments/25T/`: sixteen current optimization scripts for the 25-turbine case.
- `experiments/50T/`: sixteen current optimization scripts plus the shared site configuration, performance cache, and case-verification helper.
- `historical/`: a separately identified historical training-script snapshot.
- `scripts/run_experiment.py`: single-run launcher with preview as the default.
- `scripts/verify_archive.py`: offline geometric, connectivity and export-consistency checker for an accompanying local data archive.
- `scripts/analyze_archive.py`: paired-statistics and supplementary-figure generator for the local data archive.
- `scripts/recover_recorded_source.py`: exact-hash recovery helper requiring the local provenance records.

The archive-analysis helpers require the separate `data/` directory prepared with the supplementary materials. They do not download or publish experimental results. Run them from the complete local supplementary package, or place the corresponding local archive beside `scripts/`.

## Configurations and workflows

| Configuration | Script stem | Added planning criterion |
|---|---|---|
| AWS | v7 | Active wake steering and collection loss |
| FAE | v8-fatigue | Fatigue exposure and electrical path weighting |
| RUC | v9-robust | Adverse operating scenarios and vulnerability weighting |
| RNT | v10-reliability | Single-edge outage connectivity |

The script suffix identifies the workflow: no suffix is BASE, `-reopt` is SEQ, `-coopt` is CO, and `-coopt-optimized` is TCO. Case-specific directories prevent mixing 25- and 50-turbine settings.

## Environment

Use a dedicated Python environment. Install PyTorch for the local CUDA/CPU environment, followed by the dependencies in `requirements.txt`. Python 3.12 is a practical starting point. The dependency list is not a historical lockfile, and full training has not been rerun as part of this packaging step.

```bash
python -m pip install -r requirements.txt
```

## Preview and launch

Preview a selected experiment without running training:

```bash
python scripts/run_experiment.py --case 25T --configuration RNT --workflow TCO --seed 42
```

Start the complete optimization explicitly:

```bash
python scripts/run_experiment.py --case 25T --configuration RNT --workflow TCO --seed 42 --run
```

Change `25T` to `50T` for the multi-zone case. The job executes layout search, expert generation and pre-training, policy search, refinement, and final export. It is a long-running research computation and does not resume interrupted policy checkpoints. The launcher creates new results within the case directory; `.gitignore` excludes these outputs.

## Method and version scope

TCO allocates a population of 120, 100 generations and three layout restarts, with 2400 expert demonstrations. BASE, SEQ and CO allocate 80, 60 and one restart, with 800 demonstrations. Policy search uses 2500 episodes. These are workflow budgets, not an equal-budget controlled comparison.

The source combines feasibility-masked cable actions with hard-clipped Gaussian yaw actions and a PPO-inspired hybrid surrogate. Keep these implementation details when reproducing the search. The composite quantity is an engineering planning score, not a discounted project cash-flow estimate. The exact source snapshot used by an individual historical run must be checked against that run's metadata; the current files are not asserted to be the identical source of every historical experiment.

## Site attribution

The 50-turbine case is a study-specific adaptation of IEA Wind Task 37 WFLO Case Study 4, using a Borssele III/IV site and wind-resource basis. The original case contains 81 turbines; this implementation uses 50. Adapted geometry and aggregated wind-resource settings are embedded in `experiments/50T/large_case_config.py`.

- Original case resource: https://github.com/byuflowlab/iea37-wflo-casestudies/tree/master/cs3-4
- Benchmark paper: https://doi.org/10.5194/wes-8-865-2023

## Citation and permissions

Identify the repository commit when referring to this implementation. The manuscript is not presented as an accepted publication. An open-source redistribution license has not yet been assigned by the authors; public visibility does not by itself grant an open-source license. Third-party libraries and case resources retain their respective terms.
