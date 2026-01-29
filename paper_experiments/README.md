# Paper Experiments

This directory contains a clean, paper-ready experiment pipeline isolated from legacy scripts.
It relies on the CG implementation in the repository root and does not reuse the legacy
CSV outputs or plotting code.

## Setup
If the repository root is not already on `PYTHONPATH`, add it so the CG wrapper can import
`main.run_ai_column_generation`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

If CG dependencies (e.g., `gurobipy`, `numpy`, `torch`) are not installed, the CG solver
will record a `status=error` row instead of crashing the sweep.

## Running experiments
```bash
cd paper_experiments
python src/run_experiments.py --config configs/correctness.yaml
```

## Heterogeneity scenarios
- `U`: Uniform arrivals (baseline).
- `H1`: Clustered arrivals with two peaks.
- `H2`: 50/50 small vs large ships with different processing times and shore costs.

## Plotting and tables
```bash
python analysis/make_plots.py --in results/results_correctness.csv --outdir figs/correctness
python analysis/make_tables.py --in results/results_correctness.csv --out results/results_correctness_table.tex
```

## One-click (Windows)
```bat
paper_experiments\\scripts\\run_all.bat
```

If `matplotlib` is unavailable, `make_plots.py` will emit a warning and exit without error.
