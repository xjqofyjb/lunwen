@echo off
setlocal

cd /d %~dp0\..

python src\run_experiments.py --config configs\correctness.yaml
python src\run_experiments.py --config configs\main.yaml
python src\run_experiments.py --config configs\hetero.yaml

python analysis\make_tables.py --in results\results_correctness.csv --out results\results_correctness_table.tex
python analysis\make_tables.py --in results\results_main.csv --out results\results_main_table.tex
python analysis\make_tables.py --in results\results_hetero.csv --out results\results_hetero_table.tex

python analysis\make_plots.py --in results\results_correctness.csv --outdir figs\correctness
python analysis\make_plots.py --in results\results_main.csv --outdir figs\main
python analysis\make_plots.py --in results\results_hetero.csv --outdir figs\hetero

endlocal
