set -x

python curate_dfg_dataset.py --out data/dfg/ --download
python run_dfg_experiment.py \
  --csv data/dfg/dfg_list.csv \
  --out results/dfg/ \
  --radius 10 --gamma_fm 0.15 --sigma_t 0.04
