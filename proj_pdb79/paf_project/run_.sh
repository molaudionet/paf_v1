set -x

python paf_core_v1.py
python evaluate_metrics.py
python run_paf_enriched.py
python ablate_paf_params.py
