pip install -U numpy pandas scipy matplotlib biopython
# DSSP is optional. If you want it:
#   mac: brew install dssp   (or mkdssp)
#   linux: apt-get install dssp
#The following 4 scripts are from openai 5.2
#sound-of-molecules) bash-3.2$ vi paf_core_v1.py
#(sound-of-molecules) bash-3.2$ vi evaluate_metrics.py
#(sound-of-molecules) bash-3.2$ vi run_paf_enriched.py
#(sound-of-molecules) bash-3.2$ vi ablate_paf_params.py



python run_paf_enriched.py \
  --kinase_csv ./data/kinase_list.csv \
  --out results/paf_v1_enriched_two_anchor/

python ablate_paf_params.py \
  --kinase_csv ./data/kinase_list.csv \
  --out results/paf_v1_ablation_grid/
