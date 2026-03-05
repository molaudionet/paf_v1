set -x

python ligand_retrieval_v1.py \
  --pairs_csv data/cocrystal_pairs.csv \
  --pdb_dir . \
  --out results/ligand_retrieval_v1 \
  --radius 10.0 \
  --gamma_fm 0.15 \
  --sigma_t 0.04 \
  --easy_negatives 50 \
  --hard_negatives 50
