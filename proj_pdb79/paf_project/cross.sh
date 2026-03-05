set -x

python run_cross_family_paf.py \
     --csv data/cross_family/cross_family_list.csv \
     --out results/cross_family/ \
     --radius 10 --gamma_fm 0.15 --sigma_t 0.04

#python patch_cross_family_failures.py \
#  --csv data/cross_family/cross_family_list.csv \
#  --pdb_dir data/cross_family/pdbs/ \
#  --download
