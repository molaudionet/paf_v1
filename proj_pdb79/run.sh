# 1. Verify pipeline works (synthetic data, no PDB files needed)
python raw_fingerprint_baseline.py
python head_to_head.py --mode synthetic --out results/synthetic_test/

# 2. Prepare kinase dataset (downloads ~50 PDB files)
python prepare_kinase_dataset.py --download --out_dir data/kinase_pdbs/

# 3. Run the real experiment
python run_real_kinases.py --pdb_list data/kinase_pdbs/kinase_list.csv \
                           --out results/kinase_v0/

# 4. Look at results
cat results/kinase_v0/comparison_results.csv
open results/kinase_v0/fig_comparison_bar.png
open results/kinase_v0/fig_pca_paf.png
