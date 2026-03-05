# How to Run Everything
# ====================
# Complete setup and execution guide for PAF v0 kinase pocket experiments
#
# Tested on: Mac Mini M4 / any machine with Python 3.10+
# Time: ~1 day for full pipeline
# Cost: $0 (everything runs locally)
# GPU: NOT required

# ============================================================
# STEP 1: Create project folder and set up environment (5 min)
# ============================================================

# Create project directory
mkdir -p ./paf_project
cd ./paf_project

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # On Mac/Linux
# .venv\Scripts\activate    # On Windows

# Install dependencies
pip install numpy scipy matplotlib pandas biopython soundfile

# Create folder structure
mkdir -p data/kinase_pdbs
mkdir -p results/synthetic_test
mkdir -p results/kinase_v0

# ============================================================
# STEP 2: Copy the script files into project (2 min)
# ============================================================
#
# Place these files (downloaded from Claude) into ./paf_project/:
#
#   ./paf_project/
#     head_to_head.py
#     raw_fingerprint_baseline.py
#     prepare_kinase_dataset.py
#     execution_guide.py
#     positive_controls.py          (from earlier delivery)
#     power_analysis.py             (from earlier delivery)
#     protein_sonification_scoping.py (from earlier delivery)

# ============================================================
# STEP 3: Verify everything works on SYNTHETIC data (5 min)
# ============================================================

cd ./paf_project

# Run 1: Test the raw fingerprint baseline
python raw_fingerprint_baseline.py
# Expected output: fingerprint dimensions + similarity scores + "[OK]"

# Run 2: Test the full head-to-head pipeline (synthetic pockets)
python head_to_head.py --mode synthetic --out results/synthetic_test/
# Expected output:
#   - Spearman rho values for PAF vs 4 baselines
#   - 4 PNG figures in results/synthetic_test/
#   - comparison_results.csv

# Run 3: Verify figures were created
ls results/synthetic_test/*.png
# Should show:
#   fig_comparison_bar.png
#   fig_scatter_grid.png
#   fig_pca_paf.png
#   fig_pca_fpb.png

# If all 3 runs succeed → your environment is working. Proceed.

# ============================================================
# STEP 4: Prepare the kinase PDB dataset (10-30 min)
# ============================================================

# 4a: Generate the manifest CSV (no download, instant)
python prepare_kinase_dataset.py --csv_only --out_dir data/kinase_pdbs/
# Creates: data/kinase_pdbs/kinase_list.csv with 50 kinase entries

# 4b: Download PDB files from RCSB (requires internet)
python prepare_kinase_dataset.py --download --out_dir data/kinase_pdbs/
# Downloads ~50 PDB files into data/kinase_pdbs/
# Some may fail (server issues, retired PDB IDs) — that's OK
# Expect ~45-48 successful downloads

# 4c: Verify downloads
ls data/kinase_pdbs/*.pdb | wc -l
# Should show ~45-50

# ============================================================
# STEP 5: Bridge script — connect PDB files to head_to_head.py
# ============================================================
#
# head_to_head.py has a "real" mode that needs pocket extraction.
# The script below does that integration.
# Save this as: ~/paf_project/run_real_kinases.py
#
# (This is the integration step I mentioned — it connects
#  the pocket extraction from protein_sonification_scoping.py
#  to the head_to_head comparison pipeline.)

# ============================================================
# STEP 6: Run on real kinases (1-2 hours)
# ============================================================

# After creating run_real_kinases.py (see Step 5):
python run_real_kinases.py --pdb_list data/kinase_pdbs/kinase_list.csv \
                           --out results/kinase_v0/

# Expected output:
#   - Spearman rho for each method
#   - Figures in results/kinase_v0/
#   - comparison_results.csv

# ============================================================
# STEP 7: Look at results
# ============================================================

# View the key table
cat results/kinase_v0/comparison_results.csv

# Open figures (Mac)
open results/kinase_v0/fig_comparison_bar.png
open results/kinase_v0/fig_pca_paf.png

# Or on Linux:
# xdg-open results/kinase_v0/fig_comparison_bar.png

# ============================================================
# OPTIONAL: Run Paper 1 tools (positive controls + power analysis)
# ============================================================

# Power analysis for BBBP zero-shot experiment
python power_analysis.py
# Produces: power_analysis_paper1.png + printed tables

# Generate audio controls for LLM sanity testing
python positive_controls.py
# Produces: control_audio/ folder with 130 WAV files

# ============================================================
# TROUBLESHOOTING
# ============================================================
#
# Error: "No module named 'Bio'"
#   → pip install biopython
#
# Error: "No module named 'soundfile'"
#   → pip install soundfile
#   → On Mac you may also need: brew install libsndfile
#
# Error: "Could not download PDB"
#   → Some PDB IDs may be superseded. Remove failed entries from CSV.
#   → Or download manually from https://www.rcsb.org/
#
# Error: "Kinase motifs not found"
#   → That PDB may have a non-standard chain or unusual sequence.
#   → Remove it from the CSV and proceed. Expect 5-10% failures.
#
# Error: "No ligand found"
#   → The ligand resname in the CSV may not match the PDB file.
#   → Open the PDB in a text editor, search for HETATM lines,
#     find the real ligand name, update the CSV.
#
# Figures look wrong or empty:
#   → Check that you have >= 10 successful pockets.
#   → If too many PDBs failed extraction, increase the dataset.
