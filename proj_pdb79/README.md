# Protein Acoustic Fingerprinting (PAF)

**Audio as a molecular modality: oscillatory spectral encoding of protein binding pockets**

Emily Rong Zhou and Charles Jianping Zhou  
*Sound of Molecules LLC*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18831908.svg)](https://doi.org/10.5281/zenodo.18831908)

---

## Overview

Protein Acoustic Fingerprinting (PAF) is a deterministic, training-free method that converts protein binding pocket geometry into spectral fingerprints. Each pocket residue is mapped to a localized oscillatory function — a Gaussian-windowed sinusoid — whose temporal position encodes radial distance from the ligand, whose frequency encodes physicochemical identity, and whose phase encodes charge sign. These oscillators are summed by linear superposition, producing a composite waveform in which constructive and destructive interference creates spectral structure that depends on the *spatial arrangement* of chemical properties, not merely their inventory.

Across 1,489 co-crystal structures spanning 15 protein families, PAF achieves **85.7% classification accuracy** with a **3.0-fold effect-size improvement** (Cohen's *d* = 1.42 vs. 0.47) over aggregation baselines using identical input features. Within-family kinase subfamily discrimination reaches **78.8%** across 42 subfamilies, and DFG conformational state detection achieves **92.6% minority-class recall**.

The waveform is sampled at 16 kHz — natively compatible with self-supervised speech models (wav2vec 2.0, HuBERT) — positioning pocket analysis to leverage the rapid advances in audio AI without task-specific training.

---

## Repository structure

```
paf_v1/
│
├── README.md
│
├── ── Core engine ──────────────────────────────────────────
├── paf_core_v1.py                      # PAF embedding engine (pocket extraction + spectral projection)
│
├── ── Three main experiments ───────────────────────────────
├── Experiment 1: Cross-family classification
│   ├── curate_cross_family_dataset.py   # Curate 79-structure, 4-family dataset + download PDBs
│   ├── run_cross_family_paf.py          # Run PAF + baselines on cross-family dataset
│   ├── patch_cross_family_failures.py   # Fix extraction failures (chain/ligand mismatches)
│   └── cross.sh                         # One-liner to run the full cross-family experiment
│
├── Experiment 2: DFG conformation detection
│   ├── curate_dfg_dataset.py            # Curate paired DFG-in/DFG-out kinase dataset (28 structures)
│   ├── run_dfg_experiment.py            # Binary classification + paired analysis
│   └── dfg.sh                           # One-liner to run the full DFG experiment
│
├── Experiment 3: Ligand–pocket retrieval
│   ├── ligand_retrieval_v1.py           # Retrieval evaluation (AUROC, MRR, Top-k)
│   ├── create_cocrystal_pairs.sh        # Generate co-crystal pair CSV from kinase list
│   └── ligand_pocket.sh                 # One-liner to run ligand retrieval
│
├── ── Head-to-head comparison (general) ────────────────────
├── compare_paf_vs_baselines_v1.py       # PAF vs. FP-A/FP-B/FP-C on any CSV dataset
│
├── ── Data preparation utilities ───────────────────────────
├── prepare_kinase_dataset.py            # Curate 50-kinase manifest + PDB downloader
├── download_pdbs.sh                     # Bulk PDB download via curl
├── redownload_pdbs.py                   # Re-download PDBs with HETATM verification
├── fix_cocrystal_csv.py                 # Auto-fix wrong ligand_resname in CSVs
│
├── ── Setup & guides ───────────────────────────────────────
├── setup.sh                             # Create venv + install dependencies
├── HOW_TO_RUN.sh                        # Complete step-by-step execution guide
├── execution_guide.py                   # Decision tree + interpretation guide (printable)
├── debug.sh                             # Audit PDB-vs-CSV consistency
│
├── ── Earlier pipeline (kinase-only v0) ────────────────────
├── run_real_kinases.py                  # Bridge script: PDB → pocket extraction → comparison
├── head_to_head.py                      # Synthetic + real pocket comparison
├── raw_fingerprint_baseline.py          # 4 baseline fingerprint ablation methods
│
├── data/
│   ├── cross_family/                    # Cross-family dataset (79 structures, 4 families)
│   │   ├── pdbs/                        # Downloaded PDB files
│   │   └── cross_family_list.csv        # Manifest
│   ├── dfg/                             # DFG-in/DFG-out dataset (28 structures)
│   │   ├── pdbs/
│   │   └── dfg_list.csv
│   ├── kinase_pdbs/                     # Original 50-kinase dataset
│   │   └── kinase_list.csv
│   └── cocrystal_pairs.csv              # For ligand retrieval
│
└── results/
    ├── cross_family/                    # Cross-family experiment outputs
    ├── dfg/                             # DFG experiment outputs
    ├── ligand_retrieval_v1/             # Ligand retrieval outputs
    └── kinase_v0/                       # Original kinase comparison outputs
```

---

## Requirements

- **Python:** 3.10 or later
- **OS:** macOS, Linux, or Windows (tested on Mac Mini M4)
- **GPU:** Not required
- **Disk:** ~500 MB for PDB files and results
- **Internet:** Required only for PDB downloads

### Python dependencies

```bash
pip install numpy scipy matplotlib pandas biopython soundfile
```

Optional (for ligand retrieval with full atom featurization):

```bash
pip install rdkit requests
```

---

## Quick start

```bash
# 1. Clone and set up environment
git clone https://github.com/molaudionet/paf_v1.git
cd paf_v1
bash setup.sh
source .venv/bin/activate

# 2. Run Experiment 1 (cross-family classification)
bash curate.sh                      # Downloads 79 PDBs from RCSB
bash cross.sh                       # Runs PAF vs baselines

# 3. View results
cat results/cross_family/comparison_results.csv
open results/cross_family/fig_pca_PAF_v1.png
```

---

## The core engine: `paf_core_v1.py`

This is the heart of the repository. It provides two main functions:

**`extract_pocket(pdb_path, chain_id, ligand_resname, params)`** parses a PDB file and extracts the binding pocket. It uses a three-tier strategy for finding the pocket center: (1) ligand centroid if a co-crystallized ligand is present, (2) kinase motif anchors (VAIK/HRD/DFG) as a fallback for kinase structures without ligands, or (3) geometric chain center as a last resort. Each residue within the pocket radius is annotated with a 30-dimensional feature vector consisting of 5 physicochemical properties (charge, hydrophobicity, H-bond capacity, aromaticity, sidechain volume), flexibility from B-factors, contact density, 3 secondary structure indicators (helix/sheet/coil from DSSP), and a 20-dimensional BLOSUM62 substitution profile.

**`pocket_to_embedding(pocket, params)`** projects the extracted pocket into spectral space. Each of the 30 feature channels produces an independent waveform at 16 kHz for 6 seconds. Radial distance from the ligand maps to temporal position; physicochemical features map to frequency (150–2400 Hz via sigmoid); charge sign maps to phase (0 or π). A secondary geometric anchor modulates frequency to break radial degeneracy. The composite waveform is FFT-transformed and binned into 256 log-spaced frequency bins, yielding a 30×256 = 7,680-dimensional embedding.

**`embed_pocket(pdb_path, chain, ligand_resname, ...)`** is the public API wrapper used by downstream scripts. It accepts individual parameters (radius, gamma_fm, sigma_t, etc.) and returns `(E_mag, meta_dict)`.

### Key parameters (`PAFParams`)

| Parameter | Default | Description |
|---|---|---|
| `pocket_radius_A` | 10.0 | Pocket extraction radius (Å from ligand centroid) |
| `sr` | 16000 | Sampling rate (Hz) |
| `duration_s` | 6.0 | Waveform duration (seconds) |
| `fmin_hz` / `fmax_hz` | 150 / 2400 | Frequency range for feature-to-frequency mapping |
| `sigma_t_s` | 0.040 | Gaussian window width (seconds) |
| `gamma_fm` | 0.15 | Angular frequency modulation depth (second anchor) |
| `n_bins` | 256 | Number of log-spaced frequency bins |
| `a_hyd` / `a_charge` / `a_vol` | 1.0 / 1.0 / 0.5 | Feature weights for base frequency computation |

---

## Experiment 1: Cross-family classification

Tests whether PAF can distinguish binding pockets across 4 structurally diverse protein families (79 structures total): kinases (23), serine proteases (18), metalloproteases (19), and nuclear receptors (19).

### Run

```bash
# Step 1: Curate dataset + download PDBs (~5 min)
python curate_cross_family_dataset.py --out data/cross_family/ --download

# Step 2: Run PAF vs baselines (~30 min)
python run_cross_family_paf.py \
    --csv data/cross_family/cross_family_list.csv \
    --out results/cross_family/ \
    --radius 10 --gamma_fm 0.15 --sigma_t 0.04
```

Or simply:

```bash
bash curate.sh
bash cross.sh
```

### Fixing extraction failures

Some PDB entries may have chain or ligand mismatches. If `extraction_failures.csv` reports failures:

```bash
python patch_cross_family_failures.py \
    --csv data/cross_family/cross_family_list.csv \
    --pdb_dir data/cross_family/pdbs/ \
    --download
```

This replaces known problematic entries (e.g. 1FPC, 1LQE, 3TGI, 3TJQ) with verified alternatives and re-downloads the replacement PDB files.

### Outputs

| File | Description |
|---|---|
| `comparison_results.csv` | Accuracy and Cohen's *d* for PAF vs. FP-A/FP-B/FP-C |
| `fig_comparison_bar.png` | Bar chart comparing all methods |
| `fig_pca_PAF_v1.png` | PCA plot colored by protein family |
| `fig_within_between_*.png` | Within-family vs. between-family similarity distributions |
| `summary_*.json` | Per-method detailed metrics |
| `emb_*.npz` | Saved embeddings for downstream reuse |

---

## Experiment 2: DFG conformation detection

Tests whether PAF can distinguish active (DFG-in) from inactive (DFG-out) kinase conformations. This is the most stringent test: all structures share the canonical kinase fold and differ only in the orientation of the DFG activation loop.

The dataset uses a **paired design** — the same kinase in both DFG-in and DFG-out states (e.g. ABL + dasatinib vs. ABL + imatinib) — so any separation must come from conformational state, not sequence or fold differences.

### Dataset

28 structures total: 14 DFG-in and 14 DFG-out, spanning 10 paired kinases (ABL, p38α, B-RAF, c-KIT, EGFR, SRC, VEGFR2, CDK2, FGFR1, LCK, MET) plus unpaired entries for additional statistical power (Aurora A, PDGFRβ, CSF1R, IGF1R, JAK2, FLT3).

### Run

```bash
# Step 1: Curate + download (~3 min)
python curate_dfg_dataset.py --out data/dfg/ --download

# Step 2: Run the experiment (~20 min)
python run_dfg_experiment.py \
    --csv data/dfg/dfg_list.csv \
    --out results/dfg/ \
    --radius 10 --gamma_fm 0.15 --sigma_t 0.04
```

Or simply:

```bash
bash dfg.sh
```

### Outputs

| File | Description |
|---|---|
| `comparison_results.csv` | PAF vs. baselines: 1-NN accuracy, Cohen's *d* for DFG state |
| `fig_pca_PAF_v1.png` | PCA colored by DFG state (should show separation) |
| `fig_within_between_*.png` | Within-state vs. between-state similarity |
| `paired_analysis.json` | Per-kinase DFG-in vs. DFG-out embedding distances |
| `extraction_failures.csv` | Any PDB entries that failed pocket extraction |

### Interpretation

The paired analysis is the key result. For each kinase that has both DFG-in and DFG-out structures, the script measures: (a) the PAF embedding distance between the two states of the *same* kinase vs. (b) the distance between different kinases in the *same* state. If PAF captures conformational state, distance (a) should exceed distance (b).

---

## Experiment 3: Ligand–pocket retrieval

Tests whether PAF pocket embeddings can retrieve the cognate (co-crystallized) ligand from a pool of decoys — a direct test of pocket–ligand compatibility encoding.

### Prerequisites

- `paf_core_v1.py` must be importable (same directory or on `PYTHONPATH`)
- RDKit recommended for full ligand featurization (`pip install rdkit`); falls back to coordinate-only features if unavailable

### Run

```bash
# Step 1: Generate co-crystal pair manifest
bash create_cocrystal_pairs.sh

# Step 2: Run retrieval
python ligand_retrieval_v1.py \
    --pairs_csv data/cocrystal_pairs.csv \
    --pdb_dir . \
    --out results/ligand_retrieval_v1 \
    --radius 10.0 \
    --gamma_fm 0.15 \
    --sigma_t 0.04 \
    --easy_negatives 50 \
    --hard_negatives 50
```

Or simply:

```bash
bash create_cocrystal_pairs.sh
bash ligand_pocket.sh
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--pairs_csv` | *(required)* | CSV with `pdb_path`, `protein_chain`, `ligand_resname` columns |
| `--pdb_dir` | `./data/pdbs` | Base directory for resolving relative PDB paths |
| `--radius` | 10.0 | Pocket extraction radius (Å) |
| `--gamma_fm` | 0.15 | Angular frequency modulation depth |
| `--sigma_t` | 0.04 | Gaussian window width (s) |
| `--a_hyd` / `--a_charge` / `--a_vol` | 1.0 / 1.0 / 0.5 | Feature weights |
| `--K` | 256 | FFT bins for ligand embedding |
| `--easy_negatives` | 50 | Cross-family negative decoys per query |
| `--hard_negatives` | 50 | Within-family negative decoys per query |
| `--skip_crossspec` | *(flag)* | Disable cross-spectrum scoring |

### Outputs

| File | Description |
|---|---|
| `retrieval_table.csv` | Per-structure ranks and AUROC |
| `metrics_easy.json` | Aggregate metrics vs. cross-family negatives |
| `metrics_hard.json` | Aggregate metrics vs. within-family negatives |
| `auroc_curves.png` | AUROC bar chart |
| `mrr_by_family.png` | Mean Reciprocal Rank by protein family |
| `summary_onepage.md` | Copy-paste-ready summary for paper supplementary |

---

## General head-to-head comparison: `compare_paf_vs_baselines_v1.py`

A flexible script that runs PAF against three non-wave baselines on any CSV dataset. Use this for custom datasets or parameter sweeps.

```bash
python compare_paf_vs_baselines_v1.py \
    --kinase_csv data/cross_family/cross_family_list.csv \
    --out results/compare_v1_best/ \
    --radius 8 --gamma_fm 0.15 --sigma_t 0.06 \
    --a_hyd 1 --a_charge 1 --a_vol 0
```

### Baselines (use identical 30-d residue features)

| Method | Dimension | Strategy |
|---|---|---|
| **PAF** | 30 × 256 = 7,680 | Wave superposition + spectral transform |
| **FP-A (mean)** | 30 | Mean of residue feature vectors (no spatial info) |
| **FP-B (radial histogram)** | 16 × 30 = 480 | Radial shell binning (spatial but no interference) |
| **FP-C (sorted concat)** | 60 × 30 = 1,800 | Residues sorted by distance, concatenated |

This structure isolates the contribution of wave superposition: all methods see the same features, but only PAF encodes them through interference.

---

## Data preparation utilities

### `fix_cocrystal_csv.py`

Scans PDB files and auto-corrects wrong `ligand_resname` values in a CSV by finding the actual largest non-solvent HETATM group in each file.

```bash
python fix_cocrystal_csv.py \
    --csv data/cocrystal_pairs.csv \
    --pdb_dir . \
    --out data/cocrystal_pairs_fixed.csv
```

### `redownload_pdbs.py`

Diagnoses and re-downloads PDB files to ensure they contain full HETATM records (the most common failure mode).

```bash
# Diagnose only (no downloads)
python redownload_pdbs.py --csv data/cocrystal_pairs.csv --diagnose-only --pdb_dir .

# Re-download missing/broken files
python redownload_pdbs.py --csv data/cocrystal_pairs.csv --outdir ./data/pdbs
```

### `prepare_kinase_dataset.py`

Generates the original 50-kinase manifest and optionally downloads PDB files.

```bash
python prepare_kinase_dataset.py --csv_only --out_dir data/kinase_pdbs/
python prepare_kinase_dataset.py --download --out_dir data/kinase_pdbs/
```

---

## Interpreting results

Open `comparison_results.csv` from any experiment and compare:

1. **PAF** — 1-NN accuracy and Cohen's *d*
2. **FP-A (mean)** — composition-only floor (no spatial information)
3. **FP-B (radial histogram)** — spatial information via binning but no interference
4. **FP-C (sorted concat)** — ordered spatial features but no wave encoding

### Decision tree

| Outcome | Interpretation | Next step |
|---|---|---|
| PAF *d* > 1.0 and PAF beats all baselines | Wave superposition captures organizational info that aggregation destroys | Publish; figures are paper-ready |
| PAF *d* > 0.5 but PAF ≈ FP-B | Radial binning is the signal; FFT adds marginal value | Vary σ_t (20–80 ms), add angular info, or try atom-level features |
| All methods *d* < 0.3 | 30-d residue features are insufficient | Enrich with sequence conservation, atom-level encoding, or external descriptors |
| FP-A beats spatial methods | Composition dominates; spatial info adds noise | Reduce feature dimensionality or use coarser pocket definitions |

Inspect `fig_pca_*.png` — protein families/states should form visually distinct clusters in PAF embedding space.

### Parameter sensitivity

For promising results, sweep these parameters and report a sensitivity table:

```
Pocket radius:     R  = 6, 8, 10, 12 Å
Window width:      σt = 0.02, 0.04, 0.06, 0.08 s
FM modulation:     γ  = 0.0, 0.05, 0.10, 0.15, 0.20
Feature weights:   a_hyd, a_charge, a_vol
```

---

## Troubleshooting

| Error | Solution |
|---|---|
| `No module named 'Bio'` | `pip install biopython` |
| `No module named 'soundfile'` | `pip install soundfile` (macOS: also `brew install libsndfile`) |
| `No module named 'rdkit'` | `pip install rdkit` (only needed for `ligand_retrieval_v1.py`) |
| `No module named 'requests'` | `pip install requests` (needed for dataset curation scripts) |
| `Cannot import paf_core_v1` | Ensure `paf_core_v1.py` is in the working directory or on `PYTHONPATH` |
| `Chain 'X' not found` | Open PDB, check available chain IDs in ATOM records, update CSV |
| `No HETATM coords found` | PDB may be truncated. Run `redownload_pdbs.py` to re-fetch from RCSB |
| `No ligand found` | Ligand resname mismatch. Run `fix_cocrystal_csv.py` to auto-correct |
| `No ligand centroid, no kinase motifs` | Structure lacks both ligand and recognizable motifs; falls back to geometric center |
| Figures look empty | Need ≥10 successful pocket extractions. Check `extraction_failures.csv` |
| `Could not download PDB` | PDB ID may be superseded. Remove from CSV or download manually from [rcsb.org](https://www.rcsb.org/) |

---

## Data availability

- **PDB structures:** [RCSB Protein Data Bank](https://www.rcsb.org/)
- **Kinase annotations:** [KLIFS database](https://klifs.net/)
- **Curated datasets and methodology:** [Zenodo archive (DOI: 10.5281/zenodo.18831908)](https://doi.org/10.5281/zenodo.18831908)

---

## Citation

```bibtex
@article{zhou2026paf,
  title   = {Audio as a molecular modality: oscillatory spectral encoding
             of protein binding pockets},
  author  = {Zhou, Emily Rong and Zhou, Charles Jianping},
  journal = {Nature Computational Science},
  year    = {2026},
  note    = {Brief Communication, submitted}
}

@misc{zhou2026paf_zenodo,
  title     = {Spectral encoding of protein binding pockets via deterministic
               oscillatory superposition},
  author    = {Zhou, Emily Rong and Zhou, Charles Jianping},
  year      = {2026},
  doi       = {10.5281/zenodo.18831908},
  publisher = {Zenodo}
}
```

---

## Patents

- US Patent 9,018,506 — System and method for creating audible sound representations of atoms and molecules
- US Patent 10,381,108 — Related molecular sonification technology
- Provisional patent filed February 2026 covering the PAF methods described herein

---

## License

Copyright © 2026 Sound of Molecules LLC. All rights reserved.

Contact: zhou@uchicago.edu
