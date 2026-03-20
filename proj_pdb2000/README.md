# PAF Scale-Up Pipeline v2

**Protein Acoustic Fingerprinting at Scale: 1,489 structures × 15 families**

Emily Rong Zhou and Charles Jianping Zhou  
*Sound of Molecules LLC — March 2026*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18831908.svg)](https://doi.org/10.5281/zenodo.18831908)

---

## What this is

This is the automated scale-up pipeline that produces the main results reported in our paper. It queries RCSB and KLIFS APIs to curate datasets of ~1,500 protein–ligand co-crystal structures across 15 protein families, downloads PDB files in parallel, encodes all pockets using the real PAF engine (`paf_core_v1.py`), and evaluates against two aggregation baselines with full statistical analysis.

The pipeline runs end-to-end with a single command:

```bash
bash run_all.sh
```

---

## What changed (v1 → v2)

v1 used a reconstructed spectral encoder that differed from the real PAF in multiple ways, causing baselines to beat spectral encoding. v2 fixes all of these:

| Issue | v1 (broken) | v2 (fixed) |
|---|---|---|
| Encoder | Reconstructed from manuscript | Real `paf_core_v1.py` |
| Waveform samples | 512 | 96,000 (sr=16000 × 6s) |
| Frequency mapping | Linear ω₀ + β·f | Sigmoid [150, 2400] Hz |
| Geometry | Single-anchor (centroid) | Two-anchor (r1→time, r2→FM) |
| FFT bins | Linear | Log-spaced |
| Normalization | Global L2 | Per-channel sum |
| Phase | Feature-derived | Charge-sign (0 or π) |
| Oscillator | cos(ωt + φ) | sin(2πft + φ) with Gaussian window |
| Dimensionality | No PCA | PCA sweep (30–500 components) |
| DFG evaluation | Raw accuracy (misleading at 241:24) | Balanced accuracy + per-class recall |
| Parsing | Custom PDB parser | BioPython (handles edge cases) |

---

## Repository structure

```
paf_scaleup_v2/
├── paf_core_v1.py           # Real PAF encoder (YOU must place this here)
├── spectral_encoder.py      # Adapter wrapping paf_core_v1 for the pipeline
├── curate_datasets.py       # Step 1: RCSB + KLIFS API → CSV manifests
├── download_pdbs.py         # Step 2: Parallel PDB download from RCSB
├── run_experiments.py       # Step 3: Encode + evaluate + PCA sweep + stats
├── run_all.sh               # Master script (runs Steps 1–3)
├── README.md                # This file
│
├── data/                    # Created by pipeline
│   ├── cross_family_manifest.csv    # 15-family manifest (~1,500 entries)
│   ├── kinase_manifest.csv          # KLIFS kinase manifest (subfamily + DFG)
│   └── pdbs/                        # Downloaded PDB files
│
└── results/                 # Created by pipeline
    ├── cross_family_results.json    # Accuracy, Cohen's d, PCA sweep, p-values
    ├── cross_family_embeddings.npz  # Raw + PCA embeddings for downstream use
    ├── kinase_results.json          # Subfamily + DFG results with balanced accuracy
    └── timing_benchmark.json        # ms/pocket, µs/pair
```

---

## Requirements

- **Python:** 3.10+
- **OS:** macOS, Linux, or Windows
- **GPU:** Not required
- **Internet:** Required for Steps 1–2 (API queries + PDB downloads)
- **Disk:** ~2–3 GB for PDB files at full scale

### Python dependencies

```bash
pip install numpy requests biopython scikit-learn scipy matplotlib pandas
```

`scikit-learn` is strongly recommended — without it, PCA sweeps and balanced accuracy calculations are disabled.

### Critical prerequisite

**You must copy `paf_core_v1.py` into this directory before running.** The pipeline adapter (`spectral_encoder.py`) imports it at runtime.

```bash
cp /path/to/your/paf_core_v1.py .
```

---

## Quick start

```bash
# Full pipeline: curate → download → run experiments
bash run_all.sh

# Or start from a specific step:
bash run_all.sh --step 2     # Skip curation, start from download
bash run_all.sh --step 3     # Skip curation + download, just run experiments
```

---

## Pipeline: step by step

### Step 1 — Curate datasets (`curate_datasets.py`)

Queries three data sources to build CSV manifests:

**Cross-family dataset** queries the RCSB PDB Search API for co-crystal structures (X-ray, resolution ≤ 2.5 Å, with ligand) across 15 protein families: kinase, serine protease, metalloprotease, nuclear receptor, phosphodiesterase, proteasome, carbonic anhydrase, HSP90, cyclooxygenase, histone deacetylase, GPCR, aspartyl protease, phosphatase, bromodomain, and DHFR. Target: ~1,500 structures total.

**KLIFS kinase dataset** pulls kinase structures from the KLIFS API with DFG conformation and αC-helix annotations. Provides subfamily labels for within-kinase discrimination testing. Falls back to RCSB queries if the KLIFS API is unreachable.

**PDBbind dataset** (optional) parses the PDBbind refined set index for structures with binding affinity data. Requires a local copy of the PDBbind INDEX file.

```bash
# All datasets
python curate_datasets.py --task all --out_dir data/ --resolution 2.5

# Individual tasks
python curate_datasets.py --task cross_family --out data/cross_family_manifest.csv
python curate_datasets.py --task kinase_klifs --out data/kinase_manifest.csv
python curate_datasets.py --task pdbbind --pdbbind_index /path/to/INDEX_refined_data.2020 --out data/pdbbind_manifest.csv
```

### Step 2 — Download PDB files (`download_pdbs.py`)

Downloads PDB files from RCSB in parallel (8 workers by default). Skips files that already exist.

```bash
# From a manifest CSV
python download_pdbs.py --manifest data/cross_family_manifest.csv --out_dir data/pdbs/ --workers 8

# From explicit PDB IDs
python download_pdbs.py --pdb_ids 1ATP 2HYY 3CS9 --out_dir data/pdbs/

# Force re-download
python download_pdbs.py --manifest data/cross_family_manifest.csv --out_dir data/pdbs/ --overwrite
```

### Step 3 — Run experiments (`run_experiments.py`)

Encodes all pockets and evaluates three methods on the same extracted features.

```bash
# All experiments
python run_experiments.py \
    --experiment all \
    --manifest_dir data/ \
    --pdb_dir data/pdbs/ \
    --out_dir results/ \
    --permutations 10000

# Individual experiments
python run_experiments.py --experiment cross_family \
    --manifest data/cross_family_manifest.csv \
    --pdb_dir data/pdbs/ --out_dir results/

python run_experiments.py --experiment kinase \
    --manifest data/kinase_manifest.csv \
    --pdb_dir data/pdbs/ --out_dir results/

python run_experiments.py --experiment benchmark \
    --manifest data/cross_family_manifest.csv \
    --pdb_dir data/pdbs/ --out_dir results/
```

---

## The adapter: `spectral_encoder.py`

This module wraps `paf_core_v1.py` and provides the interface that `run_experiments.py` expects. It exposes three encoding methods that all share the same pocket extraction pass:

**`spectral_encode_from_pdb()`** — the real PAF encoder. Calls `paf_core_v1.embed_pocket()` to produce a 30×256 = 7,680-dimensional spectral embedding, then L2-normalizes for cosine similarity.

**`mean_aggregation_embedding()`** — baseline. Averages the 30-dimensional residue feature vectors across all pocket residues. No spatial information.

**`radial_histogram_embedding()`** — baseline. Bins residues into 16 radial shells and averages features per shell (16×30 = 480 dimensions). Spatial information via binning but no wave interference.

**`encode_all_methods()`** — the batch function used by experiments. Encodes every pocket once with `paf_core_v1.extract_pocket()`, then builds all three representations from the same extraction. This ensures fair comparison — identical input features, only the encoding strategy differs.

```python
from spectral_encoder import encode_all_methods

encoded = encode_all_methods(entries, pdb_dir="data/pdbs/")
# encoded["spectral"]  → np.ndarray (N, 7680)
# encoded["mean"]      → np.ndarray (N, 30)
# encoded["radial"]    → np.ndarray (N, 480)
```

---

## Experiments and evaluation

### Cross-family classification

Leave-one-out 1-nearest-neighbor classification across 15 families. For each pocket, its nearest neighbor (by cosine similarity) in the embedding space votes for the family label. This tests representational quality without a trained classifier.

**PCA sweep:** Spectral embeddings are 7,680-dimensional. At ~1,500 samples, 1-NN in 7,680-D suffers from distance concentration (the "curse of dimensionality"). The pipeline automatically sweeps PCA at [30, 50, 100, 200, 500] components, picks the setting that maximizes accuracy, and reports results at the optimal dimensionality.

**Statistical rigor:** 10,000-permutation test for p-values. Cohen's *d* on within-class vs. between-class cosine similarity distributions. Per-class accuracy breakdown.

### Kinase analysis

Two sub-experiments on the KLIFS kinase dataset:

**Subfamily discrimination** — can PAF distinguish kinase subfamilies (TK, CMGC, AGC, STE, etc.) that share the canonical kinase fold? Only subfamilies with ≥3 representatives are included.

**DFG conformation detection** — can PAF separate DFG-in (active) from DFG-out (inactive) conformations? The KLIFS dataset has a severe class imbalance (~241 DFG-in vs. ~24 DFG-out). Raw accuracy is therefore misleading — predicting "DFG-in" for everything gives 91%. The pipeline reports **balanced accuracy** (mean of per-class recall) and **per-class recall** to give an honest assessment.

### Timing benchmark

Measures encoding speed (ms/pocket) and pairwise similarity throughput (µs/pair) on a 200-pocket subset.

---

## Understanding the outputs

### `cross_family_results.json`

```json
{
  "spectral": {
    "accuracy": 0.857,
    "balanced_accuracy": 0.804,
    "cohens_d": 1.420,
    "p_value": 0.0,
    "fold_vs_random": 12.85,
    "pca_components": 100,
    "n_structures": 1489,
    "n_families": 15,
    "per_class_accuracy": { "kinase": {"recall": 0.92, "n": 293}, ... }
  },
  "mean": { ... },
  "radial": { ... },
  "spectral_pca_sweep": {
    "no_pca": {"n_components": 7680, "accuracy": 0.82, ...},
    "pca_50":  {"n_components": 50,   "accuracy": 0.84, ...},
    "pca_100": {"n_components": 100,  "accuracy": 0.86, ...},
    ...
  }
}
```

### `kinase_results.json`

Contains `subfamily_spectral`, `subfamily_mean`, `subfamily_radial` (subfamily classification), and `dfg_spectral`, `dfg_mean`, `dfg_radial` (DFG state detection) with balanced accuracy and per-class recall.

### `cross_family_embeddings.npz`

```python
import numpy as np
data = np.load("results/cross_family_embeddings.npz", allow_pickle=True)
spectral = data["spectral"]   # (N, 7680)
mean = data["mean"]            # (N, 30)
radial = data["radial"]        # (N, 480)
labels = data["labels"]        # (N,) string family labels
pdb_ids = data["pdb_ids"]      # (N,) PDB IDs
```

### `timing_benchmark.json`

```json
{
  "encoding": {"n_pockets": 185, "ms_per_pocket": 950.0},
  "pairwise": {"n_pairs": 17020, "us_per_pair": 0.15}
}
```

---

## Interpreting results

The key comparison is the **d-ratio** — how many times larger PAF's Cohen's *d* is compared to the baselines:

| Metric | What it tells you |
|---|---|
| **Cohen's *d*** | Separation between within-class and between-class similarity. Higher = better discrimination |
| **d-ratio (PAF / mean)** | How much structural organization the wave transform captures beyond composition alone |
| **Balanced accuracy** | Classification performance corrected for class imbalance |
| **PCA optimal dim** | Where the signal lives. If PCA(100) >> PCA(7680), raw embedding has distance concentration |
| **DFG per-class recall** | Critical for the minority class. 92.6% recall on 27 DFG-out structures is the paper's headline |

---

## Troubleshooting

| Error | Solution |
|---|---|
| `paf_core_v1.py not found` | Copy your real PAF encoder into this directory |
| `No module named 'requests'` | `pip install requests` |
| `No module named 'sklearn'` | `pip install scikit-learn` (PCA sweep + balanced accuracy require it) |
| `RCSB search error` | API may be rate-limited; wait a few minutes and retry |
| `KLIFS API error` | KLIFS may be down; the script falls back to RCSB queries |
| `No pockets encoded` | Check that PDB files exist in `data/pdbs/` and aren't empty |
| Most families show 0 accuracy | Likely `paf_core_v1.py` is missing or a different version |

---

## Extending the pipeline

### Adding a new protein family

Edit the `PROTEIN_FAMILIES` dict in `curate_datasets.py`:

```python
PROTEIN_FAMILIES["my_new_family"] = {
    "search_terms": ["my protein family", "alternative name"],
    "target_n": 100,
}
```

Then re-run from Step 1: `bash run_all.sh`

### Using your own CSV manifest

Any CSV with a `pdb_id` column and a `family` column works:

```bash
python download_pdbs.py --manifest my_data.csv --out_dir data/pdbs/
python run_experiments.py --experiment cross_family --manifest my_data.csv --pdb_dir data/pdbs/ --out_dir results/
```

### Adjusting PAF parameters

Edit `spectral_encoder.py` to pass custom `PAFParams`:

```python
from paf_core_v1 import PAFParams

custom_params = PAFParams(
    pocket_radius_A=8.0,    # default: 10.0
    gamma_fm=0.20,          # default: 0.15
    sigma_t_s=0.06,         # default: 0.04
)
```

---

## Relationship to paf_v1 repository

This pipeline (v2) is the **production evaluation system** that generated the paper's main results at scale (1,489 structures, 15 families, 10,000-permutation statistics). The [paf_v1](https://github.com/molaudionet/paf_v1) repository contains the earlier development pipeline with hand-curated datasets of 28–79 structures, more detailed experiment scripts (DFG paired analysis, ligand retrieval), and step-by-step execution guides.

| | paf_v1 (development) | paf_scaleup_v2 (production) |
|---|---|---|
| **Scale** | 28–79 structures | ~1,500 structures |
| **Dataset curation** | Hand-curated PDB lists | Automated RCSB/KLIFS API queries |
| **Families** | 4 (cross-family) + kinase-only | 15 families |
| **PDB download** | Sequential | 8-worker parallel |
| **Evaluation** | Cohen's *d*, 1-NN accuracy | + PCA sweep, balanced accuracy, permutation tests |
| **DFG analysis** | Paired design (28 structures) | KLIFS annotations (~265 structures) |
| **Use case** | Understanding, debugging, iteration | Final paper numbers |

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
