# PAF Scale-Up Pipeline v2

**Protein Acoustic Fingerprinting at Scale**
Sound of Molecules LLC — March 2026

## What Changed (v1 → v2)

v1 used a reconstructed spectral encoder that differed from the real PAF in 7+ ways,
causing baselines to beat spectral encoding. v2 fixes this:

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
| DFG evaluation | Raw accuracy (misleading at 241 vs 24) | Balanced accuracy + per-class recall |
| Parsing | Custom PDB parser | BioPython (handles edge cases) |

## Setup

```bash
pip install numpy requests biopython scikit-learn

# Copy your real encoder into this directory:
cp /path/to/paf_core_v1.py .
```

## Quick Start

```bash
# Full pipeline (skip steps 1-2 if you already have data/pdbs/)
bash run_all.sh

# Or just re-run experiments with real encoder:
bash run_all.sh --step 3
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `curate_datasets.py` | Query RCSB + KLIFS APIs → CSV manifests |
| 2 | `download_pdbs.py` | Parallel PDB download from RCSB |
| 3 | `run_experiments.py` | Encode + evaluate + PCA sweep + stats |

## Modules

| File | Role |
|---|---|
| `paf_core_v1.py` | **YOUR** real PAF encoder (not included — copy it here) |
| `spectral_encoder.py` | Adapter wrapping paf_core_v1 for the pipeline |
| `run_experiments.py` | Experiments: LOO 1-NN, Cohen's d, permutation tests, PCA sweep |
| `curate_datasets.py` | RCSB + KLIFS curation for 15 families + kinase subfamilies |
| `download_pdbs.py` | Parallel downloader |
| `run_all.sh` | Master script |

## Key Fixes

**PCA sweep**: Spectral embeddings are 7,680-dimensional (30 channels × 256 bins).
With 15 families, 1-NN in 7,680-D suffers from distance concentration. The pipeline
now sweeps PCA at [30, 50, 100, 200, 500] components and picks the best automatically.

**Balanced accuracy**: DFG analysis has 241 in vs 24 out. Raw accuracy of 91% just means
"predict majority class." Balanced accuracy (mean of per-class recall) and stratified
permutation tests give honest evaluation.

**Single extraction pass**: All three methods (spectral, mean, radial) share the same
pocket extraction from paf_core_v1, ensuring identical input features for fair comparison.

## Expected Output

```
results/
├── cross_family_results.json      # Includes PCA sweep results
├── cross_family_embeddings.npz    # Raw + PCA embeddings
├── kinase_results.json            # Subfamily + DFG with balanced accuracy
└── timing_benchmark.json          # ms/pocket, μs/pair
```
