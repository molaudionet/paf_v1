# Protein Acoustic Fingerprinting (PAF) v1

[![DOI: Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.18831908.svg)](https://doi.org/10.5281/zenodo.18831908)
[![DOI: Data](https://zenodo.org/badge/DOI/10.5281/zenodo.18881847.svg)](https://doi.org/10.5281/zenodo.18881847)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Deterministic spectral encoding of protein binding pockets via oscillatory superposition.**

---

## Overview

**Protein Acoustic Fingerprinting (PAF)** is a training-free, deterministic method for generating compact numerical embeddings of protein binding pockets. Instead of aggregating residue features through pooling or binning, PAF maps structural elements to localized oscillatory functions whose superposition produces interference patterns encoding collective pocket organization. Spectral transformation (FFT) yields fixed-length embedding vectors suitable for similarity search, classification, and machine learning.

PAF is the protein-pocket extension of **Molecular Acoustic Fingerprinting (MAF)**, our framework for small-molecule sonification ([companion paper](https://doi.org/10.5281/zenodo.18831908)). Together, MAF and PAF establish a unified audio modality for drug discovery — encoding both ligands and targets as waveforms natively compatible with pretrained audio AI.

### Key Features

- **Training-free**: No supervised learning or gradient descent required
- **Deterministic**: Same input always produces same output
- **Compact**: Fixed-length embeddings regardless of pocket size
- **Interpretable**: Parameters map to physicochemical properties
- **Fast**: ~18 ms per pocket on CPU
- **Audio-compatible**: Waveforms can leverage pretrained audio AI models (wav2vec 2.0, Whisper)

### What PAF Captures

| Method | Encodes | Loses |
|--------|---------|-------|
| **Mean Aggregation** | Composition (which residues) | Spatial organization |
| **Radial Histogram** | Distance-binned composition | Cross-feature relationships |
| **PAF (Ours)** | **Composition + Organization** | Explicit energetics |

> **Note:** PAF excels at tasks requiring sensitivity to **spatial organization** (e.g., kinase subfamily discrimination, conformational state detection). On tasks dominated by gross compositional differences, simple aggregation may perform comparably or better. PAF is best used as a complementary representation for selectivity profiling and structural organization analysis.

---

## Citation

If you use PAF in your research, please cite:

**Paper:**
```bibtex
@article{zhou2026spectral,
  title={Spectral Encoding of Protein Binding Pockets via Deterministic
         Oscillatory Superposition: Toward an Audio Modality for
         Molecular Representation},
  author={Zhou, Emily Rong and Zhou, Charles Jianping},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18831908}
}
```

**Data:**
```bibtex
@dataset{zhou2026pafdata,
  title={PAF v1: Precomputed Spectral Embeddings for Protein Binding Pockets},
  author={Zhou, Emily Rong and Zhou, Charles Jianping},
  publisher={Zenodo},
  year={2026},
  doi={10.5281/zenodo.18881847}
}
```

---

## Method Summary

```
PDB Structure → Pocket Extraction → Residue Features (30-dim)
                                         ↓
                            Oscillatory Mapping:
                            - Radial distance → Temporal localization (τ)
                            - Physicochemical → Frequency (ω)
                            - Angular → Modulation (γ)
                                         ↓
                            Superposition: s(t) = Σ sᵢ(t)
                                         ↓
                            FFT → Magnitude Spectrum
                                         ↓
                            Fixed-length Embedding (L2-normalized)
```

**Core Equation:**
```
sᵢ(t) = Aᵢ · exp(-(t - τᵢ)² / 2σ²) · cos(ωᵢt + φᵢ)
s(t) = Σᵢ sᵢ(t)
E = |FFT(s(t))|
```

---

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Biopython
- Matplotlib (for visualization)

### Install from Source

```bash
git clone https://github.com/molaudionet/paf_v1.git
cd paf_v1
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage

```python
from src.paf_core_v1 import PAFParams, extract_pocket, pocket_to_embedding

# Initialize parameters
params = PAFParams(
    pocket_radius_A=8.0,      # Pocket radius
    gamma_fm=0.15,            # Angular modulation strength
    sigma_t_s=0.06,           # Gaussian window width
    a_hyd=1.0,                # Hydrophobicity weight
    a_charge=1.0,             # Charge weight
    a_vol=0.0,                # Volume weight
)

# Extract pocket from PDB
pocket = extract_pocket(
    pdb_path="data/pdbs/1ATP.pdb",
    chain_id="A",
    ligand_resname="ATP",
    params=params,
)

# Generate embedding
spectrum, channel_names = pocket_to_embedding(pocket, params)

# spectrum is a (30, 256) numpy array — flatten and L2-normalize for similarity search
import numpy as np
embedding = spectrum.flatten()
embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
```

### Batch Processing

```python
import numpy as np
from pathlib import Path

pdb_files = list(Path("data/pdbs/").glob("*.pdb"))
embeddings = []

for pdb in pdb_files:
    pocket = extract_pocket(str(pdb), chain_id="A", ligand_resname="", params=params)
    spec, _ = pocket_to_embedding(pocket, params)
    emb = spec.flatten()
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    embeddings.append(emb)

embeddings = np.stack(embeddings)  # Shape: (N_pockets, 7680)
```

### Similarity Search

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise similarity matrix
S = cosine_similarity(embeddings)

# Find most similar pockets to query (index 0)
similar_indices = np.argsort(-S[0])[1:6]  # Top 5 neighbors
```

---

## Benchmarks

### Cross-Family Pocket Discrimination (N=79, 4 families)
*Results from preprint (Zenodo 18831908)*

| Method | 1-NN Accuracy | Cohen's d | p-value |
|--------|---------------|-----------|---------|
| **PAF (Ours)** | **0.772** | **0.748** | <10⁻¹⁵ |
| Radial Histogram | 0.671 | 0.317 | 6.1×10⁻¹⁴ |
| Mean Aggregation | 0.608 | 0.420 | <10⁻¹⁵ |
| Random Baseline | 0.250 | — | — |

### Kinase Subfamily Discrimination (N=50, 7 subfamilies)

| Method | 1-NN Accuracy | Cohen's d | p-value |
|--------|---------------|-----------|---------|
| **PAF (Ours)** | **0.42–0.50** | **0.49** | 2.6×10⁻⁷ |
| Radial Histogram | 0.30 | 0.05 | 0.60 |
| Mean Aggregation | 0.30 | 0.13 | 0.084 |

### Conformational State Sensitivity (DFG-in vs DFG-out, N=28)

| Method | 1-NN Accuracy | Cohen's d | p-value |
|--------|---------------|-----------|---------|
| **PAF (Ours)** | **0.571** | **0.254** | 0.014 |
| Radial Histogram | 0.643 | 0.156 | 0.132 |
| Mean Aggregation | 0.321 | -0.055 | 0.591 |

### Large-Scale Cross-Family (N=1168, 15 families)
*Internal benchmark showing task-dependent performance*

| Method | Accuracy | Cohen's d | Speed |
|--------|----------|-----------|-------|
| Mean Aggregation | **0.870** | **1.120** | 0.03 ms/pocket |
| Radial Histogram | 0.694 | 0.915 | 0.13 ms/pocket |
| **PAF (Ours)** | 0.685 | 0.189 | 17.6 ms/pocket |

> **Performance Note:** On broad cross-family tasks dominated by compositional differences, mean aggregation performs well. PAF excels on tasks requiring **spatial organization sensitivity** (kinase subfamilies, conformational states, selectivity profiling). Use PAF when composition alone is insufficient.

---

## Repository Structure

```
paf_v1/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore
│
├── src/                                # Source code
│   ├── paf_core_v1.py                  # Core PAF encoding engine
│   ├── compare_paf_vs_baselines_v1.py  # Head-to-head benchmark script
│   ├── spectral_encoder.py             # Batch encoding adapter
│   ├── curate_cross_family_dataset.py  # Cross-family dataset curation
│   ├── curate_dfg_dataset.py           # DFG conformation dataset curation
│   ├── run_cross_family_paf.py         # Cross-family experiment
│   ├── run_dfg_experiment.py           # DFG conformation experiment
│   ├── ligand_retrieval_v1.py          # Ligand–pocket retrieval experiment
│   ├── fix_cocrystal_csv.py            # Ligand name auto-correction utility
│   └── redownload_pdbs.py             # PDB re-download with HETATM verification
│
├── scripts/                            # Shell scripts for running experiments
│   ├── run_all.sh                      # Master pipeline script
│   ├── download_pdbs.sh                # Bulk PDB download
│   ├── cross.sh                        # Run cross-family experiment
│   ├── dfg.sh                          # Run DFG experiment
│   └── ligand_pocket.sh                # Run ligand retrieval experiment
│
├── data/                               # Datasets (small CSVs tracked; PDBs gitignored)
│   ├── kinase_list.csv                 # Example kinase manifest
│   ├── cross_family/                   # Cross-family dataset
│   │   └── cross_family_list.csv
│   ├── dfg/                            # DFG-in/DFG-out dataset
│   │   └── dfg_list.csv
│   └── pdbs/                           # Downloaded PDB files (gitignored)
│
├── results/                            # Experiment outputs (gitignored)
│   └── .gitkeep
│
├── figures/                            # Visualization outputs
│
└── notebooks/                          # Example Jupyter notebooks
```

---

## Precomputed Data

Precomputed embeddings for benchmark datasets are available on Zenodo:

| Dataset | Structures | DOI |
|---------|------------|-----|
| Cross-Family (79) | 79 pockets, 4 families | [10.5281/zenodo.18881847](https://doi.org/10.5281/zenodo.18881847) |
| Large-Scale (1168) | 1168 pockets, 15 families | [10.5281/zenodo.18881847](https://doi.org/10.5281/zenodo.18881847) |

Download and load:
```python
import numpy as np

data = np.load("cross_family_embeddings.npz", allow_pickle=True)
spectral = data["spectral"]   # PAF embeddings
mean = data["mean"]            # Mean aggregation baseline
radial = data["radial"]        # Radial histogram baseline
labels = data["labels"]        # Family labels
```

---

## Configuration

### Recommended Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pocket_radius_A` | 8.0 | Binding pocket radius (Å) |
| `gamma_fm` | 0.15 | Angular frequency modulation |
| `sigma_t_s` | 0.06 | Gaussian window width |
| `a_hyd` | 1.0 | Hydrophobicity amplitude weight |
| `a_charge` | 1.0 | Charge amplitude weight |
| `a_vol` | 0.0 | Volume amplitude weight |

### Ablation Insights

- **Angular modulation (γ > 0)**: Essential for subfamily discrimination
- **Charge + Hydrophobicity**: Dominant channels for kinase pockets
- **Local radius (8–10 Å)**: More discriminative than extended regions
- **Volume**: Minimal contribution in current benchmarks

---

## Applications

- **Pocket Similarity Search**: Rapid comparison across structural databases
- **Selectivity Profiling**: Identify off-target binding risks
- **Conformational State Detection**: DFG-in/out, active/inactive states
- **ML Preprocessing**: Deterministic input features for downstream models
- **Transfer Learning**: Feed waveforms to pretrained audio models (wav2vec 2.0, Whisper)

---

## Related Projects

| Project | Description | Link |
|---------|-------------|------|
| **MAF** | Molecular Acoustic Fingerprinting — small-molecule sonification for property prediction | [companion paper](https://doi.org/10.5281/zenodo.18831908) |
| **MolAudioNet** | Audio-based molecular AI platform | [github.com/molaudionet/molecularAI](https://github.com/molaudionet/molecularAI) |
| **Sound of Molecules** | Company website | [soundofmolecules.com](https://soundofmolecules.com) |

MAF encodes small molecules; PAF encodes protein pockets. Together they form a unified audio modality for drug discovery — representing both sides of the binding equation as waveforms compatible with pretrained speech AI.

---

## Intellectual Property

This software is released under the **MIT License** for academic and research use.

**Patent Status:**
- Provisional patent application filed February 2026
- Related granted patents: US 9,018,506; US 10,381,108
- Commercial licensing available for enterprise use

For commercial inquiries, contact: **info@soundofmolecules.com**

---

## Acknowledgments

This work is dedicated to:
- **Professor Richard B. Silverman** on his 80th birthday and 50 years at Northwestern University
- **Professor Philip E. Eaton** (1936–2023), whose wisdom that "good research changes people's minds" guides this work

Funding support from the **University of Illinois iVenture Accelerator**.

---

## Contact

| Role | Name | Email |
|------|------|-------|
| Corresponding Author | Charles Jianping Zhou | zhou@uchicago.edu |
| First Author | Emily Rong Zhou | erzhou2@illinois.edu |
| General Inquiries | Sound of Molecules LLC | info@soundofmolecules.com |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | March 2026 | Initial release with kinase + cross-family benchmarks |
| v1.1 | Coming | PDBbind integration, affinity prediction examples |

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Sound of Molecules LLC** | [soundofmolecules.com](https://soundofmolecules.com)

*Audio as a Molecular Modality for Drug Discovery*

</div>
