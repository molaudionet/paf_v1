# Pocket Acoustic Fingerprinting (PAF): A Wave-Domain Representation for Protein Binding Pockets

## Abstract

We introduce Pocket Acoustic Fingerprinting (PAF), a deterministic wave-domain transformation that converts protein binding pocket structure into fixed-length spectral embeddings. Rather than aggregating residue features directly in spatial domain, PAF maps pocket residues to oscillatory basis functions and derives embeddings via Fourier transform. Without any supervised training, PAF achieves 77.2% family-level classification accuracy across four structurally diverse protein families (79 structures; Cohen's d = 0.75, p ≈ 0), substantially outperforming both mean aggregation (60.8%) and radial histogram baselines (67.1%) that use identical residue features. Within the kinase family, PAF discriminates subfamilies at 42–50% leave-one-out accuracy (d ≈ 0.49, p < 10⁻⁶), and a controlled baseline comparison isolates the Fourier transformation as the active ingredient providing discriminative power. In a DFG-in versus DFG-out conformational experiment on 28 kinase structures, PAF detects pocket rearrangements with magnitude proportional to known structural change, achieving 57.1% state classification (d = 0.25, p = 0.014) while revealing that conformational sensitivity varies per kinase in a biologically interpretable pattern. These results establish PAF as a compact, interpretable, and training-free structural embedding that captures both evolutionary and conformational organization in protein binding pockets.

---

## 1. Introduction

Protein binding pockets are typically represented geometrically: as distance matrices, graph topologies, surface meshes, 3D voxel grids, or learned neural embeddings. These approaches have proven powerful but share a common design principle — they aggregate structural features directly in spatial domain.

PAF asks a different question: does transforming structural information into frequency space reveal discriminative organization not captured by spatial aggregation alone?

The conceptual foundation draws from experimental spectroscopy. In NMR, IR, and ESR, structural and electronic information is not interpreted from spatial coordinates but from frequency-domain signatures that reflect local environments and collective interactions. Chemical shifts, vibrational modes, and hyperfine splittings encode structure through spectral transformation. PAF applies an analogous signal-processing framework to protein pockets: residues become oscillators, their superposition generates a composite signal, and Fourier analysis yields a spectral embedding.

Importantly, PAF does not posit physical resonance in proteins. It leverages spectral transformation as an information-theoretic aggregation strategy that may capture higher-order interference patterns less apparent in direct geometric aggregation.

---

## 2. Method

### 2.1 Residue-to-Oscillator Mapping

For a binding pocket defined within radius R of a ligand-derived center, each residue i is mapped to an oscillatory basis function:

- **Temporal localization**: Radial distance from pocket center defines a Gaussian envelope centered at tᵢ, with width σ_t controlling spatial resolution.
- **Base frequency**: Physicochemical features (charge, hydrophobicity, hydrogen bonding capacity, aromaticity, volume) determine oscillation frequency for each feature channel.
- **Frequency modulation**: Angular position relative to a secondary structural anchor introduces frequency modulation with strength γ, encoding 3D directional information beyond radial distance alone.
- **Amplitude weighting**: Feature-specific weights (a_hyd, a_charge, a_vol) control channel importance.

### 2.2 Superposition and Fourier Transform

All residue oscillators are summed to produce a composite time-domain signal per feature channel. The magnitude of the discrete Fourier transform of each channel constitutes the embedding. The full PAF vector is the concatenation across all channels.

### 2.3 Residue Features

Each residue is represented by a 30-dimensional vector: physicochemical properties (charge, hydrophobicity, hydrogen bonding, aromaticity, volume; 5 dimensions), backbone flexibility (1), contact density (1), secondary structure one-hot encoding (helix, sheet, coil; 3), and BLOSUM62 substitution profile (20).

### 2.4 Parameters

The primary configuration used throughout: pocket radius = 10 Å, γ_fm = 0.15, σ_t = 0.04, weights = [1, 1, 0.5] for hydrophobicity, charge, and volume respectively.

---

## 3. Experiments and Results

### 3.1 Cross-Family Generalization (Experiment 1)

**Objective**: Test whether PAF separates structurally diverse protein families.

**Dataset**: 79 co-crystal structures across four families — kinases (24), serine proteases (18), metalloproteases (18), and nuclear receptors (19). Each entry is a unique protein target with a bound small-molecule ligand. Families span distinct fold types: kinase domains, trypsin-like and subtilisin-like folds, zinc-dependent catalytic domains, and α-helical ligand-binding domains.

**Baselines**: Two non-wave controls using identical 30-dimensional residue features — FP-A (mean aggregation, destroys spatial information) and FP-B (radial histogram with 16 shells, preserves radial geometry but no wave transform).

**Results**:

| Method      | 1-NN  | 3-NN  | Cohen's d | p-value   |
|-------------|-------|-------|-----------|-----------|
| PAF v1      | 0.772 | 0.722 | 0.748     | < 10⁻¹⁵  |
| FP-B radial | 0.671 | 0.633 | 0.317     | 6.1×10⁻¹⁴ |
| FP-A mean   | 0.608 | 0.684 | 0.420     | < 10⁻¹⁵  |
| Random      | 0.250 | —     | —         | —         |
| Majority    | 0.304 | —     | —         | —         |

PAF achieves 3× random baseline accuracy with a large effect size (d = 0.75). The confusion matrix shows balanced performance across all four families: kinases 83%, serine proteases 83%, metalloproteases 78%, nuclear receptors 63%. Nuclear receptors show the most cross-family confusion, consistent with the structural variability of their lipophilic ligand-binding pockets.

**Key control**: FP-B uses the same residue features and the same radial geometry but no wave transform. Its substantially lower effect size (d = 0.32 vs 0.75) isolates the Fourier transformation as the source of PAF's discriminative advantage — a 2.4-fold improvement in separation strength from the spectral step alone.

### 3.2 Kinase Subfamily Discrimination (Experiment 2)

**Objective**: Test finer-grained discrimination within a single protein family.

**Dataset**: 50 kinase structures across multiple subfamilies (TK, CMGC, AGC, STE, CK1), each co-crystallized with a ligand.

**Results (best configuration)**:

| Method      | 1-NN  | Cohen's d | p-value   |
|-------------|-------|-----------|-----------|
| PAF v1      | 0.42–0.50 | 0.49  | 2.6×10⁻⁷  |
| FP-B radial | 0.30  | 0.05      | 0.60      |
| FP-A mean   | 0.30  | 0.13      | 0.084     |

At the subfamily level, the FP-B control is essentially noise (d ≈ 0.05, p = 0.60). Radial geometry with the same features cannot discriminate kinase subfamilies — but PAF can (d = 0.49, p < 10⁻⁶). This demonstrates that the wave-domain transformation extracts structural organization invisible to radial binning.

Ablation analysis revealed that angular frequency modulation (γ > 0) substantially improves discrimination, charge and hydrophobicity channels dominate the signal, and local pocket geometry (8–10 Å) is more informative than larger radii.

### 3.3 DFG-in vs DFG-out Conformational Discrimination (Experiment 3)

**Objective**: Test whether PAF detects functional conformational states — specifically, the DFG-in (active) versus DFG-out (inactive) kinase switch.

**Dataset**: 28 kinase structures, perfectly balanced (14 DFG-in, 14 DFG-out). Includes 11 paired kinases (same protein in both states: ABL, BRAF, CDK2, EGFR, FGFR1, KIT, LCK, MET, SRC, VEGFR2, p38α) plus 3 unpaired per state.

**Results — Binary state classification**:

| Method      | 1-NN  | Cohen's d | p-value |
|-------------|-------|-----------|---------|
| PAF v1      | 0.571 | 0.254     | 0.014   |
| FP-B radial | 0.643 | 0.156     | 0.132   |
| FP-A mean   | 0.321 | −0.055    | 0.591   |

**Results — Paired analysis (11 kinases with both states)**:

| Comparison                  | PAF    | FP-B   | FP-A   |
|-----------------------------|--------|--------|--------|
| Same kinase, different state| 0.548  | 0.638  | 0.992  |
| Different kinase, same state| 0.435  | 0.620  | 0.990  |
| Different kinase, diff state| 0.402  | 0.605  | 0.990  |

**Interpretation**: Kinase identity dominates over conformational state in all methods — the DFG flip alters a subset of pocket residues while the overall fold remains conserved, so this is the expected result. However, PAF shows a clear similarity hierarchy (0.548 > 0.435 > 0.402) with larger state-dependent gaps than FP-B (0.033 vs 0.015), indicating the wave transform extracts more conformational information than radial binning. FP-A is completely blind to conformation (all similarities > 0.98).

**Per-kinase conformational sensitivity**:

| Kinase   | PAF cosine (in vs out) | Interpretation |
|----------|----------------------|----------------|
| p38α     | 0.249                | Large rearrangement detected |
| SRC      | 0.333                | Large change |
| BRAF     | 0.357                | Large change |
| CDK2     | 0.404                | Moderate |
| FGFR1    | 0.456                | Moderate |
| ABL      | 0.585                | Moderate |
| LCK      | 0.610                | Small change |
| KIT      | 0.631                | Small change |
| VEGFR2   | 0.668                | Small change |
| EGFR     | 0.751                | Minimal change |
| MET      | 0.981                | Near-identical |

This ordering is biologically interpretable. p38α with BIRB796 undergoes one of the most dramatic DFG-out rearrangements among kinases, with extensive remodeling of the active site. MET's inactive conformation involves a more subtle shift. PAF's per-kinase sensitivity tracks the magnitude of actual structural rearrangement, suggesting the embedding captures real geometric differences in pocket architecture.

---

## 4. Conceptual Basis

The design of PAF is informed by experimental spectroscopy. In NMR, IR, and ESR, structural information is encoded not through direct spatial measurement but through frequency-domain signatures reflecting local chemical environments. Chemical shifts, vibrational modes, and hyperfine splittings demonstrate that spectral representations can compactly summarize complex molecular organization.

PAF applies an analogous framework to protein pockets. Residues are mapped to localized oscillatory basis functions whose temporal position reflects radial geometry and whose frequency components encode physicochemical properties. Angular relationships are incorporated through frequency modulation. The superposition of oscillators generates a composite signal that is transformed via Fourier analysis into a spectral embedding.

PAF does not posit physical resonance in proteins. Instead, it leverages spectral transformation as an information-theoretic aggregation strategy. By mapping distributed structural features into frequency space, the approach captures interference patterns and higher-order organization that may be less apparent in direct geometric aggregation. The embedding is deterministic, compact, and parameter-controlled.

---

## 5. Discussion

### What PAF captures

The three experiments reveal a hierarchy of discriminative power. At the family level (kinase vs protease vs metalloprotease vs nuclear receptor), PAF achieves strong separation (d = 0.75) — these families differ substantially in fold, pocket geometry, and residue composition, and the spectral embedding captures these differences effectively. Within a single family, PAF still discriminates subfamilies (d = 0.49 for kinases), though at lower accuracy — subfamilies share the same fold and differ primarily in local pocket details. At the conformational level, PAF detects pocket rearrangements but does not override protein identity — a calibrated result consistent with the physical reality that the DFG flip alters a fraction of pocket residues while the overall architecture is conserved.

### The wave transform as active ingredient

The most important methodological finding is the controlled comparison against FP-B (radial histogram). This baseline uses identical residue features and identical radial geometry — the only difference is the absence of the oscillatory superposition and Fourier transform. FP-B achieves d = 0.05 on kinase subfamilies (noise) while PAF achieves d = 0.49 (strong signal). At the family level, FP-B reaches d = 0.32 while PAF reaches d = 0.75. This isolation demonstrates that the spectral transformation contributes real discriminative structure beyond what radial binning provides.

### Limitations

The current validation is limited in several respects. The dataset is modest (79 structures for cross-family, 50 for kinase subfamilies, 28 for DFG). Classification accuracy, while well above baseline, is not yet competitive with supervised methods trained on thousands of structures. The embedding has not been evaluated on functional prediction tasks such as binding affinity regression or ligand selectivity. PAF is deterministic and has not been integrated with learnable architectures. The pocket extraction depends on ligand position, limiting applicability to apo structures.

### Relationship to existing methods

PAF is not a replacement for geometric or neural pocket representations. It is a complementary signal that encodes pocket structure through a different mathematical lens. The spectral embedding could serve as an additional feature channel for graph neural networks, a pretraining signal for structure-based models, or a fast deterministic fingerprint for pocket similarity screening.

---

## 6. Summary of Key Results

| Experiment | Metric | PAF v1 | Best baseline | Random |
|-----------|--------|--------|---------------|--------|
| Cross-family (79 structs, 4 families) | 1-NN accuracy | 0.772 | 0.671 (FP-B) | 0.250 |
| Cross-family | Cohen's d | 0.748 | 0.420 (FP-A) | — |
| Kinase subfamilies (50 structs) | 1-NN accuracy | 0.42–0.50 | 0.30 (FP-B) | ~0.06 |
| Kinase subfamilies | Cohen's d | 0.49 | 0.05 (FP-B) | — |
| DFG-in/out (28 structs) | 1-NN accuracy | 0.571 | 0.643 (FP-B) | 0.500 |
| DFG-in/out | Cohen's d | 0.254 | 0.156 (FP-B) | — |
| DFG-in/out paired gap | State sensitivity | 0.033 | 0.015 (FP-B) | — |

---

## 7. Future Directions

Near-term priorities include expanding the structural benchmark to 200+ proteins, evaluating PAF on binding affinity prediction, and testing integration with graph neural network architectures as an additional input channel. The per-kinase conformational sensitivity results suggest PAF may be useful for detecting allosteric pocket changes, which warrants systematic evaluation across diverse allosteric systems. Longer-term, the spectral embedding framework could be extended to protein-ligand joint representations and generative design conditioning.

---

## Appendix: Experimental Parameters

All experiments used the following canonical configuration unless noted: pocket radius = 10 Å, γ_fm = 0.15, σ_t = 0.04, feature weights [a_hyd, a_charge, a_vol] = [1, 1, 0.5], 30-dimensional residue features (5 physicochemical + 1 flexibility + 1 contact + 3 secondary structure + 20 BLOSUM62), cosine similarity metric, leave-one-out k-NN evaluation.
