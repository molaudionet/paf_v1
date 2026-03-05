#!/usr/bin/env python3
"""
EXECUTION GUIDE: PAF v0 Validation on Kinase Pockets
=====================================================

This document ties together all the code you have and tells you
exactly what to run, in what order, and what to do with each result.

Total estimated time: 3-5 days
Compute required: Mac Mini (no GPU needed)
Cost: $0 (all local)

FILES YOU HAVE:
  raw_fingerprint_baseline.py   - 4 baseline fingerprints (the critical ablation)
  head_to_head.py               - Full comparison pipeline + figures
  prepare_kinase_dataset.py     - 50 curated kinases + download helper
  positive_controls.py          - Audio discrimination controls (for Paper 1)
  power_analysis.py             - Sample size calculations (for Paper 1)
  protein_sonification_scoping.py - Task scoping + pocket extraction code

DECISION TREE:
  The experiment produces ONE number that determines your next move:
  PAF Spearman rho vs structural similarity.

  If PAF rho > 0.3 AND PAF > best baseline:
    → Wave transform captures real structure. Write paper.
    → Send collaboration emails with figures attached.

  If PAF rho > 0.3 BUT PAF ≈ FP-B:
    → Radial binning is the signal. FFT adds overhead.
    → Revise wave mapping before publishing PAF formalism.
    → Still publishable as "radial feature profiles discriminate kinase pockets."

  If ALL methods rho < 0.15:
    → Residue-level features are too coarse.
    → Try atom-level sonification (Level 2) or add sequence features.
    → Do NOT publish PAF formalism until this is resolved.
"""


# ============================================================================
# STEP-BY-STEP EXECUTION
# ============================================================================

STEPS = """
STEP 0: INSTALL DEPENDENCIES (5 min)
======================================
pip install numpy scipy matplotlib pandas biopython

STEP 1: VERIFY PIPELINE ON SYNTHETIC DATA (10 min)
====================================================
python head_to_head.py --mode synthetic --out results/synthetic_test/

Check:
  - Does it complete without errors?
  - Do you see the 4 figures in results/synthetic_test/?
  - Does the PCA show some family separation? (it should on synthetic data)
  
If yes → proceed.
If errors → debug before touching real data.

STEP 2: PREPARE KINASE DATASET (30 min)
==========================================
# Generate the manifest CSV
python prepare_kinase_dataset.py --csv_only --out_dir data/kinase_pdbs/

# Download PDB files (requires internet)
python prepare_kinase_dataset.py --download --out_dir data/kinase_pdbs/

# Verify: you should have ~50 .pdb files in data/kinase_pdbs/
ls data/kinase_pdbs/*.pdb | wc -l

NOTE: Some PDB IDs in my curated list may have issues (chain mismatch,
ligand name differences, etc.). Expect 5-10% failures. That's normal.
Remove failed entries from the CSV before proceeding.

STEP 3: INTEGRATE POCKET EXTRACTION WITH HEAD-TO-HEAD (1-2 hours)
===================================================================
The head_to_head.py currently has a "synthetic" mode and a "real" mode
stub. You need to connect the pocket extraction code from
protein_sonification_scoping.py to populate Pocket objects from real PDBs.

The key function to adapt:

  from protein_sonification_scoping import extract_binding_pocket_residues
  
  # For each PDB in kinase_list.csv:
  residues = extract_binding_pocket_residues(
      pdb_path="data/kinase_pdbs/1IEP.pdb",
      ligand_resname="STI",
      distance_cutoff=10.0
  )

Then convert the returned dicts into head_to_head.ResidueRecord objects
and build Pocket instances. The family label comes from the CSV.

Alternatively, use the extract_pocket() function from the PAF code
blueprint (in the OpenAI conversation) which already does this, including
the motif-based fallback for structures without ligands.

STEP 4: RUN HEAD-TO-HEAD ON REAL KINASES (1-2 hours compute)
==============================================================
python head_to_head.py --mode real --pdb_list data/kinase_pdbs/kinase_list.csv \\
                       --out results/kinase_v0/

This produces:
  results/kinase_v0/comparison_results.csv     ← THE KEY TABLE
  results/kinase_v0/fig_comparison_bar.png     ← main result figure  
  results/kinase_v0/fig_scatter_grid.png       ← scatter diagnostics
  results/kinase_v0/fig_pca_paf.png            ← does PAF separate families?
  results/kinase_v0/fig_pca_fpb.png            ← comparison PCA

STEP 5: INTERPRET RESULTS (the decision point)
================================================
Open comparison_results.csv. Find:

  1. PAF spearman_rho  
  2. FP-B spearman_rho (the critical comparator)
  3. FP-A spearman_rho (composition floor)

Then look at fig_pca_paf.png:
  - Do kinase families cluster?
  - Which families overlap vs separate?

INTERPRETATION → ACTION MAP:

  CASE A: PAF ρ > 0.4, PAF > FP-B by >0.1
    ACTION: Strong result. Wave transform captures real structure.
    NEXT: Write Paper 2A (pocket similarity). Send collaboration emails.
    FIGURES: Use fig_pca_paf.png + fig_comparison_bar.png directly.
    TIMELINE: 1-2 weeks to preprint.

  CASE B: PAF ρ > 0.3, PAF ≈ FP-B (within 0.05)
    ACTION: Radial feature profiles work but FFT doesn't help beyond binning.
    NEXT: Try three things before publishing PAF:
      (a) Change sigma_t (try 20ms, 80ms) — may sharpen/blur spectral features
      (b) Add angular information (2nd anchor point)
      (c) Use atom-level instead of residue-level
    If none help, publish as "radial feature profiles" without the wave framing.
    TIMELINE: 2-3 weeks iteration.

  CASE C: All methods ρ < 0.15, PCA shows no family separation
    ACTION: These 6 residue-level features are insufficient.
    NEXT: Enrichment experiments:
      (a) Add sequence-derived features (BLOSUM, conservation scores)
      (b) Try atom-level with your existing small-molecule sonification
      (c) Try ECFP-style pocket fingerprints as a "strong baseline" to
          check if the structural similarity measure itself is working
    TIMELINE: 3-4 weeks.

  CASE D: FP-A ρ > 0.3 but FP-B and PAF ρ < FP-A
    ACTION: Composition matters but spatial info hurts (noise).
    NEXT: Fewer, more informative features. Or just use composition
          as a very cheap pocket descriptor (still publishable).

STEP 6: PARAMETER SENSITIVITY (if result is promising, ~1 day)
================================================================
If CASE A or B:
  - Vary pocket radius: R=6, 8, 10, 12 Å
  - Vary n_bins: 64, 128, 256, 512
  - Vary sigma_t: 0.02, 0.04, 0.08 s
  - Report sensitivity table (ρ vs parameter)
  
This goes in paper supplementary and shows robustness.

STEP 7: WRITE PAPER / SEND EMAILS
=====================================
Paper 2A structure:
  "Wave-domain pocket representation captures kinase structural similarity"
  
  Intro: Current protein representations are static/geometric.
         We propose encoding pockets as multi-channel spectral signals.
  
  Methods: PAF mapping, pocket extraction, kinase dataset.
  
  Results:
    Table: PAF vs baselines Spearman ρ (comparison_results.csv)
    Fig 1: Pipeline diagram (pocket → PAF)
    Fig 2: Family-colored PCA (fig_pca_paf.png)
    Fig 3: PAF vs baselines bar chart (fig_comparison_bar.png)
    Fig 4: Scatter vs structural similarity (fig_scatter_grid.png)
  
  Discussion: What PAF captures, limitations, next steps.
  
Collaboration email: send AFTER you have the PCA figure showing
family clustering. Attach the figure directly.
"""


# ============================================================================
# WHAT NOT TO DO (common mistakes to avoid)
# ============================================================================

WARNINGS = """
COMMON MISTAKES TO AVOID:

1. DON'T skip the raw fingerprint baseline.
   Without it, you don't know if PAF adds anything beyond raw features.
   Reviewers WILL ask "why not just use a feature vector?"

2. DON'T use TM-score on just the pocket.
   TM-score is designed for full-domain alignment.
   For pocket-only comparison, use pocket Cα RMSD (already implemented
   in head_to_head.py) or a pocket shape descriptor.
   If you want TM-score later, align the full kinase domains and use
   US-align or TM-align (external tools, easy to script).

3. DON'T claim "resonance" or "physical vibration" without evidence.
   PAF is a synthetic spectral representation computed from structure.
   Call it "spectral similarity in a constructed wave domain."
   If reviewers think you're claiming real molecular resonance
   without spectroscopic measurement, they'll reject it.

4. DON'T send collaboration emails before you have results.
   Figures > formalism > promises.

5. DON'T over-engineer the codebase.
   You have ~1000 lines of working code. That's enough for v0.
   The goal is results, not architecture.

6. DON'T mix Paper 1 (LLM audio) with Paper 2A (pocket similarity).
   They test different hypotheses with different methods.
   Keep them separate even if they share the sonification concept.
"""


# ============================================================================
# TIMELINE
# ============================================================================

TIMELINE = """
RECOMMENDED TIMELINE (assuming ~4 hours/day available):

Day 1: Verify synthetic pipeline + download kinase PDBs
Day 2: Integrate pocket extraction, fix PDB parsing issues
Day 3: Run full head-to-head on real kinases
Day 4: Interpret results, run parameter sensitivity if promising
Day 5: Draft paper figures + outline (if positive)
       OR iterate on mapping/features (if weak)

In parallel (independent, can do any time):
  - Paper 1 positive controls (positive_controls.py)
  - Paper 1 pilot with 50 BBBP molecules
  - Power analysis review
"""


if __name__ == "__main__":
    print(STEPS)
    print(WARNINGS)
    print(TIMELINE)
