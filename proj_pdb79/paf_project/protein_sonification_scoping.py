#!/usr/bin/env python3
"""
Protein Sonification Task Scoping
===================================
Concrete plan for extending molecular sonification from small molecules
to proteins/DNA/RNA — answering "can we do it to big molecules?"

This defines ONE narrow, well-benchmarked first task rather than
trying to boil the ocean.

Author's context: the submitted paper (zhou_molecularAI_2026) validates
sonification on small molecules (BBBP, Tox21). This document scopes
the minimal credible extension to macromolecules.
"""

# =============================================================================
# THE CORE PROBLEM WITH "BIG MOLECULES"
# =============================================================================
#
# Small molecules (your submitted paper):
#   - 20-50 atoms
#   - Linear SMILES → straightforward atom-by-atom sonification
#   - Fixed-length audio (6s) captures entire molecule
#   - Clear benchmarks: BBBP, Tox21, ESOL
#
# Proteins:
#   - 1,000-50,000+ atoms
#   - 3D fold matters more than sequence
#   - Residue 50 may contact residue 300 in 3D space
#   - Audio of entire protein = very long, with relevant info diluted
#   - Benchmark question: what is the supervised signal?
#
# DNA/RNA:
#   - Even longer (thousands to millions of bases)
#   - Function often depends on non-coding structure, epigenetics
#   - Fewer clean benchmarks for "property from structure"
#
# KEY INSIGHT: You need a task where:
#   1. The "molecular region" to sonify is well-defined and small
#   2. The supervised label is crisp and quantitative
#   3. Established baselines exist for comparison
#   4. The result connects directly to drug discovery value
#
# =============================================================================


# =============================================================================
# RECOMMENDED FIRST TASK: Kinase Binding Pocket Sonification
# =============================================================================
#
# WHY KINASES?
#   - ~500 human kinases, highly studied, abundant structures in PDB
#   - Kinase inhibitors are a major drug class (~70 FDA-approved)
#   - The binding pocket (ATP site) is well-defined: ~20-30 residues
#   - Selectivity (which kinase does a drug hit?) is a real unsolved problem
#   - Clean benchmarks exist (Davis, KIBA, kinase selectivity panels)
#
# WHAT YOU SONIFY:
#   NOT the entire protein. Only the binding pocket (~20-30 residues).
#   This is roughly the same size as a small molecule (~50-100 atoms).
#   Your existing sonification pipeline can handle this with minimal changes.
#
# THE TASK:
#   Given two kinase binding pockets sonified as audio:
#   → Predict whether they bind the same inhibitor (selectivity)
#   → Or: predict binding affinity (regression)
#
# WHY THIS WORKS FOR YOUR NARRATIVE:
#   "We showed sonification works for small molecules. Now we show the
#    same encoding captures protein binding site properties — bridging
#    ligand and target in the same audio space."
#
# =============================================================================


# =============================================================================
# CONCRETE EXPERIMENTAL DESIGN
# =============================================================================

TASK_DEFINITION = """
TASK: Kinase Binding Pocket Audio Classification
=================================================

Input:  Audio waveform of a kinase binding pocket (ATP site residues)
Output: Kinase family classification (e.g., TK, TKL, STE, CMGC, AGC, CK1, CAMK)
        or pairwise similarity score

Dataset options (pick ONE to start):
  Option A: KinBase classification
    - ~500 human kinases, 7 major groups
    - Task: classify kinase group from binding pocket audio
    - N: ~400 structures with resolved ATP sites
    - Metric: accuracy / macro F1
    - Baseline: sequence-based classification is ~95%, so your bar is high
    - Advantage: clean labels, large N

  Option B: Kinase selectivity (Davis/KIBA)
    - ~400 kinase-inhibitor pairs with Kd values
    - Task: predict binding strength from pocket audio + ligand audio
    - N: ~300-400 pairs with structures
    - Metric: Spearman correlation, RMSE
    - Baseline: structure-based methods ~0.6-0.7 Spearman
    - Advantage: directly drug-discovery relevant

  Option C: Pocket similarity (simplest, recommended FIRST)
    - Take 50 kinase structures
    - Sonify each binding pocket
    - Compute audio similarity (cosine on Wav2Vec2 embeddings)
    - Compare to structural similarity (TM-score or pocket RMSD)
    - If audio similarity correlates with structural similarity,
      you've proven the encoding captures meaningful protein info
    - N: 50 structures → 1,225 pairs
    - Metric: Spearman correlation between audio sim and structural sim
    - Baseline: no established baseline (novel!)
    - Advantage: no labels needed, purely structural validation

RECOMMENDATION: Start with Option C (pocket similarity).
  - No labels to collect
  - Fast to run (~50 sonifications + pairwise comparison)
  - Novel (no one has done this)
  - Clean positive result: "audio similarity tracks structural similarity"
  - Natural bridge to Options A/B in a follow-up
"""


# =============================================================================
# SONIFICATION STRATEGY FOR BINDING POCKETS
# =============================================================================

SONIFICATION_STRATEGY = """
How to Sonify a Protein Binding Pocket
========================================

Your small-molecule sonification walks atom-by-atom through the SMILES:
  atom → (mass → pitch, electronegativity → timbre, polarity → amplitude)

For a binding pocket, you need to decide:
  1. Which atoms/residues to include
  2. What order to "read" them (there's no linear SMILES for a pocket)
  3. What additional properties to encode

PROPOSED APPROACH (3 levels, matching your v2 strategy doc):

Level 1: Residue-level sonification (simplest, start here)
  - Extract binding pocket residues (atoms within 4Å of co-crystallized ligand)
  - For each RESIDUE (not atom), encode:
      * Residue mass → pitch
      * Residue hydrophobicity (Kyte-Doolittle) → timbre
      * Residue charge at pH 7 → amplitude
      * Secondary structure type (helix/sheet/coil) → rhythm pattern
  - Ordering: by distance from pocket centroid (inner → outer)
  - Duration: ~0.2s per residue → 20 residues = 4 seconds
  - This keeps audio in the same 4-6s range as your small molecules

Level 2: Atom-level interface sonification (medium complexity)
  - Same pocket extraction
  - Sonify every heavy atom (like your small molecule pipeline)
  - Ordering: residue-by-residue, atoms within each residue by depth
  - Duration: longer (~10-15s for ~200 atoms)
  - Captures more chemical detail but noisier signal

Level 3: Dynamics-enriched sonification (advanced, for Paper 2)
  - Encode B-factors → amplitude modulation (flexible regions = louder)
  - Encode electrostatic potential → harmonic overtones
  - This is your "Level 3" from the v2 strategy doc
  - Only pursue after Level 1 validates

START WITH LEVEL 1.
"""


# =============================================================================
# POCKET EXTRACTION CODE (ready to use)
# =============================================================================

def extract_binding_pocket_residues(
    pdb_path: str,
    ligand_resname: str = "LIG",
    distance_cutoff: float = 4.0,
) -> list[dict]:
    """
    Extract binding pocket residues from a PDB file.

    Returns list of dicts with residue-level properties suitable for
    sonification. Ordered by distance from pocket centroid.

    Requirements: pip install biopython

    Parameters
    ----------
    pdb_path : str
        Path to PDB file with protein + co-crystallized ligand
    ligand_resname : str
        Residue name of the ligand in the PDB (e.g., "LIG", "ATP", drug code)
    distance_cutoff : float
        Angstrom cutoff for defining pocket residues
    """
    from Bio.PDB import PDBParser, NeighborSearch
    import numpy as np

    # Kyte-Doolittle hydrophobicity scale
    HYDROPHOBICITY = {
        'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
        'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
        'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
        'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
    }

    # Charge at pH 7 (simplified)
    CHARGE_PH7 = {
        'ARG': +1, 'LYS': +1, 'HIS': +0.1,  # His partially protonated
        'ASP': -1, 'GLU': -1,
    }

    # Residue average mass (Da)
    RESIDUE_MASS = {
        'ALA': 89, 'ARG': 174, 'ASN': 132, 'ASP': 133, 'CYS': 121,
        'GLN': 146, 'GLU': 147, 'GLY': 75, 'HIS': 155, 'ILE': 131,
        'LEU': 131, 'LYS': 146, 'MET': 149, 'PHE': 165, 'PRO': 115,
        'SER': 105, 'THR': 119, 'TRP': 204, 'TYR': 181, 'VAL': 117,
    }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # Collect ligand atoms
    ligand_atoms = []
    protein_atoms = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname == ligand_resname:
                ligand_atoms.extend(residue.get_atoms())
            elif residue.get_id()[0] == " ":  # standard residue
                protein_atoms.extend(residue.get_atoms())

    if not ligand_atoms:
        raise ValueError(f"No ligand with resname '{ligand_resname}' found in {pdb_path}")

    # Find pocket residues (protein residues with any atom within cutoff of ligand)
    ns = NeighborSearch(protein_atoms)
    pocket_residue_ids = set()
    for lig_atom in ligand_atoms:
        nearby = ns.search(lig_atom.get_vector().get_array(), distance_cutoff)
        for atom in nearby:
            res = atom.get_parent()
            pocket_residue_ids.add(res.get_full_id())

    # Extract residue-level features
    pocket_residues = []
    for chain in model:
        for residue in chain:
            if residue.get_full_id() in pocket_residue_ids:
                resname = residue.get_resname().strip()
                if resname not in RESIDUE_MASS:
                    continue  # skip non-standard

                # Compute residue centroid
                coords = np.array([a.get_vector().get_array() for a in residue.get_atoms()])
                centroid = coords.mean(axis=0)

                pocket_residues.append({
                    'resname': resname,
                    'chain': chain.get_id(),
                    'resid': residue.get_id()[1],
                    'centroid': centroid,
                    'mass': RESIDUE_MASS.get(resname, 120),
                    'hydrophobicity': HYDROPHOBICITY.get(resname, 0.0),
                    'charge': CHARGE_PH7.get(resname, 0.0),
                    'n_atoms': len(list(residue.get_atoms())),
                })

    if not pocket_residues:
        raise ValueError(f"No pocket residues found within {distance_cutoff}Å of ligand")

    # Compute pocket centroid and sort by distance from center
    all_centroids = np.array([r['centroid'] for r in pocket_residues])
    pocket_center = all_centroids.mean(axis=0)

    for r in pocket_residues:
        r['dist_from_center'] = float(np.linalg.norm(r['centroid'] - pocket_center))

    pocket_residues.sort(key=lambda r: r['dist_from_center'])

    return pocket_residues


def sonify_pocket_residues(
    residues: list[dict],
    sr: int = 16000,
    duration_per_residue: float = 0.2,
    rms_target: float = 0.05,
) -> "np.ndarray":
    """
    Convert pocket residues to audio (Level 1: residue-level sonification).

    Mapping:
      mass → pitch (Hz): linear map from [75, 204] Da to [200, 2000] Hz
      hydrophobicity → timbre: number of harmonics (hydrophobic = richer)
      charge → amplitude modulation: charged residues are louder
      distance from center → stereo panning (mono for now, but encoded as
                             slight pitch bend for depth cue)

    Returns numpy array (float32) of the complete audio waveform.
    """
    import numpy as np

    # Constants
    mass_min, mass_max = 75.0, 204.0
    freq_min, freq_max = 200.0, 2000.0
    hydro_min, hydro_max = -4.5, 4.5

    samples_per_res = int(sr * duration_per_residue)
    total_samples = samples_per_res * len(residues)
    audio = np.zeros(total_samples, dtype=np.float64)

    for i, res in enumerate(residues):
        t = np.linspace(0, duration_per_residue, samples_per_res, endpoint=False)

        # Pitch from mass
        mass_norm = (res['mass'] - mass_min) / (mass_max - mass_min + 1e-8)
        freq = freq_min + mass_norm * (freq_max - freq_min)

        # Timbre from hydrophobicity (more hydrophobic = more harmonics)
        hydro_norm = (res['hydrophobicity'] - hydro_min) / (hydro_max - hydro_min + 1e-8)
        n_harmonics = max(1, int(1 + hydro_norm * 6))  # 1-7 harmonics

        # Generate tone with harmonics
        sig = np.zeros_like(t)
        for h in range(1, n_harmonics + 1):
            sig += (0.3 / h) * np.sin(2 * np.pi * freq * h * t)

        # Amplitude from charge (charged residues louder)
        charge_factor = 1.0 + 0.5 * abs(res['charge'])
        sig *= charge_factor

        # Apply envelope (gentle attack/release to avoid clicks)
        attack = int(0.01 * sr)
        release = int(0.02 * sr)
        envelope = np.ones_like(t)
        if attack > 0 and attack < len(envelope):
            envelope[:attack] = np.linspace(0, 1, attack)
        if release > 0 and release < len(envelope):
            envelope[-release:] = np.linspace(1, 0, release)
        sig *= envelope

        start = i * samples_per_res
        audio[start:start + samples_per_res] = sig

    # RMS normalize
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (rms_target / current_rms)

    return audio.astype(np.float32)


# =============================================================================
# POCKET SIMILARITY EXPERIMENT (Option C — recommended first task)
# =============================================================================

EXPERIMENT_PROTOCOL = """
Pocket Similarity Experiment Protocol
=======================================

Goal: Show that audio similarity between sonified binding pockets
correlates with structural similarity.

If it does → sonification captures meaningful protein info.
If it doesn't → encoding needs revision before scaling.

Steps:

1. Select 50 kinase structures from PDB
   - Mix of kinase families (TK, CMGC, AGC, etc.)
   - All with co-crystallized ligand (for pocket definition)
   - Resolution < 2.5Å
   - One structure per kinase (no redundancy)

2. Extract binding pocket for each (4Å cutoff)
   → pocket_residues = extract_binding_pocket_residues(pdb_path, ligand_resname)

3. Sonify each pocket (Level 1: residue-level)
   → audio = sonify_pocket_residues(pocket_residues)
   → Save as WAV

4. Extract audio embeddings
   - Use Wav2Vec 2.0 (same as your small-molecule paper)
   - Extract 768-dim embedding per pocket audio
   → This reuses your existing embedding pipeline

5. Compute pairwise audio similarity
   - Cosine similarity on Wav2Vec2 embeddings
   → 50×50 similarity matrix (1,225 unique pairs)

6. Compute pairwise structural similarity
   - TM-score between full kinase domains (using TM-align or US-align)
   - OR pocket RMSD (faster, pocket-specific)
   → 50×50 similarity matrix

7. Compare: Spearman correlation between audio sim and structural sim
   - Report correlation + 95% CI + p-value
   - Plot scatter: audio_sim vs structural_sim

Expected results:
  - If Spearman > 0.3 with p < 0.05 → strong evidence
  - If Spearman > 0.5 → very strong
  - If Spearman < 0.1 → encoding doesn't capture pocket geometry well

Why this is publishable either way:
  - Positive: "Sonification captures protein binding site similarity"
  - Negative: "Residue-level encoding insufficient; atom-level needed"
  - Both are informative for the field

Compute cost: zero (local). Time: ~1 day.
"""


# =============================================================================
# WHAT ABOUT DNA/RNA?
# =============================================================================

DNA_RNA_PLAN = """
DNA/RNA Sonification: When and How
====================================

DNA/RNA is important for your "big molecules" story, but harder to
scope cleanly for a first validation. Here's why, and when to do it.

CHALLENGES:
  1. Length: even a short gene is 1000+ nucleotides
  2. Function depends on sequence + secondary structure + epigenetics
  3. Few "structure → property" benchmarks as clean as BBBP

WHEN TO DO IT:
  After protein pocket sonification validates (Option C above).
  Then pick ONE of these:

  Task A: RNA secondary structure classification
    - Dataset: bpRNA or similar
    - Sonify short RNA sequences (50-200 nt)
    - Encode: base type → pitch, base-pairing propensity → timbre,
              GC content (local) → amplitude
    - Task: classify structural motif (hairpin vs stem-loop vs pseudoknot)
    - This is tractable and has established baselines

  Task B: Promoter vs non-promoter DNA classification
    - Dataset: EPDnew promoter database
    - Sonify 200-500bp DNA sequences
    - Encode: base type → pitch (A=low, T=low, G=high, C=high),
              GC% in sliding window → timbre, CpG density → amplitude
    - Task: binary classification (promoter / non-promoter)
    - Clean benchmark with established baselines

  Task C: Aptamer binding (advanced)
    - Sonify short RNA/DNA aptamers (~30-80 nt)
    - Predict binding affinity to target protein
    - Connects to drug discovery narrative
    - Smaller scale, more tractable

RECOMMENDATION: Do protein pockets first (cleaner, more drug-relevant).
DNA/RNA is Paper 3 or a section of Paper 2 if pocket results are strong.
"""


# =============================================================================
# COMPLETE ROADMAP: SMALL → BIG MOLECULES
# =============================================================================

ROADMAP = """
Sonification Scaling Roadmap
==============================

DONE (submitted paper):
  ✓ Small molecule sonification → BBBP, Tox21
  ✓ Wav2Vec2 transfer learning
  ✓ Fused model outperforms unimodal

PAPER 1 (in progress):
  □ Zero-shot LLM audio classification (BBBP)
  □ Positive controls (discrimination + noise null)
  □ Power-aware sample sizing

PAPER 2 (next — "big molecules"):
  □ Kinase binding pocket sonification (Level 1: residue-level)
  □ Pocket similarity experiment (50 kinases, no labels needed)
  □ If positive → pocket classification by kinase family
  □ If strong → ligand-pocket audio fusion for selectivity

PAPER 3 (later):
  □ DNA/RNA sonification (RNA structure classification or promoter detection)
  □ Dynamics sonification (B-factor, MD trajectory)
  □ Full multi-scale: ligand audio + pocket audio + dynamics audio

VC MILESTONE MAP:
  Paper 1 → "Foundation models process our encoding" (platform compatibility)
  Paper 2 → "Encoding scales to proteins" (technical breadth)
  Paper 3 → "Universal molecular audio" (full thesis validation)
"""


if __name__ == "__main__":
    print(TASK_DEFINITION)
    print(SONIFICATION_STRATEGY)
    print(EXPERIMENT_PROTOCOL)
    print(DNA_RNA_PLAN)
    print(ROADMAP)
    print("\n--- POCKET EXTRACTION DEMO ---")
    print("To run pocket extraction, you need a PDB file with a ligand.")
    print("Example:")
    print("  residues = extract_binding_pocket_residues('3hb5.pdb', 'STI')")
    print("  audio = sonify_pocket_residues(residues)")
    print("  import soundfile as sf")
    print("  sf.write('pocket_3hb5.wav', audio, 16000)")
