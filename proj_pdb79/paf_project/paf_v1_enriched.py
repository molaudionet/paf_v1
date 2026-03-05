#!/usr/bin/env python3
"""
PAF v1: Enriched Features for Kinase Pocket Classification
============================================================

Problem with PAF v0: 6 hand-picked bulk features (charge, hyd, hb, aro, vol, flex)
are too coarse to distinguish kinase families that share the same fold.

Solution: Add features that capture residue IDENTITY and LOCAL GEOMETRY.

Feature set (29 channels instead of 6):
  [0-19]  BLOSUM62 row: 20-dim embedding per amino acid
          Captures evolutionary substitution patterns. Asp and Glu are
          "similar" in ways charge/hyd alone don't express.
  [20]    Charge (same as v0)
  [21]    Hydrophobicity (same as v0)
  [22]    H-bond capacity (same as v0)
  [23]    Aromatic indicator (same as v0)
  [24]    Volume/size (same as v0)
  [25]    Flexibility/dynamics (same as v0)
  [26]    Contact count: how many other pocket residues within 5Å of this one
          Captures local packing density/geometry
  [27-28] Secondary structure: [helix_prob, sheet_prob]
          (coil = 1 - helix - sheet, implicit)

Why this should work better:
  - BLOSUM rows differentiate ALL 20 amino acids (v0 collapses many to same values)
  - Contact count captures whether a residue is buried or exposed in the pocket
  - Secondary structure captures the structural context (hinge helix, P-loop sheet, etc.)

Usage:
  python paf_v1_enriched.py --pdb_list data/kinase_pdbs/kinase_list.csv \
                            --out results/kinase_v1_enriched/

Dependencies:
  pip install numpy scipy matplotlib pandas biopython
  Optional: install dssp (brew install dssp) for proper secondary structure
"""

from __future__ import annotations

import os
import sys
import math
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1


def three_to_one(resname: str) -> str:
    return protein_letters_3to1[resname]


# ============================================================================
# BLOSUM62 matrix (20x20, standard amino acid order)
# ============================================================================

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Standard BLOSUM62 matrix (symmetric, half-bits)
_BLOSUM62_RAW = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
]

BLOSUM62 = np.array(_BLOSUM62_RAW, dtype=np.float32)

# Normalize BLOSUM rows to [0,1] range for consistency with other features
_B_MIN = BLOSUM62.min()
_B_MAX = BLOSUM62.max()
BLOSUM62_NORM = (BLOSUM62 - _B_MIN) / (_B_MAX - _B_MIN + 1e-8)


def blosum_embedding(aa: str) -> np.ndarray:
    """Get 20-dim BLOSUM62 row for amino acid, normalized to [0,1]."""
    idx = AA_TO_IDX.get(aa.upper(), None)
    if idx is None:
        return np.full(20, 0.5, dtype=np.float32)  # unknown AA
    return BLOSUM62_NORM[idx].copy()


# ============================================================================
# Original v0 features (kept for compatibility)
# ============================================================================

CHARGE = {"D": -1.0, "E": -1.0, "K": +1.0, "R": +1.0, "H": +0.5}
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
KD_MIN, KD_MAX = -4.5, 4.5
HB_RAW = {
    "D": 1, "E": 1, "N": 2, "Q": 2, "H": 2,
    "K": 1, "R": 1, "S": 2, "T": 2, "Y": 1, "W": 1, "C": 1,
}
AROMATIC = {"F", "Y", "W", "H"}
SIDECHAIN_HEAVY = {
    "A": 1, "C": 2, "D": 2, "E": 3, "F": 7, "G": 0, "H": 6,
    "I": 4, "K": 5, "L": 4, "M": 4, "N": 3, "P": 3, "Q": 4,
    "R": 6, "S": 2, "T": 3, "V": 3, "W": 10, "Y": 8,
}


def v0_features(aa: str) -> Dict[str, float]:
    aa = aa.upper()
    q = CHARGE.get(aa, 0.0)
    hyd = (KD.get(aa, 0.0) - KD_MIN) / (KD_MAX - KD_MIN)
    hyd = float(np.clip(hyd, 0.0, 1.0))
    hb = HB_RAW.get(aa, 0) / 2.0
    hb = float(np.clip(hb, 0.0, 1.0))
    aro = 1.0 if aa in AROMATIC else 0.0
    vol = SIDECHAIN_HEAVY.get(aa, 0) / 10.0
    vol = float(np.clip(vol, 0.0, 1.0))
    return {"charge": q, "hyd": hyd, "hb": hb, "aro": aro, "vol": vol}


# ============================================================================
# Channel definitions
# ============================================================================

# v0: 6 channels
V0_CHANNELS = ["charge", "hyd", "hb", "aro", "vol", "flex"]

# v1: 29 channels
V1_CHANNELS = [f"blosum_{i}" for i in range(20)] + [
    "charge", "hyd", "hb", "aro", "vol", "flex",
    "contacts", "ss_helix", "ss_sheet",
]

K_V0 = len(V0_CHANNELS)   # 6
K_V1 = len(V1_CHANNELS)   # 29


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ResidueV1:
    aa: str
    ca_xyz: np.ndarray
    radial_A: float
    features_v1: np.ndarray   # (29,) full feature vector
    features_v0: np.ndarray   # (6,) for v0 comparison


@dataclass
class PocketV1:
    pocket_id: str
    center_xyz: np.ndarray
    residues: List[ResidueV1]
    family: str = ""


# ============================================================================
# PAF parameters
# ============================================================================

@dataclass(frozen=True)
class PAFParamsV1:
    sr: int = 16000
    duration_s: float = 6.0
    pocket_radius_A: float = 10.0
    fmin_hz: float = 150.0
    fmax_hz: float = 2400.0
    sigma_t_s: float = 0.040
    n_bins: int = 256
    eps: float = 1e-8


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


# ============================================================================
# PAF embedding (generalized for K channels)
# ============================================================================

def paf_embedding(pocket: PocketV1, p: PAFParamsV1, use_v1: bool = True) -> np.ndarray:
    """
    Multi-channel PAF embedding.
    use_v1=True:  29 channels (enriched)
    use_v1=False: 6 channels (v0 for comparison)
    Returns: (K, n_bins) array
    """
    K = K_V1 if use_v1 else K_V0
    n = int(p.sr * p.duration_s)
    t = np.arange(n, dtype=np.float64) / float(p.sr)
    R = p.pocket_radius_A
    sig = np.zeros((K, n), dtype=np.float64)

    for rr in pocket.residues:
        r = float(np.clip(rr.radial_A, 0.0, R))
        tau = p.duration_s * (r / R)

        feats = rr.features_v1 if use_v1 else rr.features_v0

        # Frequency: use charge + hyd + vol from features (same mapping)
        if use_v1:
            q = feats[20]    # charge
            hyd = feats[21]  # hyd
            vol = feats[24]  # vol
        else:
            q = feats[0]
            hyd = feats[1]
            vol = feats[4]

        z = 1.0 * hyd + 1.0 * q + 0.5 * vol
        f = p.fmin_hz + (p.fmax_hz - p.fmin_hz) * sigmoid(z)
        phi = 0.0 if q >= 0 else np.pi

        # Gaussian-windowed sinusoid
        x = t - tau
        win = np.exp(-(x * x) / (2.0 * p.sigma_t_s ** 2))
        wave = win * np.sin(2.0 * np.pi * f * x + phi)

        # Each channel gets amplitude = its feature value
        for k in range(K):
            a = float(feats[k])
            if abs(a) > 1e-8:
                sig[k, :] += a * wave

    # FFT + log-spaced binning
    freqs = np.fft.rfftfreq(n, d=1.0 / p.sr)
    bin_edges = np.logspace(np.log10(p.fmin_hz), np.log10(p.fmax_hz), p.n_bins + 1)
    spec = np.zeros((K, p.n_bins), dtype=np.float64)

    for k in range(K):
        mag = np.abs(np.fft.rfft(sig[k, :]))
        for b in range(p.n_bins):
            mask = (freqs >= bin_edges[b]) & (freqs < bin_edges[b + 1])
            if mask.any():
                spec[k, b] = mag[mask].sum()
        total = spec[k, :].sum() + p.eps
        spec[k, :] /= total

    return spec.astype(np.float32)


# ============================================================================
# Raw fingerprint baselines (generalized for K channels)
# ============================================================================

def _to_matrix(pocket: PocketV1, use_v1: bool) -> Tuple[np.ndarray, np.ndarray]:
    sorted_res = sorted(pocket.residues, key=lambda r: r.radial_A)
    N = len(sorted_res)
    K = K_V1 if use_v1 else K_V0
    X = np.zeros((N, K), dtype=np.float32)
    radii = np.zeros(N, dtype=np.float32)
    for i, rr in enumerate(sorted_res):
        X[i, :] = rr.features_v1 if use_v1 else rr.features_v0
        radii[i] = rr.radial_A
    return X, radii


def fp_composition(pocket: PocketV1, use_v1: bool = True) -> np.ndarray:
    X, _ = _to_matrix(pocket, use_v1)
    K = K_V1 if use_v1 else K_V0
    return X.mean(axis=0) if len(X) > 0 else np.zeros(K, dtype=np.float32)


def fp_radial_hist(pocket: PocketV1, use_v1: bool = True,
                   n_shells: int = 16, R: float = 10.0) -> np.ndarray:
    X, radii = _to_matrix(pocket, use_v1)
    K = K_V1 if use_v1 else K_V0
    edges = np.linspace(0, R, n_shells + 1)
    fp = np.zeros((n_shells, K), dtype=np.float32)
    for s in range(n_shells):
        lo, hi = edges[s], edges[s + 1]
        mask = (radii >= lo) & (radii < hi)
        if s == n_shells - 1:
            mask = (radii >= lo) & (radii <= hi)
        if mask.sum() > 0:
            fp[s, :] = X[mask].mean(axis=0)
    return fp.reshape(-1)


def fp_sorted_concat(pocket: PocketV1, use_v1: bool = True,
                     n_max: int = 60) -> np.ndarray:
    X, _ = _to_matrix(pocket, use_v1)
    K = K_V1 if use_v1 else K_V0
    fp = np.zeros((n_max, K), dtype=np.float32)
    n_copy = min(len(X), n_max)
    fp[:n_copy, :] = X[:n_copy, :]
    return fp.reshape(-1)


# ============================================================================
# Pocket extraction with enriched features
# ============================================================================

def extract_pocket_v1(
    pdb_path: str,
    chain_id: str,
    ligand_resname: str = None,
    pocket_radius: float = 10.0,
    contact_cutoff: float = 5.0,
    family: str = "",
    kinase_name: str = "",
) -> PocketV1:
    """
    Extract pocket with v1 enriched features.
    New compared to v0:
      - BLOSUM62 embedding (20-dim)
      - Contact count (how many pocket neighbors within contact_cutoff)
      - Secondary structure from B-factor heuristic + backbone geometry
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    if chain_id not in model:
        available = [c.id for c in model.get_chains()]
        raise ValueError(f"Chain '{chain_id}' not found. Available: {available}")
    chain = model[chain_id]

    # --- Find ligand center ---
    ligand_atoms = []
    skip_het = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "SO4", "PO4",
                "GOL", "EDO", "ACE", "NH2", "DMS", "PEG", "MPD", "BME", "TRS"}

    if ligand_resname:
        for ch in model.get_chains():
            for res in ch.get_residues():
                if res.get_id()[0] == " ":
                    continue
                if res.get_resname().strip() == ligand_resname.strip():
                    for atom in res.get_atoms():
                        if atom.element != "H":
                            ligand_atoms.append(atom.get_coord())

    if not ligand_atoms:
        best_group = []
        for ch in model.get_chains():
            for res in ch.get_residues():
                if res.get_id()[0] == " ":
                    continue
                rname = res.get_resname().strip()
                if rname in skip_het:
                    continue
                atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                if len(atoms) > len(best_group):
                    best_group = atoms
        if best_group:
            ligand_atoms = best_group

    if not ligand_atoms:
        raise ValueError(f"No ligand found in {pdb_path}")

    center = np.mean(np.array(ligand_atoms, dtype=np.float32), axis=0)

    # --- Collect protein residue data ---
    res_data = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != " " or "CA" not in res:
            continue
        try:
            aa = three_to_one(res.get_resname())
        except KeyError:
            continue

        ca_coord = np.array(res["CA"].get_coord(), dtype=np.float32)
        dist = float(np.linalg.norm(ca_coord - center))
        if dist > pocket_radius:
            continue

        # B-factor
        bb_atoms_names = ["N", "CA", "C", "O"]
        bvals = [float(res[a].get_bfactor()) for a in bb_atoms_names if a in res]
        bfactor = float(np.mean(bvals)) if bvals else None

        # Backbone geometry for secondary structure estimation
        n_coord = np.array(res["N"].get_coord()) if "N" in res else None
        c_coord = np.array(res["C"].get_coord()) if "C" in res else None
        o_coord = np.array(res["O"].get_coord()) if "O" in res else None

        res_data.append({
            "aa": aa, "ca_xyz": ca_coord, "radial_A": dist,
            "bfactor": bfactor, "resseq": resseq,
            "n_coord": n_coord, "c_coord": c_coord, "o_coord": o_coord,
        })

    if len(res_data) < 5:
        raise ValueError(
            f"Only {len(res_data)} pocket residues in {pdb_path} "
            f"chain {chain_id} within {pocket_radius}Å"
        )

    # --- Compute B-factor z-scores ---
    bfactors = [rd["bfactor"] for rd in res_data if rd["bfactor"] is not None]
    bmean = float(np.mean(bfactors)) if bfactors else 0.0
    bstd = float(np.std(bfactors)) + 1e-8 if bfactors else 1.0

    # --- Estimate secondary structure from backbone phi/psi-like geometry ---
    # This is a rough approximation using CA-CA distances
    # Helix: ~5.4Å between CA_i and CA_i+2
    # Sheet: ~6.5-7.0Å between CA_i and CA_i+2
    ss_map = {}
    cas_sorted = sorted(res_data, key=lambda r: r["resseq"])
    for idx in range(len(cas_sorted)):
        resseq = cas_sorted[idx]["resseq"]
        if idx >= 2:
            d = float(np.linalg.norm(
                cas_sorted[idx]["ca_xyz"] - cas_sorted[idx-2]["ca_xyz"]))
            if d < 5.8:
                ss_map[resseq] = "H"  # helix-like
            elif d > 6.2:
                ss_map[resseq] = "E"  # sheet-like
            else:
                ss_map[resseq] = "C"  # coil
        else:
            ss_map[resseq] = "C"

    # --- Compute contact counts ---
    ca_coords = np.array([rd["ca_xyz"] for rd in res_data], dtype=np.float32)
    n_res = len(res_data)
    contact_counts = np.zeros(n_res, dtype=np.float32)
    for i in range(n_res):
        for j in range(i + 1, n_res):
            d = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            if d < contact_cutoff:
                contact_counts[i] += 1
                contact_counts[j] += 1
    # Normalize to [0, 1]
    max_contacts = contact_counts.max() + 1e-8
    contact_norm = contact_counts / max_contacts

    # --- Build residue objects ---
    residues = []
    for i, rd in enumerate(res_data):
        aa = rd["aa"]

        # BLOSUM embedding (20-dim)
        blosum = blosum_embedding(aa)

        # v0 features
        v0f = v0_features(aa)

        # Flexibility
        if rd["bfactor"] is not None:
            z = (rd["bfactor"] - bmean) / bstd
            z = float(np.clip(z, -2.0, 2.0))
            flex = float((z + 2.0) / 4.0)
        else:
            flex = 0.5

        # Secondary structure
        ss = ss_map.get(rd["resseq"], "C")
        ss_helix = 1.0 if ss == "H" else 0.0
        ss_sheet = 1.0 if ss == "E" else 0.0

        # Contact count (normalized)
        contacts = float(contact_norm[i])

        # Assemble v1 feature vector (29-dim)
        features_v1 = np.zeros(K_V1, dtype=np.float32)
        features_v1[0:20] = blosum                  # BLOSUM62 row
        features_v1[20] = v0f["charge"]             # charge
        features_v1[21] = v0f["hyd"]                # hydrophobicity
        features_v1[22] = v0f["hb"]                 # H-bond capacity
        features_v1[23] = v0f["aro"]                # aromatic
        features_v1[24] = v0f["vol"]                # volume
        features_v1[25] = flex                       # flexibility
        features_v1[26] = contacts                   # contact count
        features_v1[27] = ss_helix                   # helix indicator
        features_v1[28] = ss_sheet                   # sheet indicator

        # Assemble v0 feature vector (6-dim)
        features_v0 = np.array([
            v0f["charge"], v0f["hyd"], v0f["hb"],
            v0f["aro"], v0f["vol"], flex,
        ], dtype=np.float32)

        residues.append(ResidueV1(
            aa=aa, ca_xyz=rd["ca_xyz"], radial_A=rd["radial_A"],
            features_v1=features_v1, features_v0=features_v0,
        ))

    pocket_id = f"{kinase_name}|{os.path.basename(pdb_path)}"
    return PocketV1(
        pocket_id=pocket_id, center_xyz=center,
        residues=residues, family=family,
    )


# ============================================================================
# Evaluation metrics (same as evaluate_family_classification.py)
# ============================================================================

def cosine_matrix(embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    keys = list(embeddings.keys())
    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    S = X_norm @ X_norm.T
    return S


def silhouette_score(S: np.ndarray, labels: List[str]) -> float:
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    n = len(labels)
    unique = sorted(set(labels))
    if len(unique) < 2:
        return 0.0

    scores = []
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not same:
            continue
        a_i = float(np.mean([D[i, j] for j in same]))
        b_i = float("inf")
        for other in unique:
            if other == labels[i]:
                continue
            other_idx = [j for j in range(n) if labels[j] == other]
            if other_idx:
                b_i = min(b_i, float(np.mean([D[i, j] for j in other_idx])))
        if b_i < float("inf"):
            scores.append((b_i - a_i) / max(a_i, b_i, 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def loo_knn(S: np.ndarray, labels: List[str], k: int = 1) -> Tuple[float, Dict]:
    n = len(labels)
    S_copy = S.copy()
    np.fill_diagonal(S_copy, -np.inf)
    correct = 0
    fam_correct = Counter()
    fam_total = Counter()
    for i in range(n):
        fam_total[labels[i]] += 1
        nn_idx = np.argsort(S_copy[i, :])[-k:]
        nn_labels = [labels[j] for j in nn_idx]
        pred = Counter(nn_labels).most_common(1)[0][0]
        if pred == labels[i]:
            correct += 1
            fam_correct[labels[i]] += 1
    overall = correct / n
    per_fam = {f: fam_correct[f] / fam_total[f] for f in sorted(fam_total)}
    return overall, per_fam


def cohen_d_test(S: np.ndarray, labels: List[str]) -> Tuple[float, float, float, float]:
    n = len(labels)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(S[i, j])
            else:
                between.append(S[i, j])
    within, between = np.array(within), np.array(between)
    mw, mb = float(within.mean()), float(between.mean())
    pooled = float(np.sqrt(
        (within.var() * len(within) + between.var() * len(between))
        / (len(within) + len(between))
    )) + 1e-8
    d = (mw - mb) / pooled
    _, pval = ttest_ind(within, between, equal_var=False)
    return mw, mb, d, float(pval)


# ============================================================================
# Full comparison: v0 vs v1 features, PAF vs baselines
# ============================================================================

def run_comparison(pockets: Dict[str, PocketV1], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    params = PAFParamsV1()
    keys = list(pockets.keys())
    labels = [pockets[k].family for k in keys]
    n = len(keys)

    fam_counts = Counter(labels)
    print(f"\nFamily distribution ({n} pockets):")
    for f in sorted(fam_counts):
        print(f"  {f:8s}: {fam_counts[f]}")

    # --- Compute all embeddings ---
    print("\nComputing embeddings (this takes a few minutes for v1 with 29 channels)...")

    methods = {}

    # PAF v1 (29 channels) — the main test
    print("  PAF v1 (29ch)...", end="", flush=True)
    emb = {k: paf_embedding(pockets[k], params, use_v1=True) for k in keys}
    methods["PAF v1 (29ch)"] = emb
    print(f" dim={emb[keys[0]].reshape(-1).shape[0]}")

    # PAF v0 (6 channels) — previous version for comparison
    print("  PAF v0 (6ch)...", end="", flush=True)
    emb = {k: paf_embedding(pockets[k], params, use_v1=False) for k in keys}
    methods["PAF v0 (6ch)"] = emb
    print(f" dim={emb[keys[0]].reshape(-1).shape[0]}")

    # FP-A v1 composition (29-dim)
    emb = {k: fp_composition(pockets[k], use_v1=True) for k in keys}
    methods["FP-A v1 (comp)"] = emb
    print(f"  FP-A v1 (comp)  dim={emb[keys[0]].reshape(-1).shape[0]}")

    # FP-A v0 composition (6-dim)
    emb = {k: fp_composition(pockets[k], use_v1=False) for k in keys}
    methods["FP-A v0 (comp)"] = emb
    print(f"  FP-A v0 (comp)  dim={emb[keys[0]].reshape(-1).shape[0]}")

    # FP-B v1 radial histogram (29 * 16 = 464-dim)
    emb = {k: fp_radial_hist(pockets[k], use_v1=True) for k in keys}
    methods["FP-B v1 (radial)"] = emb
    print(f"  FP-B v1 (radial) dim={emb[keys[0]].reshape(-1).shape[0]}")

    # FP-B v0 radial histogram (6 * 16 = 96-dim)
    emb = {k: fp_radial_hist(pockets[k], use_v1=False) for k in keys}
    methods["FP-B v0 (radial)"] = emb
    print(f"  FP-B v0 (radial) dim={emb[keys[0]].reshape(-1).shape[0]}")

    # BLOSUM-only composition (20-dim, no wave, no spatial)
    emb = {}
    for k in keys:
        X = np.stack([r.features_v1[:20] for r in pockets[k].residues])
        emb[k] = X.mean(axis=0)
    methods["BLOSUM only (comp)"] = emb
    print(f"  BLOSUM only (comp) dim={emb[keys[0]].reshape(-1).shape[0]}")

    # --- Evaluate ---
    print(f"\n{'='*90}")
    print(f"{'Method':24s} {'Dim':>6s} {'Silh':>7s} {'1-NN':>7s} {'3-NN':>7s} {'Cohen d':>8s} {'p-val':>10s}")
    print(f"{'='*90}")

    results = []
    for name, embs in methods.items():
        S = cosine_matrix(embs)
        sil = silhouette_score(S, labels)
        acc1, per1 = loo_knn(S, labels, k=1)
        acc3, per3 = loo_knn(S, labels, k=3)
        mw, mb, d, pval = cohen_d_test(S, labels)
        dim = embs[keys[0]].reshape(-1).shape[0]

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {name:22s} {dim:>6d} {sil:>7.3f} {acc1:>6.1%} {acc3:>6.1%} {d:>8.3f} {pval:>9.2e} {sig}")

        results.append({
            "method": name, "dim": dim, "silhouette": sil,
            "loo_1nn": acc1, "loo_3nn": acc3,
            "cohen_d": d, "pval": pval,
            "mean_within": mw, "mean_between": mb,
        })

    print(f"{'='*90}")

    # Random baseline
    rng = np.random.default_rng(42)
    rand_accs = []
    S_paf = cosine_matrix(methods["PAF v1 (29ch)"])
    for _ in range(1000):
        shuf = labels.copy()
        rng.shuffle(shuf)
        a, _ = loo_knn(S_paf, shuf, k=1)
        rand_accs.append(a)
    chance = float(np.mean(rand_accs))
    majority = max(fam_counts.values()) / n
    print(f"\n  Random baseline: {chance:.1%}")
    print(f"  Majority class:  {majority:.1%}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "comparison_v0_vs_v1.csv"), index=False)

    # Per-family breakdown for best v1 method
    best_v1_name = None
    best_v1_acc = 0
    for r in results:
        if "v1" in r["method"] and r["loo_1nn"] > best_v1_acc:
            best_v1_acc = r["loo_1nn"]
            best_v1_name = r["method"]

    if best_v1_name:
        S_best = cosine_matrix(methods[best_v1_name])
        _, per_fam = loo_knn(S_best, labels, k=1)
        print(f"\n  Per-family 1-NN ({best_v1_name}):")
        for f in sorted(per_fam):
            print(f"    {f:8s} (n={fam_counts[f]:2d}): {per_fam[f]:.1%}")

    # --- FIGURES ---
    print("\nGenerating figures...")

    # Figure 1: v0 vs v1 comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    method_names = [r["method"] for r in results]
    x = np.arange(len(results))

    colors = []
    for m in method_names:
        if "PAF v1" in m:
            colors.append("#1565C0")
        elif "PAF v0" in m:
            colors.append("#90CAF9")
        elif "BLOSUM" in m:
            colors.append("#4CAF50")
        elif "v1" in m:
            colors.append("#FF9800")
        else:
            colors.append("#FFCC80")

    for ax, metric, title in zip(axes,
        ["loo_1nn", "cohen_d", "silhouette"],
        ["1-NN Accuracy (LOO)", "Cohen's d", "Silhouette Score"]):
        vals = [r[metric] for r in results]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        short = [m.replace(" (comp)", "\ncomp").replace(" (radial)", "\nradial").replace(" (29ch)", "\n29ch").replace(" (6ch)", "\n6ch") for m in method_names]
        ax.set_xticklabels(short, fontsize=7, rotation=0, ha="center")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if "accuracy" in title.lower():
            ax.axhline(chance, color="red", linewidth=1, linestyle="--", label=f"chance ({chance:.0%})")
            ax.axhline(majority, color="orange", linewidth=1, linestyle=":", label=f"majority ({majority:.0%})")
            ax.legend(fontsize=7)
            ax.set_ylim(0, max(vals) * 1.3 + 0.05)

    plt.suptitle("PAF v0 vs v1 (Enriched): Kinase Family Classification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_v0_vs_v1_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: fig_v0_vs_v1_comparison.png")

    # Figure 2: PCA for PAF v1
    for name in ["PAF v1 (29ch)", "PAF v0 (6ch)", "BLOSUM only (comp)"]:
        if name in methods:
            _plot_pca(methods[name], labels, name, out_dir, fam_counts)

    # Figure 3: Confusion matrix for best v1
    if best_v1_name:
        _plot_confusion(methods[best_v1_name], labels, best_v1_name, out_dir)

    # Figure 4: Feature importance — which BLOSUM dimensions matter most
    if "PAF v1 (29ch)" in methods:
        _plot_channel_variance(methods["PAF v1 (29ch)"], labels, out_dir)

    print(f"\nAll results saved to: {out_dir}/")
    return results


def _plot_pca(embs, labels, name, out_dir, fam_counts):
    keys = list(embs.keys())
    X = np.stack([embs[k].reshape(-1) for k in keys], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]
    total_var = (S ** 2).sum()
    v1 = S[0] ** 2 / total_var * 100
    v2 = S[1] ** 2 / total_var * 100

    palette = {
        "TK": "#E53935", "CMGC": "#7B1FA2", "AGC": "#1E88E5",
        "CAMK": "#FB8C00", "STE": "#78909C", "TKL": "#00ACC1",
        "CK1": "#43A047", "Other": "#795548",
    }
    unique_fams = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(10, 8))
    for fam in unique_fams:
        mask = [i for i, l in enumerate(labels) if l == fam]
        ax.scatter(Z[mask, 0], Z[mask, 1],
                  c=palette.get(fam, "#999"), s=70,
                  edgecolors="black", linewidth=0.5,
                  label=f"{fam} (n={fam_counts[fam]})", zorder=3)
    for i, k in enumerate(keys):
        short = k.split("|")[0]
        ax.annotate(short, (Z[i, 0], Z[i, 1]),
                   fontsize=6, alpha=0.8, xytext=(4, 4), textcoords="offset points")

    ax.legend(fontsize=9, title="Family", loc="best", framealpha=0.9)
    ax.set_xlabel(f"PC1 ({v1:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({v2:.1f}%)", fontsize=11)
    ax.set_title(f"{name} — PCA", fontsize=12)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    plt.savefig(os.path.join(out_dir, f"fig_pca_{safe}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: fig_pca_{safe}.png")


def _plot_confusion(embs, labels, name, out_dir):
    keys = list(embs.keys())
    S = cosine_matrix(embs)
    np.fill_diagonal(S, -np.inf)
    unique_fams = sorted(set(labels))
    fam_idx = {f: i for i, f in enumerate(unique_fams)}
    nf = len(unique_fams)
    cm = np.zeros((nf, nf), dtype=int)
    for i in range(len(keys)):
        nn = np.argmax(S[i, :])
        cm[fam_idx[labels[i]], fam_idx[labels[nn]]] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(nf))
    ax.set_yticks(range(nf))
    ax.set_xticklabels(unique_fams, rotation=45, ha="right")
    ax.set_yticklabels(unique_fams)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"1-NN Confusion — {name}", fontsize=12)
    for i in range(nf):
        for j in range(nf):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_confusion_v1.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: fig_confusion_v1.png")


def _plot_channel_variance(embs, labels, out_dir):
    """Show which channels contribute most to family separation."""
    keys = list(embs.keys())
    unique_fams = sorted(set(labels))

    # Get per-channel embeddings and compute F-ratio
    # PAF v1 embedding is (29, 256) = 29 channels x 256 freq bins
    all_emb = np.stack([embs[k] for k in keys])  # (N, 29, 256)
    N = all_emb.shape[0]

    # Per-channel: mean embedding per family, then between-vs-within variance
    channel_names = V1_CHANNELS
    f_ratios = []

    for ch in range(all_emb.shape[1]):
        ch_data = all_emb[:, ch, :]  # (N, 256)
        grand_mean = ch_data.mean(axis=0)
        ss_between = 0.0
        ss_within = 0.0
        for fam in unique_fams:
            mask = [i for i, l in enumerate(labels) if l == fam]
            if not mask:
                continue
            fam_data = ch_data[mask]
            fam_mean = fam_data.mean(axis=0)
            ss_between += len(mask) * np.sum((fam_mean - grand_mean) ** 2)
            ss_within += np.sum((fam_data - fam_mean) ** 2)

        df_b = len(unique_fams) - 1
        df_w = N - len(unique_fams)
        f_ratio = (ss_between / max(df_b, 1)) / (ss_within / max(df_w, 1) + 1e-8)
        f_ratios.append(f_ratio)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(channel_names))
    colors = ["#1565C0"] * 20 + ["#FF9800"] * 6 + ["#4CAF50"] * 3
    ax.bar(x, f_ratios, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("F-ratio (between/within family variance)", fontsize=10)
    ax.set_title("Channel Importance: Which Features Separate Kinase Families?", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Add category labels
    ax.text(10, max(f_ratios) * 0.95, "BLOSUM62", ha="center", fontsize=10,
            color="#1565C0", fontweight="bold")
    ax.text(23, max(f_ratios) * 0.95, "Physico-\nchem", ha="center", fontsize=9,
            color="#FF9800", fontweight="bold")
    ax.text(27, max(f_ratios) * 0.95, "Geom", ha="center", fontsize=9,
            color="#4CAF50", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_channel_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: fig_channel_importance.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_list", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=10.0)
    args = parser.parse_args()

    df = pd.read_csv(args.pdb_list)
    pockets = {}
    failed = 0

    print(f"Loading {len(df)} kinase pockets with enriched features...")
    for i, row in df.iterrows():
        pdb_path = str(row["pdb_path"])
        chain_id = str(row["chain_id"])
        label = str(row.get("pocket_label", f"kinase_{i}"))
        family = str(row.get("family", "unknown"))
        ligand = str(row.get("ligand_resname", "")) if pd.notna(row.get("ligand_resname")) else None

        if not os.path.exists(pdb_path):
            failed += 1
            continue

        try:
            pocket = extract_pocket_v1(
                pdb_path=pdb_path, chain_id=chain_id,
                ligand_resname=ligand, pocket_radius=args.radius,
                family=family, kinase_name=label,
            )
            pockets[pocket.pocket_id] = pocket
            n_res = len(pocket.residues)
            print(f"  OK  {label:20s} res={n_res:3d} fam={family}")
        except Exception as e:
            print(f"  FAIL {label:20s} {e}")
            failed += 1

    print(f"\nLoaded {len(pockets)} pockets ({failed} failed)")

    if len(pockets) < 10:
        print("ERROR: Too few pockets")
        sys.exit(1)

    run_comparison(pockets, args.out)
