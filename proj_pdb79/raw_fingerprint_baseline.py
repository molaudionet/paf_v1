#!/usr/bin/env python3
"""
Raw Fingerprint Baseline for PAF Ablation
==========================================

This is the single most important scientific control for the PAF concept.

Question: Does the wave-domain transformation (feature -> signal -> FFT -> bins)
          actually add information, or is it equivalent to just using the raw
          residue features directly?

Method:
  Same 6 residue features as PAF (charge, hyd, hb, aro, vol, flex).
  Same pocket extraction (same residues, same radial distances).
  But NO wave mapping, NO FFT, NO spectral binning.

  Instead, we build fixed-length fingerprints using simple aggregation:

  FP-A: "Composition fingerprint" (6-dim)
    Mean of each feature across pocket residues.
    Simplest possible baseline. If PAF can't beat this, the wave
    mapping adds nothing.

  FP-B: "Radial histogram fingerprint" (6 x R_bins = 6 x 16 = 96-dim)
    Bin residues by radial distance into R_bins shells.
    Within each shell, average each feature.
    This captures the same radial structure that PAF encodes via
    time-placement, but without the wave/spectral transformation.
    THIS IS THE KEY COMPARISON: same info, no FFT.

  FP-C: "Sorted feature fingerprint" (6 x N_max = 6 x 60 = 360-dim)
    Sort residues by radial distance (same ordering as PAF time axis).
    Concatenate feature vectors. Pad/truncate to fixed length.
    This preserves per-residue identity but uses no wave mapping.

If PAF substantially outperforms all three → wave transform captures real structure.
If PAF ≈ FP-B → radial binning is the useful part, FFT is cosmetic.
If PAF ≈ FP-A → even radial structure doesn't help, just composition matters.

Dependencies: numpy, scipy
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Feature extraction (identical to PAF — must use same features for fair test)
# ============================================================================

# These tables are duplicated from the PAF blueprint intentionally so this
# file is self-contained. In your actual repo, import from a shared module.

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

FEATURE_NAMES = ["charge", "hyd", "hb", "aro", "vol", "dyn"]
K_FEATURES = len(FEATURE_NAMES)  # 6


def aa_features(aa: str) -> Dict[str, float]:
    """Same feature extraction as PAF — must be identical for fair comparison."""
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
# Pocket data structure (lightweight, shared with PAF)
# ============================================================================

@dataclass
class ResidueEntry:
    """Minimal residue info needed for fingerprinting."""
    aa: str
    radial_A: float  # distance from pocket center
    features: Dict[str, float]  # charge, hyd, hb, aro, vol
    flex: float = 0.5  # dynamics proxy, 0..1


def residues_to_feature_matrix(residues: List[ResidueEntry]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert residues to (N, K) feature matrix + (N,) radial distances,
    sorted by radial distance (innermost first).
    """
    # Sort by radial distance
    sorted_res = sorted(residues, key=lambda r: r.radial_A)

    N = len(sorted_res)
    X = np.zeros((N, K_FEATURES), dtype=np.float32)
    radii = np.zeros(N, dtype=np.float32)

    for i, rr in enumerate(sorted_res):
        X[i, 0] = rr.features.get("charge", 0.0)
        X[i, 1] = rr.features.get("hyd", 0.0)
        X[i, 2] = rr.features.get("hb", 0.0)
        X[i, 3] = rr.features.get("aro", 0.0)
        X[i, 4] = rr.features.get("vol", 0.0)
        X[i, 5] = rr.flex
        radii[i] = rr.radial_A

    return X, radii


# ============================================================================
# Fingerprint A: Composition (simplest baseline)
# ============================================================================

def fp_composition(residues: List[ResidueEntry]) -> np.ndarray:
    """
    Mean of each feature across all pocket residues.
    Output: (K,) = (6,) vector.

    This is the absolute floor. If PAF can't beat this, the wave
    mapping is adding zero information.
    """
    X, _ = residues_to_feature_matrix(residues)
    if len(X) == 0:
        return np.zeros(K_FEATURES, dtype=np.float32)
    return X.mean(axis=0)  # (K,)


# ============================================================================
# Fingerprint B: Radial histogram (key comparator)
# ============================================================================

def fp_radial_histogram(
    residues: List[ResidueEntry],
    pocket_radius: float = 10.0,
    n_shells: int = 16,
) -> np.ndarray:
    """
    Bin residues by radial distance into n_shells concentric shells.
    Within each shell, compute mean feature vector.
    Output: (K * n_shells,) = (6 * 16 = 96,) vector.

    THIS IS THE CRITICAL ABLATION BASELINE.
    It captures the same radial-distance structure that PAF encodes
    via time-placement, but without wave mapping or FFT.

    If PAF ≈ FP-B in similarity correlation, the FFT is cosmetic.
    If PAF >> FP-B, the spectral transformation captures something
    that simple radial binning misses.
    """
    X, radii = residues_to_feature_matrix(residues)
    N = len(X)

    # Shell boundaries (uniform in distance)
    shell_edges = np.linspace(0, pocket_radius, n_shells + 1)

    fp = np.zeros((n_shells, K_FEATURES), dtype=np.float32)

    for s in range(n_shells):
        lo, hi = shell_edges[s], shell_edges[s + 1]
        mask = (radii >= lo) & (radii < hi)
        if s == n_shells - 1:
            # Include residues exactly at boundary
            mask = (radii >= lo) & (radii <= hi)
        count = int(mask.sum())
        if count > 0:
            fp[s, :] = X[mask].mean(axis=0)
        # else: zeros (empty shell)

    return fp.reshape(-1)  # (n_shells * K,)


# ============================================================================
# Fingerprint C: Sorted concatenation (preserves per-residue identity)
# ============================================================================

def fp_sorted_concat(
    residues: List[ResidueEntry],
    n_max: int = 60,
) -> np.ndarray:
    """
    Sort residues by radial distance, concatenate feature vectors.
    Pad with zeros if fewer than n_max; truncate if more.
    Output: (K * n_max,) = (6 * 60 = 360,) vector.

    This preserves individual residue features and their radial ordering
    but uses no wave transformation.
    """
    X, _ = residues_to_feature_matrix(residues)
    N = len(X)

    fp = np.zeros((n_max, K_FEATURES), dtype=np.float32)
    n_copy = min(N, n_max)
    fp[:n_copy, :] = X[:n_copy, :]

    return fp.reshape(-1)  # (n_max * K,)


# ============================================================================
# Fingerprint D: Radial histogram with LOG-SPACED shells
# (matches PAF's log-spaced frequency bins more closely)
# ============================================================================

def fp_radial_histogram_log(
    residues: List[ResidueEntry],
    pocket_radius: float = 10.0,
    n_shells: int = 16,
    r_min: float = 1.0,
) -> np.ndarray:
    """
    Like FP-B but with log-spaced shell boundaries.
    This more closely mirrors PAF's log-spaced frequency bins,
    making the comparison even more controlled.
    """
    X, radii = residues_to_feature_matrix(residues)

    shell_edges = np.logspace(np.log10(r_min), np.log10(pocket_radius), n_shells + 1)
    # First shell captures everything from 0 to r_min
    shell_edges[0] = 0.0

    fp = np.zeros((n_shells, K_FEATURES), dtype=np.float32)

    for s in range(n_shells):
        lo, hi = shell_edges[s], shell_edges[s + 1]
        mask = (radii >= lo) & (radii < hi)
        if s == n_shells - 1:
            mask = (radii >= lo) & (radii <= hi)
        if mask.sum() > 0:
            fp[s, :] = X[mask].mean(axis=0)

    return fp.reshape(-1)


# ============================================================================
# All fingerprints in one call (convenience)
# ============================================================================

def compute_all_fingerprints(
    residues: List[ResidueEntry],
    pocket_radius: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Compute all baseline fingerprints for one pocket.
    Returns dict mapping fingerprint name -> vector.
    """
    return {
        "FP-A_composition": fp_composition(residues),
        "FP-B_radial_hist": fp_radial_histogram(residues, pocket_radius=pocket_radius),
        "FP-B_radial_log": fp_radial_histogram_log(residues, pocket_radius=pocket_radius),
        "FP-C_sorted_concat": fp_sorted_concat(residues),
    }


# ============================================================================
# Similarity (same cosine as PAF, for fair comparison)
# ============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    return num / den


def similarity_matrix(embeddings: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    keys = list(embeddings.keys())
    n = len(keys)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            s = cosine_sim(embeddings[keys[i]], embeddings[keys[j]])
            M[i, j] = s
            M[j, i] = s
    return keys, M


# ============================================================================
# Usage example (standalone test with synthetic data)
# ============================================================================

def _make_synthetic_pocket(n_residues: int = 25, seed: int = 0) -> List[ResidueEntry]:
    """Generate a fake pocket for testing the pipeline."""
    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    residues = []
    for i in range(n_residues):
        aa = rng.choice(aas)
        radial = rng.uniform(2.0, 10.0)
        feats = aa_features(aa)
        flex = rng.uniform(0.2, 0.8)
        residues.append(ResidueEntry(aa=aa, radial_A=radial, features=feats, flex=flex))
    return residues


if __name__ == "__main__":
    print("=" * 60)
    print("Raw Fingerprint Baseline - Self-Test")
    print("=" * 60)

    # Make two synthetic pockets (should be somewhat similar)
    pocket_a = _make_synthetic_pocket(25, seed=0)
    pocket_b = _make_synthetic_pocket(25, seed=1)
    pocket_c = _make_synthetic_pocket(25, seed=99)  # more different

    fps_a = compute_all_fingerprints(pocket_a)
    fps_b = compute_all_fingerprints(pocket_b)
    fps_c = compute_all_fingerprints(pocket_c)

    print("\nFingerprint dimensions:")
    for name, vec in fps_a.items():
        print(f"  {name}: {vec.shape}")

    print("\nSimilarity A vs B (same-ish):")
    for name in fps_a:
        s = cosine_sim(fps_a[name], fps_b[name])
        print(f"  {name}: {s:.4f}")

    print("\nSimilarity A vs C (more different):")
    for name in fps_a:
        s = cosine_sim(fps_a[name], fps_c[name])
        print(f"  {name}: {s:.4f}")

    print("\n[OK] All fingerprints computed successfully.")
    print("Next: run head-to-head comparison with PAF on real kinase pockets.")
