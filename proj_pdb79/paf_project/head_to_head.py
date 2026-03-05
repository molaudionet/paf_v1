#!/usr/bin/env python3
"""
Head-to-Head: PAF vs Raw Fingerprints vs Structural Similarity
================================================================

This is the decisive experiment. It answers:

  "Does the wave-domain transformation add information
   beyond the raw residue features it's built from?"

Pipeline:
  1. Load kinase PDB structures
  2. Extract pockets (shared — identical residues for all methods)
  3. Compute PAF embedding (wave mapping + FFT + bins)
  4. Compute all raw fingerprints (same features, no wave)
  5. Compute structural similarity (pocket Cα RMSD or TM-score)
  6. For each method: correlate its similarity matrix with structural similarity
  7. Report Spearman correlations + confidence intervals
  8. Generate comparison figures

The method with highest Spearman ρ vs structural similarity wins.
If PAF wins → wave transform captures real structure beyond raw features.
If FP-B wins → radial binning alone is enough, FFT is overhead.
If FP-A wins → even composition dominates, no spatial info needed.

Dependencies:
  pip install numpy scipy matplotlib pandas biopython

Usage:
  python head_to_head.py --pdb_dir data/kinase_pdbs/ --pdb_list kinase_list.csv --out results/
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform

# ============================================================================
# Shared constants and feature tables
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
FEATURE_NAMES = ["charge", "hyd", "hb", "aro", "vol", "dyn"]
K_FEATURES = 6


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def aa_features(aa: str) -> Dict[str, float]:
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
# Pocket data structure (shared by all methods)
# ============================================================================

@dataclass
class ResidueRecord:
    aa: str
    ca_xyz: np.ndarray
    radial_A: float
    features: Dict[str, float]
    flex: float = 0.5


@dataclass
class Pocket:
    pocket_id: str
    center_xyz: np.ndarray
    residues: List[ResidueRecord]
    family: str = ""  # kinase family label for coloring


# ============================================================================
# PAF parameters (frozen v0)
# ============================================================================

@dataclass(frozen=True)
class PAFParams:
    sr: int = 16000
    duration_s: float = 6.0
    pocket_radius_A: float = 10.0
    fmin_hz: float = 150.0
    fmax_hz: float = 2400.0
    sigma_t_s: float = 0.040
    n_bins: int = 256
    eps: float = 1e-8
    a_hyd: float = 1.0
    a_charge: float = 1.0
    a_vol: float = 0.5


CHANNELS = ["chg", "hyd", "hb", "aro", "vol", "dyn"]


# ============================================================================
# PAF computation (from blueprint)
# ============================================================================

def paf_freq(hyd: float, charge: float, vol: float, p: PAFParams) -> float:
    z = p.a_hyd * hyd + p.a_charge * charge + p.a_vol * vol
    return p.fmin_hz + (p.fmax_hz - p.fmin_hz) * sigmoid(z)


def paf_embedding(pocket: Pocket, p: PAFParams) -> np.ndarray:
    """Compute PAF spectral embedding (K, B) for a pocket."""
    K = len(CHANNELS)
    n = int(p.sr * p.duration_s)
    t = np.arange(n, dtype=np.float64) / float(p.sr)
    R = p.pocket_radius_A
    sig = np.zeros((K, n), dtype=np.float64)

    for rr in pocket.residues:
        r = float(np.clip(rr.radial_A, 0.0, R))
        tau = p.duration_s * (r / R)
        q = rr.features.get("charge", 0.0)
        hyd = rr.features.get("hyd", 0.0)
        hb = rr.features.get("hb", 0.0)
        aro = rr.features.get("aro", 0.0)
        vol = rr.features.get("vol", 0.0)
        dyn = rr.flex

        f = paf_freq(hyd, q, vol, p)
        phi = 0.0 if q >= 0 else np.pi

        # Gaussian-windowed sinusoid
        x = t - tau
        win = np.exp(-(x * x) / (2.0 * p.sigma_t_s ** 2))
        wave = win * np.sin(2.0 * np.pi * f * x + phi)

        amps = {"chg": abs(q), "hyd": hyd, "hb": hb, "aro": aro, "vol": vol, "dyn": dyn}
        for k, name in enumerate(CHANNELS):
            a = amps[name]
            if a != 0.0:
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

    return spec.astype(np.float32)  # (K, B)


# ============================================================================
# Raw fingerprints (from baseline module)
# ============================================================================

def _to_matrix(pocket: Pocket) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted by radial distance, return (N, K) features + (N,) radii."""
    sorted_res = sorted(pocket.residues, key=lambda r: r.radial_A)
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


def fp_composition(pocket: Pocket) -> np.ndarray:
    X, _ = _to_matrix(pocket)
    return X.mean(axis=0) if len(X) > 0 else np.zeros(K_FEATURES, dtype=np.float32)


def fp_radial_hist(pocket: Pocket, n_shells: int = 16, R: float = 10.0) -> np.ndarray:
    X, radii = _to_matrix(pocket)
    edges = np.linspace(0, R, n_shells + 1)
    fp = np.zeros((n_shells, K_FEATURES), dtype=np.float32)
    for s in range(n_shells):
        lo, hi = edges[s], edges[s + 1]
        mask = (radii >= lo) & (radii < hi)
        if s == n_shells - 1:
            mask = (radii >= lo) & (radii <= hi)
        if mask.sum() > 0:
            fp[s, :] = X[mask].mean(axis=0)
    return fp.reshape(-1)


def fp_radial_hist_log(pocket: Pocket, n_shells: int = 16, R: float = 10.0) -> np.ndarray:
    X, radii = _to_matrix(pocket)
    edges = np.logspace(np.log10(1.0), np.log10(R), n_shells + 1)
    edges[0] = 0.0
    fp = np.zeros((n_shells, K_FEATURES), dtype=np.float32)
    for s in range(n_shells):
        lo, hi = edges[s], edges[s + 1]
        mask = (radii >= lo) & (radii < hi)
        if s == n_shells - 1:
            mask = (radii >= lo) & (radii <= hi)
        if mask.sum() > 0:
            fp[s, :] = X[mask].mean(axis=0)
    return fp.reshape(-1)


def fp_sorted_concat(pocket: Pocket, n_max: int = 60) -> np.ndarray:
    X, _ = _to_matrix(pocket)
    fp = np.zeros((n_max, K_FEATURES), dtype=np.float32)
    n_copy = min(len(X), n_max)
    fp[:n_copy, :] = X[:n_copy, :]
    return fp.reshape(-1)


# ============================================================================
# Structural similarity (pocket Cα RMSD after optimal superposition)
# ============================================================================

def _centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Optimal RMSD between two Nx3 coordinate sets after superposition.
    Uses Kabsch algorithm (SVD-based).
    Handles different sizes by truncating to min(N_P, N_Q) sorted by
    radial distance — this is a rough alignment, not perfect, but
    sufficient for "does PAF track structure?" experiments.
    """
    n = min(len(P), len(Q))
    if n < 3:
        return float("inf")

    # Use first n points (both are sorted by radial distance)
    P = P[:n].copy()
    Q = Q[:n].copy()

    # Center
    p_center = _centroid(P)
    q_center = _centroid(Q)
    P -= p_center
    Q -= q_center

    # Kabsch
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    P_rot = P @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1))))
    return rmsd


def pocket_ca_coords_sorted(pocket: Pocket) -> np.ndarray:
    """Get Cα coordinates sorted by radial distance."""
    sorted_res = sorted(pocket.residues, key=lambda r: r.radial_A)
    return np.array([rr.ca_xyz for rr in sorted_res], dtype=np.float64)


def structural_similarity_matrix(pockets: Dict[str, Pocket]) -> Tuple[List[str], np.ndarray]:
    """
    Compute pairwise structural similarity as 1/(1+RMSD).
    This converts RMSD (distance) to similarity in [0,1].
    """
    keys = list(pockets.keys())
    n = len(keys)
    coords = {k: pocket_ca_coords_sorted(pockets[k]) for k in keys}
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        M[i, i] = 1.0
        for j in range(i + 1, n):
            rmsd = kabsch_rmsd(coords[keys[i]], coords[keys[j]])
            sim = 1.0 / (1.0 + rmsd)
            M[i, j] = sim
            M[j, i] = sim
    return keys, M


# ============================================================================
# Cosine similarity matrix
# ============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def cosine_similarity_matrix(embeddings: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
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
# Spearman correlation between similarity matrices (upper triangle)
# ============================================================================

def upper_tri_values(M: np.ndarray) -> np.ndarray:
    """Extract upper triangle (excluding diagonal) as flat array."""
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def compare_to_structural(
    method_M: np.ndarray,
    struct_M: np.ndarray,
    method_name: str,
) -> Dict:
    """
    Spearman correlation between a method's similarity matrix and
    structural similarity matrix.
    """
    a = upper_tri_values(method_M)
    b = upper_tri_values(struct_M)
    rho, pval = spearmanr(a, b)

    # Bootstrap CI (1000 resamples)
    n_pairs = len(a)
    rng = np.random.default_rng(42)
    rhos_boot = []
    for _ in range(1000):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        r, _ = spearmanr(a[idx], b[idx])
        if not np.isnan(r):
            rhos_boot.append(r)
    ci_lo = float(np.percentile(rhos_boot, 2.5)) if rhos_boot else float("nan")
    ci_hi = float(np.percentile(rhos_boot, 97.5)) if rhos_boot else float("nan")

    return {
        "method": method_name,
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n_pairs": n_pairs,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison_bar(results: List[Dict], out_png: str):
    """Bar chart of Spearman ρ for each method vs structural similarity."""
    methods = [r["method"] for r in results]
    rhos = [r["spearman_rho"] for r in results]
    ci_lo = [r["spearman_rho"] - r["ci95_lo"] for r in results]
    ci_hi = [r["ci95_hi"] - r["spearman_rho"] for r in results]

    colors = []
    for m in methods:
        if "PAF" in m:
            colors.append("#2196F3")
        elif "composition" in m.lower():
            colors.append("#9E9E9E")
        else:
            colors.append("#FF9800")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, rhos, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, rhos, yerr=[ci_lo, ci_hi], fmt="none", capsize=5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Spearman ρ vs Structural Similarity", fontsize=11)
    ax.set_title("PAF vs Raw Fingerprints: Correlation with Pocket Structure", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out_png}")


def plot_scatter_grid(
    method_matrices: Dict[str, np.ndarray],
    struct_M: np.ndarray,
    out_png: str,
):
    """Scatter plots: each method's similarity vs structural similarity."""
    struct_vals = upper_tri_values(struct_M)
    methods = list(method_matrices.keys())
    n_methods = len(methods)

    cols = min(3, n_methods)
    rows = math.ceil(n_methods / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    for idx, method in enumerate(methods):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        method_vals = upper_tri_values(method_matrices[method])
        rho, _ = spearmanr(method_vals, struct_vals)

        color = "#2196F3" if "PAF" in method else "#FF9800"
        ax.scatter(struct_vals, method_vals, alpha=0.3, s=8, color=color)
        ax.set_xlabel("Structural similarity", fontsize=9)
        ax.set_ylabel(f"{method} similarity", fontsize=9)
        ax.set_title(f"{method}\nρ = {rho:.3f}", fontsize=10)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(n_methods, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out_png}")


def plot_pca_colored(
    embeddings: Dict[str, np.ndarray],
    families: Dict[str, str],
    method_name: str,
    out_png: str,
):
    """PCA projection colored by kinase family."""
    keys = list(embeddings.keys())
    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]

    # Color by family
    fam_list = [families.get(k, "unknown") for k in keys]
    unique_fams = sorted(set(fam_list))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_fams), 1))
    fam_to_color = {f: cmap(i) for i, f in enumerate(unique_fams)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(keys):
        fam = fam_list[i]
        ax.scatter(Z[i, 0], Z[i, 1], color=fam_to_color[fam], s=40, edgecolors="black", linewidth=0.3)
        ax.annotate(k.split("|")[0][:6], (Z[i, 0], Z[i, 1]), fontsize=5, alpha=0.7)

    # Legend
    for fam in unique_fams:
        ax.scatter([], [], color=fam_to_color[fam], label=fam, s=40)
    ax.legend(fontsize=8, loc="best", title="Kinase family")
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(f"PCA of {method_name} embeddings", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {out_png}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_head_to_head(pockets: Dict[str, Pocket], out_dir: str):
    """
    Run full comparison: PAF vs all raw fingerprints vs structural similarity.
    """
    os.makedirs(out_dir, exist_ok=True)
    params = PAFParams()
    keys = list(pockets.keys())
    n = len(keys)

    print(f"\n{'='*60}")
    print(f"Head-to-Head Comparison: {n} pockets")
    print(f"{'='*60}")

    # --- Compute all embeddings ---
    print("\n[1/4] Computing embeddings...")
    emb_paf = {}
    emb_fpa = {}
    emb_fpb = {}
    emb_fpb_log = {}
    emb_fpc = {}

    for k in keys:
        pocket = pockets[k]
        emb_paf[k] = paf_embedding(pocket, params)
        emb_fpa[k] = fp_composition(pocket)
        emb_fpb[k] = fp_radial_hist(pocket, R=params.pocket_radius_A)
        emb_fpb_log[k] = fp_radial_hist_log(pocket, R=params.pocket_radius_A)
        emb_fpc[k] = fp_sorted_concat(pocket)

    all_methods = {
        "PAF (wave+FFT)": emb_paf,
        "FP-A (composition)": emb_fpa,
        "FP-B (radial hist)": emb_fpb,
        "FP-B-log (radial log)": emb_fpb_log,
        "FP-C (sorted concat)": emb_fpc,
    }

    # --- Compute similarity matrices ---
    print("[2/4] Computing similarity matrices...")
    method_sim = {}
    for name, embs in all_methods.items():
        _, M = cosine_similarity_matrix(embs)
        method_sim[name] = M

    _, struct_M = structural_similarity_matrix(pockets)

    # --- Compare ---
    print("[3/4] Computing Spearman correlations...")
    results = []
    for name, M in method_sim.items():
        r = compare_to_structural(M, struct_M, name)
        results.append(r)
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"  {name:25s}  ρ={r['spearman_rho']:.4f}  "
              f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]  p={r['p_value']:.2e}  {sig}")

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(out_dir, "comparison_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n  Results table: {results_csv}")

    # --- Figures ---
    print("[4/4] Generating figures...")
    plot_comparison_bar(results, os.path.join(out_dir, "fig_comparison_bar.png"))
    plot_scatter_grid(method_sim, struct_M, os.path.join(out_dir, "fig_scatter_grid.png"))

    # PCA for PAF and best fingerprint
    families = {k: pockets[k].family for k in keys}
    plot_pca_colored(emb_paf, families, "PAF", os.path.join(out_dir, "fig_pca_paf.png"))
    plot_pca_colored(emb_fpb, families, "FP-B (radial hist)", os.path.join(out_dir, "fig_pca_fpb.png"))

    # Save similarity matrices
    np.save(os.path.join(out_dir, "sim_structural.npy"), struct_M)
    for name, M in method_sim.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        np.save(os.path.join(out_dir, f"sim_{safe_name}.npy"), M)
    with open(os.path.join(out_dir, "pocket_keys.json"), "w") as f:
        json.dump(keys, f, indent=2)

    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*60}")
    print("""
  If PAF ρ >> FP-B ρ:
    → Wave-domain transformation captures structure beyond radial binning.
    → PAF formalism is justified. Proceed with confidence.

  If PAF ρ ≈ FP-B ρ:
    → Radial structure is the useful signal; FFT adds overhead but no info.
    → Revise wave mapping (try different kernels, add angular info) before publishing.
    → Or: reframe PAF as "a principled way to do radial feature binning."

  If PAF ρ ≈ FP-A ρ:
    → Neither radial structure nor wave transform helps.
    → Pocket composition dominates. Need richer features or atom-level encoding.

  If PAF ρ < FP-A ρ:
    → Wave transform is actively losing information. Debug the mapping.

  If ALL methods ρ < 0.1:
    → These 6 features are too coarse for kinase pocket discrimination.
    → Consider: sequence-derived features, atom-level, or side-chain geometry.
    """)

    return results


# ============================================================================
# Synthetic test (validates pipeline without real PDBs)
# ============================================================================

def make_synthetic_test_pockets(n_pockets: int = 20) -> Dict[str, Pocket]:
    """
    Generate synthetic pockets for pipeline testing.
    Three "families" with different feature distributions.
    """
    rng = np.random.default_rng(42)
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    families = {
        "TK": {"hydro_bias": 0.3, "charge_bias": 0.0},
        "CMGC": {"hydro_bias": -0.2, "charge_bias": 0.3},
        "AGC": {"hydro_bias": 0.0, "charge_bias": -0.2},
    }
    fam_names = list(families.keys())

    pockets = {}
    for i in range(n_pockets):
        fam = fam_names[i % len(fam_names)]
        fam_params = families[fam]
        n_res = rng.integers(15, 40)
        center = rng.normal(0, 1, 3).astype(np.float32)

        residues = []
        for j in range(n_res):
            aa = rng.choice(aa_list)
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-8
            r = rng.uniform(2.0, 10.0)
            ca = center + direction * r
            feats = aa_features(aa)
            # Apply family bias (simulates different pocket compositions)
            feats["hyd"] = float(np.clip(feats["hyd"] + fam_params["hydro_bias"] + rng.normal(0, 0.05), 0, 1))
            feats["charge"] = feats["charge"] + fam_params["charge_bias"]
            flex = rng.uniform(0.2, 0.8)
            residues.append(ResidueRecord(
                aa=aa, ca_xyz=ca.astype(np.float32),
                radial_A=r, features=feats, flex=flex,
            ))

        pid = f"synth_{fam}_{i:03d}"
        pockets[pid] = Pocket(
            pocket_id=pid, center_xyz=center,
            residues=residues, family=fam,
        )

    return pockets


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAF vs Raw Fingerprints Head-to-Head")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                        help="'synthetic' for pipeline test, 'real' for PDB structures")
    parser.add_argument("--pdb_list", type=str, default=None,
                        help="CSV with pdb_path, chain_id, pocket_label columns (for --mode real)")
    parser.add_argument("--out", type=str, default="results/head_to_head",
                        help="Output directory")
    args = parser.parse_args()

    if args.mode == "synthetic":
        print("Running synthetic pipeline test (no PDB files needed)...")
        pockets = make_synthetic_test_pockets(n_pockets=30)
        run_head_to_head(pockets, args.out)
    else:
        # Real PDB mode — requires biopython + PDB extraction
        # You would integrate the pocket extraction from the PAF blueprint here
        print("Real PDB mode: implement pocket loading from extract_pocket() + your PDB list")
        print("See PAF code blueprint for extract_pocket() function.")
        sys.exit(1)
