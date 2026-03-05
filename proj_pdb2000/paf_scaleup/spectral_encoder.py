#!/usr/bin/env python3
"""
spectral_encoder.py — Adapter wrapping paf_core_v1.py for the scale-up pipeline.

This module provides the interface that run_experiments.py expects,
using the REAL PAF encoder from paf_core_v1.py.

Key differences from the previous (incorrect) reconstruction:
  - 96,000 waveform samples (sr=16000 × 6s), not 512
  - Sigmoid frequency mapping [150, 2400] Hz, not linear
  - Two-anchor geometry (r1→time, r2→FM), not single-anchor
  - Log-spaced frequency bins, not linear FFT bins
  - Per-channel sum-normalization, not global L2
  - Phase = 0 or π based on charge sign
  - Gaussian × sin(2πft + φ), not cos(ωt + φ)
  - BioPython parsing with DSSP support
"""

import os
import sys
import numpy as np
from typing import Optional, Dict, List, Tuple

# Import the real PAF encoder
try:
    from paf_core_v1 import (
        PAFParams,
        extract_pocket as paf_extract_pocket,
        pocket_to_embedding as paf_pocket_to_embedding,
        embed_pocket,
        load_structure,
        ligand_centroid,
        aa_physchem,
        blosum62_row,
        AA_ORDER,
    )
    HAS_PAF_CORE = True
except ImportError:
    HAS_PAF_CORE = False
    print("WARNING: paf_core_v1.py not found. Place it in the same directory.")
    print("         Only baseline methods (mean, radial) will work.")


# ── Default PAF parameters (matching paf_core_v1 defaults) ───────────────────

DEFAULT_PAF_PARAMS = PAFParams() if HAS_PAF_CORE else None


# ── Core encoding functions ──────────────────────────────────────────────────

def spectral_encode_from_pdb(
    pdb_path: str,
    chain: str = "A",
    ligand_resname: Optional[str] = None,
    params: Optional[object] = None,
    flatten: bool = True,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Compute PAF spectral embedding directly from a PDB file.

    Returns
    -------
    (embedding, meta) or None if extraction fails.

    embedding : np.ndarray
        If flatten=True: shape (K * n_bins,) = (7680,) for 30 channels × 256 bins.
        If flatten=False: shape (K, n_bins) = (30, 256).
    meta : dict
        Metadata from extraction (center method, n_residues, etc.)
    """
    if not HAS_PAF_CORE:
        return None

    if params is None:
        params = DEFAULT_PAF_PARAMS

    try:
        E_mag, meta = embed_pocket(
            pdb_path=pdb_path,
            chain=chain,
            ligand_resname=ligand_resname or "",
            radius=params.pocket_radius_A,
            gamma_fm=params.gamma_fm,
            sigma_t=params.sigma_t_s,
            a_hyd=params.a_hyd,
            a_charge=params.a_charge,
            a_vol=params.a_vol,
        )
        # E_mag shape: (K, n_bins) = (30, 256)

        if flatten:
            emb = E_mag.flatten().astype(np.float64)
            # L2 normalize the flattened embedding for cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            return emb, meta
        else:
            return E_mag, meta

    except Exception as e:
        return None


def spectral_encode_batch(
    entries: List[Dict],
    pdb_dir: str,
    params: Optional[object] = None,
    flatten: bool = True,
    verbose: bool = True,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[dict]]]:
    """
    Batch encode pockets from PDB files using the real PAF encoder.

    Parameters
    ----------
    entries : list of dict
        Each has 'pdb_id', optionally 'chain', 'ligand_resname'.
    pdb_dir : str
        Directory containing {pdb_id}.pdb files.

    Returns
    -------
    embeddings : list of np.ndarray or None
    metas : list of dict or None
    """
    embeddings = []
    metas = []
    success = 0
    fail = 0

    for i, entry in enumerate(entries):
        pdb_id = entry["pdb_id"].lower()
        chain = entry.get("chain", "A") or "A"
        ligand_resname = entry.get("ligand_resname", "") or ""

        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            embeddings.append(None)
            metas.append(None)
            fail += 1
            continue

        result = spectral_encode_from_pdb(
            pdb_path, chain=chain, ligand_resname=ligand_resname,
            params=params, flatten=flatten,
        )

        if result is not None:
            emb, meta = result
            embeddings.append(emb)
            metas.append(meta)
            success += 1
        else:
            embeddings.append(None)
            metas.append(None)
            fail += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    Encoded {i+1}/{len(entries)} "
                  f"(success={success}, fail={fail})")

    if verbose:
        print(f"    Done: {success} encoded, {fail} failed")

    return embeddings, metas


# ── Baseline methods (use features extracted by paf_core_v1) ─────────────────

def extract_features_from_pdb(
    pdb_path: str,
    chain: str = "A",
    ligand_resname: Optional[str] = None,
    params: Optional[object] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract (coords, features) from a PDB file using paf_core_v1's parser.

    Returns (coords, features) with shapes (N,3) and (N,30).
    Uses BioPython for robust parsing and the same feature computation
    as the real encoder.
    """
    if not HAS_PAF_CORE:
        return None

    if params is None:
        params = DEFAULT_PAF_PARAMS

    try:
        pocket = paf_extract_pocket(
            pdb_path=pdb_path,
            chain_id=chain,
            ligand_resname=ligand_resname or None,
            params=params,
        )

        N = len(pocket.residues)
        coords = np.zeros((N, 3), dtype=np.float64)
        features = np.zeros((N, 30), dtype=np.float64)

        for i, rr in enumerate(pocket.residues):
            coords[i] = rr.ca_xyz

            # [0:5] physicochemical
            features[i, 0] = rr.physchem["charge"]
            features[i, 1] = rr.physchem["hyd"]
            features[i, 2] = rr.physchem["hb"]
            features[i, 3] = rr.physchem["aro"]
            features[i, 4] = rr.physchem["vol"]

            # [5] flexibility
            features[i, 5] = rr.flex

            # [6] contact density
            features[i, 6] = rr.contact

            # [7:10] secondary structure
            features[i, 7] = rr.ss_onehot[0]  # helix
            features[i, 8] = rr.ss_onehot[1]  # sheet
            features[i, 9] = rr.ss_onehot[2]  # coil

            # [10:30] BLOSUM62 profile
            features[i, 10:30] = rr.blosum

        return coords, features

    except Exception:
        return None


def mean_aggregation_embedding(features: np.ndarray) -> np.ndarray:
    """Baseline: mean pooling across residues → L2-normalized."""
    emb = features.mean(axis=0)
    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb = emb / norm
    return emb


def radial_histogram_embedding(
    coords: np.ndarray,
    features: np.ndarray,
    n_shells: int = 16,
) -> np.ndarray:
    """Baseline: radial histogram binning → L2-normalized."""
    centroid = coords.mean(axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    max_dist = dists.max() + 1e-8
    D = features.shape[1]

    hist = np.zeros(n_shells * D)
    for i, d in enumerate(dists):
        shell = min(int(d / max_dist * n_shells), n_shells - 1)
        hist[shell * D : (shell + 1) * D] += features[i]

    norm = np.linalg.norm(hist)
    if norm > 1e-12:
        hist = hist / norm
    return hist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


# ── Batch encoding with all three methods ────────────────────────────────────

def encode_all_methods(
    entries: List[Dict],
    pdb_dir: str,
    params: Optional[object] = None,
    verbose: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Encode all pockets using spectral, mean, and radial methods.

    Returns dict mapping method name to (embeddings, labels, pdb_ids)
    where only successfully encoded pockets are included.
    All three methods share the same set of valid pockets.
    """
    if params is None and HAS_PAF_CORE:
        params = DEFAULT_PAF_PARAMS

    spectral_embs = []
    mean_embs = []
    radial_embs = []
    valid_labels = []
    valid_pdb_ids = []
    valid_entries = []

    success = 0
    fail = 0

    for i, entry in enumerate(entries):
        pdb_id = entry["pdb_id"].lower()
        chain = entry.get("chain", "A") or "A"
        ligand_resname = entry.get("ligand_resname", "") or ""
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

        if not os.path.exists(pdb_path):
            fail += 1
            continue

        # Get spectral embedding
        spec_result = spectral_encode_from_pdb(
            pdb_path, chain=chain, ligand_resname=ligand_resname,
            params=params, flatten=True,
        )

        if spec_result is None:
            fail += 1
            continue

        # Get features for baselines (using the same pocket extraction)
        feat_result = extract_features_from_pdb(
            pdb_path, chain=chain, ligand_resname=ligand_resname,
            params=params,
        )

        if feat_result is None:
            fail += 1
            continue

        spec_emb, meta = spec_result
        coords, features = feat_result

        spectral_embs.append(spec_emb)
        mean_embs.append(mean_aggregation_embedding(features))
        radial_embs.append(radial_histogram_embedding(coords, features))
        valid_entries.append(entry)
        valid_pdb_ids.append(pdb_id)
        success += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(entries)} "
                  f"(success={success}, fail={fail})")

    if verbose:
        print(f"    Done: {success} encoded, {fail} failed out of {len(entries)}")

    if not spectral_embs:
        return {}

    return {
        "spectral": np.array(spectral_embs),
        "mean": np.array(mean_embs),
        "radial": np.array(radial_embs),
        "valid_entries": valid_entries,
        "valid_pdb_ids": valid_pdb_ids,
    }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if HAS_PAF_CORE:
        print("paf_core_v1 loaded successfully.")
        print(f"Default params: sr={DEFAULT_PAF_PARAMS.sr}, "
              f"duration={DEFAULT_PAF_PARAMS.duration_s}s, "
              f"fmin={DEFAULT_PAF_PARAMS.fmin_hz}Hz, "
              f"fmax={DEFAULT_PAF_PARAMS.fmax_hz}Hz, "
              f"n_bins={DEFAULT_PAF_PARAMS.n_bins}, "
              f"sigma_t={DEFAULT_PAF_PARAMS.sigma_t_s}s")
        print(f"Waveform samples: {DEFAULT_PAF_PARAMS.sr * DEFAULT_PAF_PARAMS.duration_s:.0f}")
    else:
        print("ERROR: paf_core_v1.py not found!")
        print("Copy your paf_core_v1.py into this directory.")
