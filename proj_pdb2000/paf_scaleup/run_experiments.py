#!/usr/bin/env python3
"""
run_experiments.py — Run PAF experiments at scale with the REAL encoder.

Changes from v1:
  - Uses paf_core_v1.py via spectral_encoder.py adapter (real PAF, not reconstruction)
  - PCA dimensionality reduction sweep (50, 100, 200, 500, no-PCA)
  - Balanced accuracy for imbalanced classes (DFG, subfamily)
  - Stratified permutation tests
  - Single extraction pass shared across all three methods

Usage:
  python run_experiments.py --experiment cross_family \
    --manifest data/cross_family_manifest.csv \
    --pdb_dir data/pdbs/ \
    --out_dir results/

  python run_experiments.py --experiment all \
    --manifest_dir data/ \
    --pdb_dir data/pdbs/ \
    --out_dir results/
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

from spectral_encoder import encode_all_methods

# Try sklearn for PCA + balanced accuracy
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not found. PCA sweep disabled.")
    print("  pip install scikit-learn")


# ── Evaluation Metrics ───────────────────────────────────────────────────────

def loo_1nn_classify(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Leave-one-out 1-nearest-neighbor classification using cosine similarity."""
    N = len(embeddings)
    predictions = np.empty(N, dtype=labels.dtype)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    for i in range(N):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        nn = np.argmax(sims)
        predictions[i] = labels[nn]

    accuracy = float(np.mean(predictions == labels))
    return accuracy, predictions


def balanced_acc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy (mean of per-class recall). Handles imbalanced classes."""
    if HAS_SKLEARN:
        return float(balanced_accuracy_score(labels, predictions))
    classes = sorted(set(labels))
    recalls = []
    for c in classes:
        mask = labels == c
        if mask.sum() > 0:
            recalls.append(float(np.mean(predictions[mask] == c)))
    return float(np.mean(recalls)) if recalls else 0.0


def cohens_d(within_sims: np.ndarray, between_sims: np.ndarray) -> float:
    """Cohen's d effect size."""
    if len(within_sims) < 2 or len(between_sims) < 2:
        return 0.0
    m1, m2 = within_sims.mean(), between_sims.mean()
    s1, s2 = within_sims.std(ddof=1), between_sims.std(ddof=1)
    n1, n2 = len(within_sims), len(between_sims)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std < 1e-12:
        return 0.0
    return float((m1 - m2) / pooled_std)


def within_between_analysis(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute within/between class similarity statistics."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    within_sims = []
    between_sims = []
    N = len(labels)
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    within_sims = np.array(within_sims) if within_sims else np.array([0.0])
    between_sims = np.array(between_sims) if between_sims else np.array([0.0])

    return {
        "within_mean": float(within_sims.mean()),
        "within_std": float(within_sims.std()),
        "between_mean": float(between_sims.mean()),
        "between_std": float(between_sims.std()),
        "cohens_d": cohens_d(within_sims, between_sims),
        "n_within_pairs": len(within_sims),
        "n_between_pairs": len(between_sims),
    }


def permutation_test(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 10000,
    use_balanced: bool = False,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Permutation test. Returns (observed, p_value, random_baseline).
    If use_balanced=True, uses balanced accuracy (for imbalanced classes).
    """
    rng = np.random.RandomState(seed)
    obs_acc, obs_preds = loo_1nn_classify(embeddings, labels)

    if use_balanced:
        observed = balanced_acc(obs_preds, labels)
    else:
        observed = obs_acc

    n_labels = len(set(labels))
    if use_balanced:
        random_baseline = 1.0 / n_labels
    else:
        counts = Counter(labels)
        total = len(labels)
        random_baseline = sum((n/total)**2 for n in counts.values())

    perm_scores = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_labels = rng.permutation(labels)
        _, perm_preds = loo_1nn_classify(embeddings, perm_labels)
        if use_balanced:
            perm_scores[p] = balanced_acc(perm_preds, perm_labels)
        else:
            perm_scores[p] = np.mean(perm_preds == perm_labels)

    p_value = float(np.mean(perm_scores >= observed))
    return observed, p_value, random_baseline


def per_class_accuracy(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """Per-class recall breakdown."""
    classes = sorted(set(labels))
    result = {}
    for c in classes:
        mask = labels == c
        if mask.sum() > 0:
            result[str(c)] = {
                "recall": float(np.mean(predictions[mask] == c)),
                "n": int(mask.sum()),
            }
    return result


# ── PCA Sweep ────────────────────────────────────────────────────────────────

def pca_sweep(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components_list: List[int] = [30, 50, 100, 200, 500],
) -> Dict:
    """Sweep PCA dimensions. Returns dict mapping key to results."""
    if not HAS_SKLEARN:
        acc, preds = loo_1nn_classify(embeddings, labels)
        return {"no_pca": {"n_components": embeddings.shape[1],
                           "accuracy": acc,
                           "balanced_accuracy": balanced_acc(preds, labels)}}

    results = {}
    max_components = min(embeddings.shape[0], embeddings.shape[1])

    # No PCA baseline
    acc, preds = loo_1nn_classify(embeddings, labels)
    results["no_pca"] = {
        "n_components": embeddings.shape[1],
        "accuracy": acc,
        "balanced_accuracy": balanced_acc(preds, labels),
        "variance_explained": 1.0,
    }

    for n_comp in n_components_list:
        if n_comp >= max_components:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        emb_pca = pca.fit_transform(embeddings)
        var_exp = float(pca.explained_variance_ratio_.sum())
        acc_p, preds_p = loo_1nn_classify(emb_pca, labels)

        results[f"pca_{n_comp}"] = {
            "n_components": n_comp,
            "accuracy": acc_p,
            "balanced_accuracy": balanced_acc(preds_p, labels),
            "variance_explained": var_exp,
        }

    return results


def apply_best_pca(embeddings, labels, pca_results, metric="accuracy"):
    """Apply PCA using best setting from sweep. Returns (embs, n_components_used)."""
    best_key = max(pca_results, key=lambda k: pca_results[k][metric])
    if best_key == "no_pca":
        return embeddings, None
    n_comp = pca_results[best_key]["n_components"]
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(embeddings), n_comp


# ── Experiment: Cross-Family ─────────────────────────────────────────────────

def run_cross_family_experiment(
    manifest_path: str,
    pdb_dir: str,
    out_dir: str,
    n_permutations: int = 10000,
) -> Dict:
    print("\n" + "="*70)
    print("  EXPERIMENT: Cross-Family Pocket Classification")
    print("="*70)

    entries = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            entries.append(row)
    print(f"\n  Loaded {len(entries)} entries")

    # Encode all methods in one pass
    print("\n  Encoding pockets (spectral + mean + radial)...")
    t0 = time.time()
    encoded = encode_all_methods(entries, pdb_dir, verbose=True)
    t_total = time.time() - t0

    if not encoded:
        print("  ERROR: No pockets encoded.")
        return {}

    valid_entries = encoded["valid_entries"]
    print(f"\n  Encoding time: {t_total:.1f}s ({len(valid_entries)} pockets)")

    # Family distribution
    family_counts = Counter(e["family"] for e in valid_entries)
    print("\n  Family distribution:")
    for fam, n in family_counts.most_common():
        print(f"    {fam:30s}: {n:4d}")

    # Filter families with >= 5 members
    min_size = 5
    valid_families = {f for f, n in family_counts.items() if n >= min_size}
    mask = np.array([e["family"] in valid_families for e in valid_entries])
    labels = np.array([e["family"] for e in valid_entries])[mask]
    n_families = len(valid_families)
    print(f"\n  After filtering (>={min_size}/family): {mask.sum()} structures, {n_families} families")

    results = {}

    for method in ["spectral", "mean", "radial"]:
        embs = encoded[method][mask]
        print(f"\n  ── {method.upper()} (dim={embs.shape[1]}) ──")

        # PCA sweep for high-dimensional spectral embeddings
        used_pca = None
        if method == "spectral" and HAS_SKLEARN and embs.shape[1] > 500:
            print(f"    PCA sweep...")
            pca_results = pca_sweep(embs, labels, [30, 50, 100, 200, 500])
            results[f"{method}_pca_sweep"] = pca_results
            for k, v in sorted(pca_results.items()):
                print(f"      {k:12s}: acc={v['accuracy']:.3f}  "
                      f"bacc={v['balanced_accuracy']:.3f}  "
                      f"var={v['variance_explained']:.3f}")
            embs_eval, used_pca = apply_best_pca(embs, labels, pca_results)
            if used_pca:
                print(f"    Using PCA({used_pca})")
        else:
            embs_eval = embs

        # Evaluate
        acc, preds = loo_1nn_classify(embs_eval, labels)
        bacc = balanced_acc(preds, labels)
        wb = within_between_analysis(embs_eval, labels)

        print(f"    Permutation test ({n_permutations} perms)...")
        obs, p_value, rand_bl = permutation_test(embs_eval, labels, n_permutations)

        class_acc = per_class_accuracy(preds, labels)

        results[method] = {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "cohens_d": wb["cohens_d"],
            "p_value": p_value,
            "random_baseline": rand_bl,
            "fold_vs_random": acc / rand_bl if rand_bl > 0 else 0,
            "within_between": wb,
            "per_class_accuracy": class_acc,
            "n_structures": int(mask.sum()),
            "n_families": n_families,
            "embedding_dim_raw": encoded[method].shape[1],
            "pca_components": used_pca,
        }

        print(f"    Accuracy:          {acc:.3f}")
        print(f"    Balanced accuracy: {bacc:.3f}")
        print(f"    Cohen's d:         {wb['cohens_d']:.3f}")
        print(f"    p-value:           {p_value:.2e}")
        print(f"    Fold vs random:    {acc/rand_bl:.2f}x")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cross_family_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    np.savez_compressed(
        os.path.join(out_dir, "cross_family_embeddings.npz"),
        spectral=encoded["spectral"][mask],
        mean=encoded["mean"][mask],
        radial=encoded["radial"][mask],
        labels=labels,
        pdb_ids=np.array([e["pdb_id"] for e in valid_entries])[mask],
    )

    # Summary
    print("\n" + "="*70)
    print(f"  RESULTS: Cross-Family ({mask.sum()} structures, {n_families} families)")
    print("="*70)
    print(f"  {'Method':<25} {'Acc':>7} {'BalAcc':>7} {'d':>8} {'p':>12} {'Fold':>7}")
    print(f"  {'-'*68}")
    for method in ["spectral", "mean", "radial"]:
        r = results[method]
        tag = f" (PCA={r['pca_components']})" if r.get('pca_components') else ""
        print(f"  {method+tag:<25} {r['accuracy']:>7.3f} {r['balanced_accuracy']:>7.3f} "
              f"{r['cohens_d']:>8.3f} {r['p_value']:>12.2e} {r['fold_vs_random']:>6.2f}x")

    return results


# ── Experiment: Kinase ───────────────────────────────────────────────────────

def run_kinase_experiment(
    manifest_path: str,
    pdb_dir: str,
    out_dir: str,
    n_permutations: int = 10000,
) -> Dict:
    print("\n" + "="*70)
    print("  EXPERIMENT: Kinase Analysis")
    print("="*70)

    entries = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            entries.append(row)
    print(f"\n  Loaded {len(entries)} kinase entries")

    print("\n  Encoding pockets...")
    encoded = encode_all_methods(entries, pdb_dir, verbose=True)
    if not encoded:
        return {}

    valid_entries = encoded["valid_entries"]
    results = {}

    # ── Subfamily ──
    sf_indices = [i for i, e in enumerate(valid_entries) if e.get("subfamily", "").strip()]
    if len(sf_indices) >= 20:
        sf_labels_raw = np.array([valid_entries[i]["subfamily"] for i in sf_indices])
        sf_counts = Counter(sf_labels_raw)
        valid_sfs = {sf for sf, n in sf_counts.items() if n >= 3}
        sf_keep = np.array([l in valid_sfs for l in sf_labels_raw])

        if sf_keep.sum() >= 20:
            sf_idx = np.array(sf_indices)[sf_keep]
            sf_labels = sf_labels_raw[sf_keep]
            print(f"\n  Subfamily: {len(sf_idx)} structures, {len(valid_sfs)} subfamilies")

            for method in ["spectral", "mean", "radial"]:
                embs = encoded[method][sf_idx]

                if method == "spectral" and HAS_SKLEARN and embs.shape[1] > 500:
                    pca_res = pca_sweep(embs, sf_labels, [30, 50, 100, 200])
                    embs, n_pca = apply_best_pca(embs, sf_labels, pca_res)
                    results[f"subfamily_spectral_pca_sweep"] = pca_res

                acc, preds = loo_1nn_classify(embs, sf_labels)
                bacc = balanced_acc(preds, sf_labels)
                wb = within_between_analysis(embs, sf_labels)
                _, p_val, rand_bl = permutation_test(embs, sf_labels, n_permutations, use_balanced=True)

                results[f"subfamily_{method}"] = {
                    "accuracy": acc, "balanced_accuracy": bacc,
                    "cohens_d": wb["cohens_d"], "p_value": p_val,
                    "n_structures": int(sf_keep.sum()), "n_subfamilies": len(valid_sfs),
                }
                print(f"    {method:15s}: acc={acc:.3f}  bacc={bacc:.3f}  d={wb['cohens_d']:.3f}  p={p_val:.2e}")

    # ── DFG ──
    dfg_indices = []
    dfg_labels_list = []
    for i, e in enumerate(valid_entries):
        dfg = (e.get("dfg_state", "") or "").strip().lower()
        if dfg in ("in", "dfg-in"):
            dfg_indices.append(i)
            dfg_labels_list.append("in")
        elif dfg in ("out", "dfg-out"):
            dfg_indices.append(i)
            dfg_labels_list.append("out")

    if len(dfg_indices) >= 10:
        dfg_labels = np.array(dfg_labels_list)
        dfg_idx = np.array(dfg_indices)
        dfg_counts = Counter(dfg_labels)
        print(f"\n  DFG: {len(dfg_idx)} structures (in={dfg_counts.get('in',0)}, out={dfg_counts.get('out',0)})")

        if min(dfg_counts.values()) >= 3:
            for method in ["spectral", "mean", "radial"]:
                embs = encoded[method][dfg_idx]

                if method == "spectral" and HAS_SKLEARN and embs.shape[1] > 500:
                    pca_res = pca_sweep(embs, dfg_labels, [30, 50, 100])
                    embs, _ = apply_best_pca(embs, dfg_labels, pca_res, metric="balanced_accuracy")

                acc, preds = loo_1nn_classify(embs, dfg_labels)
                bacc = balanced_acc(preds, dfg_labels)
                wb = within_between_analysis(embs, dfg_labels)
                _, p_val, _ = permutation_test(embs, dfg_labels, n_permutations, use_balanced=True)
                class_acc = per_class_accuracy(preds, dfg_labels)

                results[f"dfg_{method}"] = {
                    "accuracy": acc, "balanced_accuracy": bacc,
                    "cohens_d": wb["cohens_d"], "p_value": p_val,
                    "n_in": int(dfg_counts.get("in", 0)),
                    "n_out": int(dfg_counts.get("out", 0)),
                    "per_class": class_acc,
                }
                rin = class_acc.get("in", {}).get("recall", 0)
                rout = class_acc.get("out", {}).get("recall", 0)
                print(f"    {method:15s}: acc={acc:.3f}  bacc={bacc:.3f}  d={wb['cohens_d']:.3f}  "
                      f"p={p_val:.2e}  in={rin:.3f}  out={rout:.3f}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kinase_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {os.path.join(out_dir, 'kinase_results.json')}")
    return results


# ── Benchmark ────────────────────────────────────────────────────────────────

def run_timing_benchmark(manifest_path, pdb_dir, out_dir):
    print("\n" + "="*70)
    print("  BENCHMARK: Encoding Speed")
    print("="*70)

    entries = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            entries.append(row)

    subset = entries[:200]
    print(f"\n  Timing on {len(subset)} entries...")

    t0 = time.time()
    encoded = encode_all_methods(subset, pdb_dir, verbose=False)
    t_total = time.time() - t0

    if not encoded:
        return {}

    N = len(encoded["valid_entries"])
    ms_per = t_total / N * 1000 if N else 0
    print(f"  Encoding: {ms_per:.1f} ms/pocket ({N} valid)")

    embs = encoded["spectral"]
    norms = np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    normed = embs / norms

    t0 = time.time()
    sim = normed @ normed.T
    t_sim = time.time() - t0
    n_pairs = N*(N-1)//2
    us_per = t_sim / n_pairs * 1e6 if n_pairs else 0

    results = {
        "encoding": {"n_pockets": N, "total_sec": t_total, "ms_per_pocket": ms_per},
        "pairwise": {"n_pockets": N, "n_pairs": n_pairs, "total_sec": t_sim, "us_per_pair": us_per},
    }
    print(f"  Pairwise: {N}x{N} ({n_pairs:,} pairs) in {t_sim:.3f}s ({us_per:.2f} us/pair)")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "timing_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["cross_family", "kinase", "benchmark", "all"], default="all")
    ap.add_argument("--manifest")
    ap.add_argument("--manifest_dir", default="data")
    ap.add_argument("--pdb_dir", default="data/pdbs")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--permutations", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.experiment in ("cross_family", "all"):
        m = args.manifest or os.path.join(args.manifest_dir, "cross_family_manifest.csv")
        if os.path.exists(m):
            run_cross_family_experiment(m, args.pdb_dir, args.out_dir, args.permutations)

    if args.experiment in ("kinase", "all"):
        m = args.manifest if args.experiment != "all" else os.path.join(args.manifest_dir, "kinase_manifest.csv")
        if os.path.exists(m):
            run_kinase_experiment(m, args.pdb_dir, args.out_dir, args.permutations)

    if args.experiment in ("benchmark", "all"):
        m = args.manifest or os.path.join(args.manifest_dir, "cross_family_manifest.csv")
        if os.path.exists(m):
            run_timing_benchmark(m, args.pdb_dir, args.out_dir)


if __name__ == "__main__":
    main()
