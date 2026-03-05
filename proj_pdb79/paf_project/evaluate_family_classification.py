#!/usr/bin/env python3
"""
evaluate_family_classification.py
===================================
Proper evaluation of PAF vs baselines using kinase family labels.

The Kabsch RMSD ground truth was broken (matching residues by radial
order is not a valid structural alignment). Instead, we evaluate:

  1. Silhouette score: How well does each representation separate families?
  2. Leave-one-out 1-NN classification: Can we predict family from embedding?
  3. Leave-one-out 3-NN classification: More robust neighbor voting
  4. Within-family vs between-family similarity: Effect size (Cohen's d)

These metrics use the family labels (TK, CMGC, AGC, CAMK, STE, TKL, CK1, Other)
as ground truth — no structural alignment needed.

Usage:
  python evaluate_family_classification.py \
      --pdb_list data/kinase_pdbs/kinase_list.csv \
      --out results/kinase_v1/

  Or run standalone on already-extracted pockets:
  python evaluate_family_classification.py --from_npy results/kinase_v0/ --out results/kinase_v1/
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ============================================================================
# Import shared code
# ============================================================================

from head_to_head import (
    Pocket, ResidueRecord, PAFParams, CHANNELS,
    aa_features, paf_embedding,
    fp_composition, fp_radial_hist, fp_radial_hist_log, fp_sorted_concat,
    cosine_sim, cosine_similarity_matrix,
)


# ============================================================================
# Metric 1: Silhouette score (using cosine distance)
# ============================================================================

def silhouette_score(embeddings: Dict[str, np.ndarray], labels: Dict[str, str]) -> float:
    """
    Silhouette score measures cluster quality.
    +1 = perfect separation, 0 = overlapping, -1 = wrong assignment.
    Uses cosine distance.
    """
    keys = list(embeddings.keys())
    n = len(keys)
    lab = [labels[k] for k in keys]
    unique_labels = sorted(set(lab))

    if len(unique_labels) < 2:
        return 0.0

    # Pairwise cosine distance matrix
    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0).astype(np.float64)
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    D = 1.0 - X_norm @ X_norm.T  # cosine distance
    np.fill_diagonal(D, 0.0)

    silhouettes = []
    for i in range(n):
        my_label = lab[i]
        same_mask = np.array([lab[j] == my_label for j in range(n)])
        same_mask[i] = False  # exclude self

        if same_mask.sum() == 0:
            continue  # singleton cluster

        a_i = D[i, same_mask].mean()  # mean intra-cluster distance

        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == my_label:
                continue
            other_mask = np.array([lab[j] == other_label for j in range(n)])
            if other_mask.sum() == 0:
                continue
            b_other = D[i, other_mask].mean()
            b_i = min(b_i, b_other)

        if b_i == float("inf"):
            continue

        s_i = (b_i - a_i) / max(a_i, b_i, 1e-8)
        silhouettes.append(s_i)

    return float(np.mean(silhouettes)) if silhouettes else 0.0


# ============================================================================
# Metric 2 & 3: Leave-one-out k-NN classification
# ============================================================================

def loo_knn_accuracy(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    k: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """
    Leave-one-out k-nearest-neighbor classification.
    Returns overall accuracy and per-family accuracy.
    """
    keys = list(embeddings.keys())
    n = len(keys)
    lab = [labels[k_] for k_ in keys]

    # Cosine similarity matrix
    X = np.stack([embeddings[k_].reshape(-1) for k_ in keys], axis=0).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    S = X_norm @ X_norm.T
    np.fill_diagonal(S, -np.inf)  # exclude self

    correct = 0
    family_correct = Counter()
    family_total = Counter()

    for i in range(n):
        true_label = lab[i]
        family_total[true_label] += 1

        # k nearest neighbors (highest cosine similarity)
        neighbor_idx = np.argsort(S[i, :])[-k:]
        neighbor_labels = [lab[j] for j in neighbor_idx]
        predicted = Counter(neighbor_labels).most_common(1)[0][0]

        if predicted == true_label:
            correct += 1
            family_correct[true_label] += 1

    overall = correct / n
    per_family = {
        fam: family_correct[fam] / family_total[fam]
        for fam in sorted(family_total.keys())
    }
    return overall, per_family


# ============================================================================
# Metric 4: Within vs between family similarity (Cohen's d)
# ============================================================================

def within_between_effect_size(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
) -> Tuple[float, float, float, float]:
    """
    Compare within-family vs between-family cosine similarity.
    Returns: mean_within, mean_between, cohen_d, p_value_ttest
    """
    from scipy.stats import ttest_ind

    keys = list(embeddings.keys())
    n = len(keys)
    lab = [labels[k] for k in keys]

    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    S = X_norm @ X_norm.T

    within = []
    between = []
    for i in range(n):
        for j in range(i + 1, n):
            if lab[i] == lab[j]:
                within.append(S[i, j])
            else:
                between.append(S[i, j])

    within = np.array(within)
    between = np.array(between)

    mean_w = float(within.mean()) if len(within) > 0 else 0.0
    mean_b = float(between.mean()) if len(between) > 0 else 0.0

    # Cohen's d
    pooled_std = float(np.sqrt(
        (within.var() * len(within) + between.var() * len(between))
        / (len(within) + len(between))
    )) + 1e-8
    d = (mean_w - mean_b) / pooled_std

    # t-test
    if len(within) > 1 and len(between) > 1:
        _, pval = ttest_ind(within, between, equal_var=False)
        pval = float(pval)
    else:
        pval = 1.0

    return mean_w, mean_b, d, pval


# ============================================================================
# Run all evaluations
# ============================================================================

def evaluate_all(
    pockets: Dict[str, Pocket],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    params = PAFParams()
    keys = list(pockets.keys())
    labels = {k: pockets[k].family for k in keys}

    # Family distribution
    fam_counts = Counter(labels.values())
    print(f"\nFamily distribution ({len(keys)} pockets):")
    for fam in sorted(fam_counts.keys()):
        print(f"  {fam:8s}: {fam_counts[fam]}")

    # Compute embeddings
    print("\nComputing embeddings...")
    all_methods = {}

    emb = {}
    for k in keys:
        emb[k] = paf_embedding(pockets[k], params)
    all_methods["PAF (wave+FFT)"] = emb

    emb = {}
    for k in keys:
        emb[k] = fp_composition(pockets[k])
    all_methods["FP-A (composition)"] = emb

    emb = {}
    for k in keys:
        emb[k] = fp_radial_hist(pockets[k], R=params.pocket_radius_A)
    all_methods["FP-B (radial hist)"] = emb

    emb = {}
    for k in keys:
        emb[k] = fp_radial_hist_log(pockets[k], R=params.pocket_radius_A)
    all_methods["FP-B-log (radial log)"] = emb

    emb = {}
    for k in keys:
        emb[k] = fp_sorted_concat(pockets[k])
    all_methods["FP-C (sorted concat)"] = emb

    # Evaluate each method
    print(f"\n{'='*80}")
    print(f"{'Method':28s} {'Silhouette':>10s} {'1-NN':>8s} {'3-NN':>8s} {'Cohen d':>9s} {'W>B p':>10s}")
    print(f"{'='*80}")

    results = []
    for name, embs in all_methods.items():
        sil = silhouette_score(embs, labels)
        acc1, per_fam1 = loo_knn_accuracy(embs, labels, k=1)
        acc3, per_fam3 = loo_knn_accuracy(embs, labels, k=3)
        mean_w, mean_b, cohen_d, pval = within_between_effect_size(embs, labels)

        print(f"  {name:26s} {sil:>10.4f} {acc1:>8.1%} {acc3:>8.1%} {cohen_d:>9.3f} {pval:>10.2e}")

        results.append({
            "method": name,
            "silhouette": sil,
            "loo_1nn_accuracy": acc1,
            "loo_3nn_accuracy": acc3,
            "mean_within_sim": mean_w,
            "mean_between_sim": mean_b,
            "cohen_d": cohen_d,
            "within_vs_between_pval": pval,
        })

    print(f"{'='*80}")

    # Random baseline
    rng = np.random.default_rng(42)
    random_accs = []
    for _ in range(1000):
        shuffled = list(labels.values())
        rng.shuffle(shuffled)
        shuffled_labels = dict(zip(keys, shuffled))
        dummy_embs = {k: all_methods["PAF (wave+FFT)"][k] for k in keys}
        a, _ = loo_knn_accuracy(dummy_embs, shuffled_labels, k=1)
        random_accs.append(a)
    chance = float(np.mean(random_accs))
    print(f"\n  Random baseline (shuffled labels): {chance:.1%}")
    print(f"  Majority class baseline: {max(fam_counts.values())/len(keys):.1%}")

    # Save
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "family_classification_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Results table: {csv_path}")

    # Per-family breakdown for best method
    best_idx = df["loo_1nn_accuracy"].idxmax()
    best_name = df.loc[best_idx, "method"]
    _, per_fam = loo_knn_accuracy(all_methods[best_name], labels, k=1)
    print(f"\n  Per-family 1-NN accuracy ({best_name}):")
    for fam in sorted(per_fam.keys()):
        n_fam = fam_counts[fam]
        acc = per_fam[fam]
        print(f"    {fam:8s} (n={n_fam:2d}): {acc:.1%}")

    # =========================================
    # FIGURES
    # =========================================

    # Figure 1: Bar chart of all metrics
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    methods = [r["method"] for r in results]
    x = np.arange(len(methods))
    colors = ["#2196F3" if "PAF" in m else "#9E9E9E" if "composition" in m.lower() else "#FF9800" for m in methods]

    for ax, metric, title in zip(axes, 
        ["silhouette", "loo_1nn_accuracy", "loo_3nn_accuracy", "cohen_d"],
        ["Silhouette Score", "1-NN Accuracy (LOO)", "3-NN Accuracy (LOO)", "Cohen's d (W vs B)"]):
        vals = [r[metric] for r in results]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("(")[0].strip() for m in methods], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if "accuracy" in metric.lower():
            ax.axhline(chance, color="red", linewidth=1, linestyle="--", label=f"chance ({chance:.0%})")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.0)

    plt.suptitle("PAF vs Baselines: Kinase Family Classification", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_family_metrics.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {fig_path}")

    # Figure 2: Confusion-style heatmap for best method (1-NN)
    plot_knn_confusion(all_methods[best_name], labels, best_name, out_dir)

    # Figure 3: Within vs between similarity distributions
    plot_within_between(all_methods, labels, out_dir)

    # Figure 4: PCA with proper family coloring (cleaner version)
    for name in ["PAF (wave+FFT)", "FP-B (radial hist)"]:
        plot_pca_clean(all_methods[name], labels, name, out_dir)

    return results


def plot_knn_confusion(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    method_name: str,
    out_dir: str,
):
    """1-NN confusion matrix."""
    keys = list(embeddings.keys())
    n = len(keys)
    lab = [labels[k] for k in keys]
    unique_fams = sorted(set(lab))
    fam_to_idx = {f: i for i, f in enumerate(unique_fams)}
    nf = len(unique_fams)

    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0).astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms
    S = X_norm @ X_norm.T
    np.fill_diagonal(S, -np.inf)

    cm = np.zeros((nf, nf), dtype=int)
    for i in range(n):
        true_idx = fam_to_idx[lab[i]]
        nn = np.argmax(S[i, :])
        pred_idx = fam_to_idx[lab[nn]]
        cm[true_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(nf))
    ax.set_yticks(range(nf))
    ax.set_xticklabels(unique_fams, rotation=45, ha="right")
    ax.set_yticklabels(unique_fams)
    ax.set_xlabel("Predicted family", fontsize=11)
    ax.set_ylabel("True family", fontsize=11)
    ax.set_title(f"1-NN Confusion Matrix — {method_name}", fontsize=12)

    for i in range(nf):
        for j in range(nf):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_confusion_1nn.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {path}")


def plot_within_between(
    all_methods: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, str],
    out_dir: str,
):
    """Violin/box plot of within vs between family similarities."""
    fig, axes = plt.subplots(1, len(all_methods), figsize=(4 * len(all_methods), 5), squeeze=False)

    for idx, (name, embs) in enumerate(all_methods.items()):
        ax = axes[0][idx]
        keys = list(embs.keys())
        n = len(keys)
        lab = [labels[k] for k in keys]

        X = np.stack([embs[k].reshape(-1) for k in keys], axis=0).astype(np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms
        S = X_norm @ X_norm.T

        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if lab[i] == lab[j]:
                    within.append(S[i, j])
                else:
                    between.append(S[i, j])

        data = [between, within]
        bp = ax.boxplot(data, labels=["Between", "Within"], patch_artist=True,
                       widths=0.6, showfliers=False)
        bp["boxes"][0].set_facecolor("#FFCCBC")
        bp["boxes"][1].set_facecolor("#BBDEFB")

        short_name = name.split("(")[0].strip()
        ax.set_title(short_name, fontsize=10)
        ax.set_ylabel("Cosine similarity" if idx == 0 else "", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Within-Family vs Between-Family Similarity", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_within_between.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {path}")


def plot_pca_clean(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, str],
    method_name: str,
    out_dir: str,
):
    """Clean PCA with larger labels and better colors."""
    keys = list(embeddings.keys())
    X = np.stack([embeddings[k].reshape(-1) for k in keys], axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S_vals, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S_vals[:2]

    # Variance explained
    total_var = (S_vals ** 2).sum()
    var1 = S_vals[0] ** 2 / total_var * 100
    var2 = S_vals[1] ** 2 / total_var * 100

    fam_list = [labels[k] for k in keys]
    unique_fams = sorted(set(fam_list))

    # Better color palette
    palette = {
        "TK": "#E53935", "CMGC": "#7B1FA2", "AGC": "#1E88E5",
        "CAMK": "#FB8C00", "STE": "#78909C", "TKL": "#00ACC1",
        "CK1": "#43A047", "Other": "#795548",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for fam in unique_fams:
        mask = [i for i, f in enumerate(fam_list) if f == fam]
        ax.scatter(Z[mask, 0], Z[mask, 1],
                  c=palette.get(fam, "#999999"), s=60,
                  edgecolors="black", linewidth=0.5, label=fam, zorder=3)

    # Labels
    for i, k in enumerate(keys):
        short = k.split("|")[0]
        ax.annotate(short, (Z[i, 0], Z[i, 1]),
                   fontsize=6, alpha=0.8,
                   xytext=(4, 4), textcoords="offset points")

    ax.legend(fontsize=9, title="Kinase Family", loc="best",
             framealpha=0.9, edgecolor="gray")
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)

    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    ax.set_title(f"PCA of {method_name} — Kinase Family Clustering", fontsize=12)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(out_dir, f"fig_pca_clean_{safe_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Wrote: {path}")


# ============================================================================
# Main: load pockets from PDB (reuses run_real_kinases extraction)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_list", required=True,
                       help="CSV manifest (same as run_real_kinases.py)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--radius", type=float, default=10.0)
    args = parser.parse_args()

    # Import pocket extraction from the bridge script
    from run_real_kinases import extract_pocket_from_pdb

    df = pd.read_csv(args.pdb_list)
    pockets = {}
    failed = 0

    print(f"Loading {len(df)} kinase pockets...")
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
            pocket = extract_pocket_from_pdb(
                pdb_path=pdb_path, chain_id=chain_id,
                ligand_resname=ligand, pocket_radius=args.radius,
                family=family, kinase_name=label,
            )
            pockets[pocket.pocket_id] = pocket
        except Exception as e:
            print(f"  FAIL {label}: {e}")
            failed += 1

    print(f"Loaded {len(pockets)} pockets ({failed} failed)")

    if len(pockets) < 10:
        print("ERROR: Too few pockets.")
        sys.exit(1)

    evaluate_all(pockets, args.out)
