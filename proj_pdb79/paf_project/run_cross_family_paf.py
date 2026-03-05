#!/usr/bin/env python3
"""
run_cross_family_paf.py

Cross-family generalization experiment for PAF v1.

Tests whether PAF embeddings separate protein families (kinases, serine
proteases, metalloproteases, nuclear receptors) and whether subfamily
structure is preserved within each family.

Evaluation levels:
  Level 1 – FAMILY separation: Can PAF tell kinases from proteases?
  Level 2 – SUBFAMILY separation (within each family): Can PAF tell
            MMP1 from MMP13, or CDK2 from ABL?
  Level 3 – Cross-family confusion matrix: Which families are most/least
            discriminated?

Methods compared:
  - PAF_v1:     Wave-domain FFT embedding
  - FP-A_mean:  Mean aggregation (no spatial info)
  - FP-B_radial: Radial histogram (spatial, no wave)

Usage:
  python run_cross_family_paf.py \
    --csv data/cross_family/cross_family_list.csv \
    --out results/cross_family/ \
    --radius 10 --gamma_fm 0.15 --sigma_t 0.04
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

# ---- Import PAF core (adjust path as needed) ----
sys.path.insert(0, os.path.dirname(__file__))
try:
    from paf_core_v1 import PAFParams, extract_pocket, pocket_to_embedding
except ImportError:
    print("[ERROR] Cannot import paf_core_v1. Make sure paf_core_v1.py is in the same directory.")
    sys.exit(1)


# ============================================================
# Feature extraction (same as compare_paf_vs_baselines_v1.py)
# ============================================================

def residue_feature30(r) -> np.ndarray:
    pc = r.physchem
    v = [
        float(pc.get("charge", 0.0)),
        float(pc.get("hyd", 0.0)),
        float(pc.get("hb", 0.0)),
        float(pc.get("aro", 0.0)),
        float(pc.get("vol", 0.0)),
        float(r.flex if getattr(r, "flex", None) is not None else 0.0),
        float(r.contact if getattr(r, "contact", None) is not None else 0.0),
        float(r.ss_onehot[0] if getattr(r, "ss_onehot", None) is not None else 0.0),
        float(r.ss_onehot[1] if getattr(r, "ss_onehot", None) is not None else 0.0),
        float(r.ss_onehot[2] if getattr(r, "ss_onehot", None) is not None else 1.0),
    ]
    bl = np.asarray(getattr(r, "blosum", np.zeros((20,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if bl.shape[0] != 20:
        bl = np.zeros((20,), dtype=np.float32)
    out = np.concatenate([np.asarray(v, dtype=np.float32), bl], axis=0)
    return out


def fp_a_mean(pocket) -> np.ndarray:
    X = np.stack([residue_feature30(r) for r in pocket.residues], axis=0)
    return X.mean(axis=0).astype(np.float32).reshape(-1)


def fp_b_radial_hist(pocket, radius_A: float, n_shells: int = 16) -> np.ndarray:
    edges = np.linspace(0.0, radius_A, n_shells + 1)
    shell_feats = []
    for si in range(n_shells):
        lo, hi = edges[si], edges[si + 1]
        rs = [r for r in pocket.residues if (float(r.radial1_A) >= lo and float(r.radial1_A) < hi)]
        if not rs:
            shell_feats.append(np.zeros((30,), dtype=np.float32))
        else:
            X = np.stack([residue_feature30(r) for r in rs], axis=0)
            shell_feats.append(X.mean(axis=0).astype(np.float32))
    return np.concatenate(shell_feats, axis=0).astype(np.float32).reshape(-1)


# ============================================================
# Metrics (from compare script, extended)
# ============================================================

def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    u = u.reshape(-1).astype(np.float32)
    v = v.reshape(-1).astype(np.float32)
    du, dv = float(np.dot(u, u)), float(np.dot(v, v))
    if du < eps or dv < eps:
        return 0.0
    return float(np.dot(u, v) / (math.sqrt(du) * math.sqrt(dv)))


def pairwise_similarity(keys: List[str], emb: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(keys)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine(emb[keys[i]], emb[keys[j]])
            S[i, j] = s
            S[j, i] = s
    return S


def loo_knn_accuracy(keys: List[str], labels: Dict[str, str],
                     S: np.ndarray, k: int = 1) -> float:
    n = len(keys)
    if n == 0:
        return 0.0
    correct = 0
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -1e9
        nn = np.argsort(-sims)[:k]
        if k == 1:
            pred = labels[keys[int(nn[0])]]
        else:
            votes = {}
            for idx in nn:
                lab = labels[keys[int(idx)]]
                votes[lab] = votes.get(lab, 0) + 1
            pred = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
        if pred == labels[keys[i]]:
            correct += 1
    return correct / n


def within_between_stats(keys: List[str], labels: Dict[str, str], S: np.ndarray):
    within, between = [], []
    n = len(keys)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[keys[i]] == labels[keys[j]]:
                within.append(float(S[i, j]))
            else:
                between.append(float(S[i, j]))
    within = np.asarray(within, dtype=np.float32)
    between = np.asarray(between, dtype=np.float32)
    if len(within) < 2 or len(between) < 2:
        return 0.0, 1.0, 0.0, within, between
    mw, mb = float(within.mean()), float(between.mean())
    sw, sb = float(within.std(ddof=1)), float(between.std(ddof=1))
    nw, nb = len(within), len(between)
    pooled = math.sqrt(((nw-1)*sw*sw + (nb-1)*sb*sb) / (nw+nb-2)) if (nw+nb-2) > 0 else 0.0
    d = (mw - mb) / pooled if pooled > 1e-12 else 0.0
    denom = math.sqrt((sw*sw)/nw + (sb*sb)/nb) if (nw > 0 and nb > 0) else 1e9
    t = (mw - mb) / denom if denom > 1e-12 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return d, p, mw - mb, within, between


def confusion_matrix_knn(keys: List[str], labels: Dict[str, str],
                         S: np.ndarray, k: int = 1) -> pd.DataFrame:
    """Build family-level confusion matrix from LOO k-NN."""
    classes = sorted(set(labels[k] for k in keys))
    n = len(keys)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    c2i = {c: i for i, c in enumerate(classes)}

    for i in range(n):
        sims = S[i].copy()
        sims[i] = -1e9
        nn = np.argsort(-sims)[:k]
        if k == 1:
            pred = labels[keys[int(nn[0])]]
        else:
            votes = {}
            for idx in nn:
                lab = labels[keys[int(idx)]]
                votes[lab] = votes.get(lab, 0) + 1
            pred = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
        true = labels[keys[i]]
        cm[c2i[true], c2i[pred]] += 1

    return pd.DataFrame(cm, index=classes, columns=classes)


# ============================================================
# Visualization
# ============================================================

def plot_pca_families(emb: Dict[str, np.ndarray],
                     family_labels: Dict[str, str],
                     subfamily_labels: Dict[str, str],
                     out_png: str, title: str):
    """PCA colored by family, with subfamily markers."""
    keys = sorted(emb.keys())
    X = np.stack([emb[k].reshape(-1) for k in keys], axis=0).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S_vals, _ = np.linalg.svd(X, full_matrices=False)
    pc = U[:, :2] * S_vals[:2]

    # Variance explained
    var_exp = S_vals[:2]**2 / (S_vals**2).sum() * 100

    families = [family_labels[k] for k in keys]
    uniq_fam = sorted(set(families))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(uniq_fam), 3)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for fi, fam in enumerate(uniq_fam):
        idx = [i for i, f in enumerate(families) if f == fam]
        ax.scatter(pc[idx, 0], pc[idx, 1],
                   c=[colors[fi]], label=fam, s=60, alpha=0.7,
                   edgecolors="k", linewidths=0.5)
        # Label each point with subfamily
        for ii in idx:
            sub = subfamily_labels[keys[ii]]
            ax.annotate(sub, (pc[ii, 0], pc[ii, 1]),
                       fontsize=5, alpha=0.6,
                       xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm_df: pd.DataFrame, out_png: str, title: str):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(cm_df.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            v = cm_df.values[i, j]
            color = "white" if v > cm_df.values.max() / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_family_bar_comparison(results: List[dict], out_png: str):
    """Bar chart comparing methods across metrics."""
    methods = [r["method"] for r in results]
    x = np.arange(len(methods))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, [r["family_1nn"] for r in results], width=w, label="Family 1-NN")
    ax.bar(x,     [r["family_3nn"] for r in results], width=w, label="Family 3-NN")
    ax.bar(x + w, [r["family_cohen_d"] for r in results], width=w, label="Family Cohen's d")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title("Cross-family PAF generalization: method comparison")
    ax.legend()
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_within_between_by_family(emb: Dict[str, np.ndarray],
                                   family_labels: Dict[str, str],
                                   subfamily_labels: Dict[str, str],
                                   out_png: str, title: str):
    """Boxplot: within-subfamily vs between-subfamily similarity, per family."""
    families = sorted(set(family_labels.values()))
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        fam_keys = sorted([k for k in emb if family_labels[k] == fam])
        if len(fam_keys) < 3:
            ax.set_title(f"{fam}\n(too few)")
            continue

        within, between = [], []
        for i in range(len(fam_keys)):
            for j in range(i + 1, len(fam_keys)):
                s = _cosine(emb[fam_keys[i]], emb[fam_keys[j]])
                if subfamily_labels[fam_keys[i]] == subfamily_labels[fam_keys[j]]:
                    within.append(s)
                else:
                    between.append(s)

        data = []
        labels_bp = []
        if within:
            data.append(within)
            labels_bp.append(f"within\n(n={len(within)})")
        if between:
            data.append(between)
            labels_bp.append(f"between\n(n={len(between)})")

        if data:
            ax.boxplot(data, tick_labels=labels_bp, showfliers=False)
        ax.set_title(f"{fam}\n({len(fam_keys)} structs)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================
# Main experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="cross_family_list.csv")
    ap.add_argument("--out", required=True, help="output directory")

    # PAF params (use best from ablation)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--gamma_fm", type=float, default=0.15)
    ap.add_argument("--sigma_t", type=float, default=0.04)
    ap.add_argument("--a_hyd", type=float, default=1.0)
    ap.add_argument("--a_charge", type=float, default=1.0)
    ap.add_argument("--a_vol", type=float, default=0.5)
    ap.add_argument("--n_shells", type=int, default=16)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params = PAFParams(
        pocket_radius_A=args.radius,
        gamma_fm=args.gamma_fm,
        sigma_t_s=args.sigma_t,
        a_hyd=args.a_hyd,
        a_charge=args.a_charge,
        a_vol=args.a_vol,
    )

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} entries from {args.csv}")
    print(f"Families: {dict(Counter(df['family']))}")
    print()

    # ------- Extract pockets + compute embeddings -------
    family_labels: Dict[str, str] = {}
    subfamily_labels: Dict[str, str] = {}
    emb_paf: Dict[str, np.ndarray] = {}
    emb_fpa: Dict[str, np.ndarray] = {}
    emb_fpb: Dict[str, np.ndarray] = {}
    failures = []

    for _, r in df.iterrows():
        pdb_path = str(r["pdb_path"])
        chain_id = str(r["chain_id"])
        ligand_resname = str(r["ligand_resname"]) if not pd.isna(r.get("ligand_resname", "")) else ""
        family = str(r["family"])
        subfamily = str(r["subfamily"])
        pdb_id = str(r.get("pdb_id", os.path.basename(pdb_path).replace(".pdb", "")))

        key = f"{pdb_id}|{chain_id}"

        try:
            pocket = extract_pocket(
                pdb_path=pdb_path,
                chain_id=chain_id,
                ligand_resname=ligand_resname,
                params=params,
            )

            # PAF embedding
            spec, _ = pocket_to_embedding(pocket, params)
            emb_paf[key] = np.asarray(spec, dtype=np.float32).reshape(-1)

            # Baselines
            emb_fpa[key] = fp_a_mean(pocket).reshape(-1)
            emb_fpb[key] = fp_b_radial_hist(pocket, radius_A=args.radius,
                                              n_shells=args.n_shells).reshape(-1)

            family_labels[key] = family
            subfamily_labels[key] = subfamily

            print(f"  [OK] {key:20s} fam={family:20s} sub={subfamily:20s} nres={len(pocket.residues)}")

        except Exception as e:
            failures.append({"key": key, "pdb_path": pdb_path, "error": str(e)})
            print(f"  [FAIL] {key}: {e}")

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(args.out, "extraction_failures.csv"), index=False)
        print(f"\n{len(failures)} extraction failures saved.")

    keys = sorted(emb_paf.keys())
    print(f"\nSuccessfully extracted: {len(keys)} pockets")
    print(f"Family distribution: {dict(Counter(family_labels[k] for k in keys))}")

    if len(keys) < 10:
        raise SystemExit("[FATAL] Too few successful extractions.")

    # ------- Evaluate all methods -------
    methods = {
        "PAF_v1": emb_paf,
        "FP-A_mean": emb_fpa,
        "FP-B_radial": emb_fpb,
    }

    all_results = []

    for method_name, emb in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        S = pairwise_similarity(keys, emb)

        # --- Level 1: Family-level classification ---
        fam_1nn = loo_knn_accuracy(keys, family_labels, S, k=1)
        fam_3nn = loo_knn_accuracy(keys, family_labels, S, k=3)
        fam_d, fam_p, fam_gap, fam_w, fam_b = within_between_stats(keys, family_labels, S)
        cm_fam = confusion_matrix_knn(keys, family_labels, S, k=1)

        print(f"  FAMILY level:")
        print(f"    1-NN accuracy: {fam_1nn:.3f}")
        print(f"    3-NN accuracy: {fam_3nn:.3f}")
        print(f"    Cohen's d:     {fam_d:.3f}")
        print(f"    p-value:       {fam_p:.2e}")
        print(f"    Gap:           {fam_gap:.4f}")
        print(f"    Confusion matrix:\n{cm_fam}\n")

        # --- Level 2: Subfamily-level (within each family) ---
        sub_results = {}
        for fam in sorted(set(family_labels[k] for k in keys)):
            fam_keys = sorted([k for k in keys if family_labels[k] == fam])
            if len(fam_keys) < 4:
                continue
            sub_labs = {k: subfamily_labels[k] for k in fam_keys}
            S_fam = pairwise_similarity(fam_keys, emb)
            sub_1nn = loo_knn_accuracy(fam_keys, sub_labs, S_fam, k=1)
            sub_d, sub_p, sub_gap, _, _ = within_between_stats(fam_keys, sub_labs, S_fam)

            n_subs = len(set(sub_labs.values()))
            random_baseline = 1.0 / n_subs if n_subs > 0 else 0.0

            sub_results[fam] = {
                "n": len(fam_keys),
                "n_subfamilies": n_subs,
                "random_baseline": random_baseline,
                "1nn": sub_1nn,
                "cohen_d": sub_d,
                "p_value": sub_p,
            }
            print(f"  SUBFAMILY level ({fam}, n={len(fam_keys)}, {n_subs} subs):")
            print(f"    1-NN: {sub_1nn:.3f}  (random={random_baseline:.3f})")
            print(f"    Cohen's d: {sub_d:.3f}  p={sub_p:.2e}")

        result = {
            "method": method_name,
            "family_1nn": fam_1nn,
            "family_3nn": fam_3nn,
            "family_cohen_d": fam_d,
            "family_p_value": fam_p,
            "family_gap": fam_gap,
            "subfamily_results": sub_results,
            "confusion_matrix": cm_fam.to_dict(),
        }
        all_results.append(result)

        # --- Plots for this method ---
        plot_pca_families(
            emb, family_labels, subfamily_labels,
            os.path.join(args.out, f"fig_pca_{method_name}.png"),
            f"{method_name}: PCA by family"
        )
        plot_confusion_matrix(
            cm_fam,
            os.path.join(args.out, f"fig_confusion_{method_name}.png"),
            f"{method_name}: Family confusion matrix (1-NN)"
        )
        plot_within_between_by_family(
            emb, family_labels, subfamily_labels,
            os.path.join(args.out, f"fig_wb_by_family_{method_name}.png"),
            f"{method_name}: within vs between subfamily similarity"
        )

    # ------- Comparative plot -------
    plot_family_bar_comparison(
        all_results,
        os.path.join(args.out, "fig_method_comparison.png")
    )

    # ------- Save results -------
    # Flatten for CSV
    csv_rows = []
    for r in all_results:
        row = {
            "method": r["method"],
            "family_1nn": r["family_1nn"],
            "family_3nn": r["family_3nn"],
            "family_cohen_d": r["family_cohen_d"],
            "family_p_value": r["family_p_value"],
            "family_gap": r["family_gap"],
        }
        for fam, sr in r["subfamily_results"].items():
            row[f"sub_{fam}_1nn"] = sr["1nn"]
            row[f"sub_{fam}_d"] = sr["cohen_d"]
            row[f"sub_{fam}_p"] = sr["p_value"]
        csv_rows.append(row)

    results_df = pd.DataFrame(csv_rows)
    results_df.to_csv(os.path.join(args.out, "cross_family_results.csv"), index=False)

    # Full JSON
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(args.out, "cross_family_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    # Save embeddings
    for method_name, emb in methods.items():
        np.savez(
            os.path.join(args.out, f"emb_{method_name}.npz"),
            **{k: v for k, v in emb.items()}
        )

    # ------- Print summary -------
    print(f"\n{'='*70}")
    print("CROSS-FAMILY RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':20s} {'Fam 1-NN':>10s} {'Fam 3-NN':>10s} {'Cohen d':>10s} {'p-value':>12s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:20s} {r['family_1nn']:10.3f} {r['family_3nn']:10.3f} "
              f"{r['family_cohen_d']:10.3f} {r['family_p_value']:12.2e}")

    print(f"\nRandom baseline (4 families): {1/4:.3f}")
    fam_counts = Counter(family_labels[k] for k in keys)
    majority = max(fam_counts.values()) / len(keys)
    print(f"Majority baseline: {majority:.3f}")

    print(f"\nAll results saved to {args.out}")
    print(f"Key outputs:")
    print(f"  cross_family_results.csv   – summary table")
    print(f"  cross_family_results.json  – full results")
    print(f"  fig_pca_*.png              – PCA by family")
    print(f"  fig_confusion_*.png        – confusion matrices")
    print(f"  fig_wb_by_family_*.png     – within/between per family")
    print(f"  fig_method_comparison.png  – method comparison bars")


if __name__ == "__main__":
    main()
