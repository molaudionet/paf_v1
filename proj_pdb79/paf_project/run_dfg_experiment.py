#!/usr/bin/env python3
"""
run_dfg_experiment.py

Test whether PAF embeddings distinguish DFG-in (active) from DFG-out
(inactive) kinase conformations.

Evaluation:
  Level 1 – Binary classification: DFG-in vs DFG-out (LOO k-NN)
  Level 2 – Paired analysis: For each kinase with both states, measure
            embedding distance between its DFG-in and DFG-out structures
            vs distance to other kinases in the same state
  Level 3 – Can we separate state *after controlling for kinase identity*?
            (Within-kinase vs between-state distances)

The paired design is critical: if PAF just groups by kinase identity
(ABL near ABL regardless of state), that's NOT functional discrimination.
We need DFG-in structures to cluster with other DFG-in structures more
than with their own kinase's DFG-out structure.

Usage:
  python run_dfg_experiment.py \
    --csv data/dfg/dfg_list.csv \
    --out results/dfg/ \
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

sys.path.insert(0, os.path.dirname(__file__))
try:
    from paf_core_v1 import PAFParams, extract_pocket, pocket_to_embedding
except ImportError:
    print("[ERROR] Cannot import paf_core_v1.")
    sys.exit(1)


# ============================================================
# Feature extraction (same as cross-family)
# ============================================================

def residue_feature30(r) -> np.ndarray:
    pc = r.physchem
    v = [
        float(pc.get("charge", 0.0)), float(pc.get("hyd", 0.0)),
        float(pc.get("hb", 0.0)), float(pc.get("aro", 0.0)),
        float(pc.get("vol", 0.0)),
        float(r.flex if getattr(r, "flex", None) is not None else 0.0),
        float(r.contact if getattr(r, "contact", None) is not None else 0.0),
        float(r.ss_onehot[0] if getattr(r, "ss_onehot", None) is not None else 0.0),
        float(r.ss_onehot[1] if getattr(r, "ss_onehot", None) is not None else 0.0),
        float(r.ss_onehot[2] if getattr(r, "ss_onehot", None) is not None else 1.0),
    ]
    bl = np.asarray(getattr(r, "blosum", np.zeros(20, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if bl.shape[0] != 20:
        bl = np.zeros(20, dtype=np.float32)
    return np.concatenate([np.asarray(v, dtype=np.float32), bl], axis=0)


def fp_a_mean(pocket) -> np.ndarray:
    X = np.stack([residue_feature30(r) for r in pocket.residues], axis=0)
    return X.mean(axis=0).astype(np.float32).reshape(-1)


def fp_b_radial_hist(pocket, radius_A: float, n_shells: int = 16) -> np.ndarray:
    edges = np.linspace(0.0, radius_A, n_shells + 1)
    shell_feats = []
    for si in range(n_shells):
        lo, hi = edges[si], edges[si + 1]
        rs = [r for r in pocket.residues if lo <= float(r.radial1_A) < hi]
        if not rs:
            shell_feats.append(np.zeros(30, dtype=np.float32))
        else:
            X = np.stack([residue_feature30(r) for r in rs], axis=0)
            shell_feats.append(X.mean(axis=0).astype(np.float32))
    return np.concatenate(shell_feats, axis=0).astype(np.float32).reshape(-1)


# ============================================================
# Metrics
# ============================================================

def _cosine(u, v, eps=1e-12):
    u, v = u.reshape(-1).astype(np.float32), v.reshape(-1).astype(np.float32)
    du, dv = float(np.dot(u, u)), float(np.dot(v, v))
    if du < eps or dv < eps:
        return 0.0
    return float(np.dot(u, v) / (math.sqrt(du) * math.sqrt(dv)))


def pairwise_similarity(keys, emb):
    n = len(keys)
    S = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine(emb[keys[i]], emb[keys[j]])
            S[i, j] = s
            S[j, i] = s
    return S


def loo_knn_accuracy(keys, labels, S, k=1):
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


def within_between_stats(keys, labels, S):
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
    denom = math.sqrt(sw*sw/nw + sb*sb/nb) if (nw > 0 and nb > 0) else 1e9
    t = (mw - mb) / denom if denom > 1e-12 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
    return d, p, mw - mb, within, between


# ============================================================
# Paired analysis (the key test)
# ============================================================

def paired_analysis(emb: Dict[str, np.ndarray],
                    dfg_labels: Dict[str, str],
                    kinase_labels: Dict[str, str]) -> dict:
    """
    For paired kinases (both DFG-in and DFG-out available):

    Compute:
    1. same_kinase_diff_state: cosine(ABL_in, ABL_out) for each paired kinase
    2. diff_kinase_same_state: cosine(ABL_in, SRC_in) for all same-state pairs
    3. diff_kinase_diff_state: cosine(ABL_in, SRC_out) for all cross pairs

    If PAF captures functional state:
      diff_kinase_same_state > same_kinase_diff_state
      (i.e., ABL-in is more like SRC-in than ABL-out)

    If PAF only captures identity:
      same_kinase_diff_state > diff_kinase_same_state
      (i.e., ABL-in is more like ABL-out than SRC-in)
    """
    keys = sorted(emb.keys())

    # Find paired kinases
    kin_states = defaultdict(dict)
    for k in keys:
        kinase = kinase_labels[k]
        state = dfg_labels[k]
        kin_states[kinase][state] = k

    paired_kinases = {kin: states for kin, states in kin_states.items()
                      if "DFG-in" in states and "DFG-out" in states}

    if len(paired_kinases) < 2:
        return {"error": "Need at least 2 paired kinases", "n_paired": len(paired_kinases)}

    # 1. Same kinase, different state
    same_kin_diff_state = []
    for kin, states in paired_kinases.items():
        k_in = states["DFG-in"]
        k_out = states["DFG-out"]
        sim = _cosine(emb[k_in], emb[k_out])
        same_kin_diff_state.append({"kinase": kin, "similarity": sim})

    # 2. Different kinase, same state
    diff_kin_same_state = []
    paired_keys_in = [states["DFG-in"] for states in paired_kinases.values()]
    paired_keys_out = [states["DFG-out"] for states in paired_kinases.values()]

    for i in range(len(paired_keys_in)):
        for j in range(i + 1, len(paired_keys_in)):
            sim = _cosine(emb[paired_keys_in[i]], emb[paired_keys_in[j]])
            diff_kin_same_state.append(sim)
            sim = _cosine(emb[paired_keys_out[i]], emb[paired_keys_out[j]])
            diff_kin_same_state.append(sim)

    # 3. Different kinase, different state
    diff_kin_diff_state = []
    for i in range(len(paired_keys_in)):
        for j in range(len(paired_keys_out)):
            kin_i = kinase_labels[paired_keys_in[i]]
            kin_j = kinase_labels[paired_keys_out[j]]
            if kin_i != kin_j:
                sim = _cosine(emb[paired_keys_in[i]], emb[paired_keys_out[j]])
                diff_kin_diff_state.append(sim)

    same_mean = float(np.mean([x["similarity"] for x in same_kin_diff_state]))
    diff_same_mean = float(np.mean(diff_kin_same_state))
    diff_diff_mean = float(np.mean(diff_kin_diff_state))

    # The key question: does state or identity dominate?
    identity_dominates = same_mean > diff_same_mean
    state_sensitive = diff_same_mean > diff_diff_mean

    return {
        "n_paired_kinases": len(paired_kinases),
        "paired_kinases": sorted(paired_kinases.keys()),
        "same_kinase_diff_state_mean": same_mean,
        "same_kinase_diff_state_pairs": same_kin_diff_state,
        "diff_kinase_same_state_mean": diff_same_mean,
        "diff_kinase_same_state_n": len(diff_kin_same_state),
        "diff_kinase_diff_state_mean": diff_diff_mean,
        "diff_kinase_diff_state_n": len(diff_kin_diff_state),
        "identity_dominates": identity_dominates,
        "state_sensitive": state_sensitive,
        "interpretation": (
            "STATE > IDENTITY: DFG state is the primary organizing principle"
            if not identity_dominates else
            "IDENTITY > STATE: Kinase identity dominates over DFG state"
        ) + (" | Same-state pairs more similar than cross-state pairs"
             if state_sensitive else
             " | No clear state sensitivity in cross-kinase comparisons"),
    }


# ============================================================
# Visualization
# ============================================================

def plot_pca_dfg(emb, dfg_labels, kinase_labels, out_png, title):
    """PCA colored by DFG state, labeled by kinase."""
    keys = sorted(emb.keys())
    X = np.stack([emb[k].reshape(-1) for k in keys], axis=0).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S_vals, _ = np.linalg.svd(X, full_matrices=False)
    pc = U[:, :2] * S_vals[:2]
    var_exp = S_vals[:2]**2 / (S_vals**2).sum() * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {"DFG-in": "tab:blue", "DFG-out": "tab:red"}
    markers = {"DFG-in": "o", "DFG-out": "s"}

    for state in ["DFG-in", "DFG-out"]:
        idx = [i for i, k in enumerate(keys) if dfg_labels[k] == state]
        ax.scatter(pc[idx, 0], pc[idx, 1],
                   c=colors[state], marker=markers[state],
                   label=state, s=80, alpha=0.7, edgecolors="k", linewidths=0.5)

    # Draw lines between paired kinases
    kin_states = defaultdict(dict)
    for i, k in enumerate(keys):
        kin_states[kinase_labels[k]][dfg_labels[k]] = i

    for kin, states in kin_states.items():
        if "DFG-in" in states and "DFG-out" in states:
            i_in, i_out = states["DFG-in"], states["DFG-out"]
            ax.plot([pc[i_in, 0], pc[i_out, 0]], [pc[i_in, 1], pc[i_out, 1]],
                    "k-", alpha=0.3, linewidth=0.8)

    # Label points
    for i, k in enumerate(keys):
        kin = kinase_labels[k]
        ax.annotate(kin, (pc[i, 0], pc[i, 1]),
                    fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
    ax.set_title(title)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_paired_distances(paired_result: dict, out_png: str, title: str):
    """Bar chart of paired analysis distances."""
    if "error" in paired_result:
        return

    categories = [
        "Same kinase\ndiff state",
        "Diff kinase\nsame state",
        "Diff kinase\ndiff state",
    ]
    values = [
        paired_result["same_kinase_diff_state_mean"],
        paired_result["diff_kinase_same_state_mean"],
        paired_result["diff_kinase_diff_state_mean"],
    ]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="k", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_per_kinase_pairs(paired_result: dict, out_png: str, title: str):
    """Show per-kinase pair: similarity between DFG-in and DFG-out."""
    if "error" in paired_result:
        return

    pairs = paired_result["same_kinase_diff_state_pairs"]
    kinases = [p["kinase"] for p in pairs]
    sims = [p["similarity"] for p in pairs]

    # Sort by similarity
    order = np.argsort(sims)
    kinases = [kinases[i] for i in order]
    sims = [sims[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, len(kinases) * 0.4)))
    y = range(len(kinases))
    ax.barh(y, sims, color="steelblue", edgecolor="k", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(kinases)
    ax.set_xlabel("Cosine similarity (DFG-in vs DFG-out)")
    ax.set_title(title)
    ax.axvline(x=np.mean(sims), color="red", linestyle="--", alpha=0.7,
               label=f"mean={np.mean(sims):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
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
    print(f"Loaded {len(df)} entries")
    print(f"DFG states: {dict(Counter(df['dfg_state']))}")
    print(f"Kinases: {dict(Counter(df['kinase']))}")
    print()

    # ------- Extract -------
    dfg_labels: Dict[str, str] = {}
    kinase_labels: Dict[str, str] = {}
    emb_paf: Dict[str, np.ndarray] = {}
    emb_fpa: Dict[str, np.ndarray] = {}
    emb_fpb: Dict[str, np.ndarray] = {}
    failures = []

    for _, r in df.iterrows():
        pdb_path = str(r["pdb_path"])
        chain_id = str(r["chain_id"])
        ligand_resname = str(r["ligand_resname"]) if not pd.isna(r.get("ligand_resname", "")) else ""
        dfg_state = str(r["dfg_state"])
        kinase = str(r["kinase"])
        pdb_id = str(r.get("pdb_id", ""))

        key = f"{pdb_id}|{chain_id}"

        try:
            pocket = extract_pocket(
                pdb_path=pdb_path, chain_id=chain_id,
                ligand_resname=ligand_resname, params=params,
            )

            spec, _ = pocket_to_embedding(pocket, params)
            emb_paf[key] = np.asarray(spec, dtype=np.float32).reshape(-1)
            emb_fpa[key] = fp_a_mean(pocket).reshape(-1)
            emb_fpb[key] = fp_b_radial_hist(pocket, radius_A=args.radius,
                                             n_shells=args.n_shells).reshape(-1)

            dfg_labels[key] = dfg_state
            kinase_labels[key] = kinase

            print(f"  [OK] {key:15s} kin={kinase:12s} state={dfg_state:8s} nres={len(pocket.residues)}")

        except Exception as e:
            failures.append({"key": key, "pdb_path": pdb_path, "error": str(e)})
            print(f"  [FAIL] {key}: {e}")

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(args.out, "extraction_failures.csv"), index=False)

    keys = sorted(emb_paf.keys())
    print(f"\nExtracted: {len(keys)} pockets")
    print(f"DFG distribution: {dict(Counter(dfg_labels[k] for k in keys))}")

    if len(keys) < 6:
        raise SystemExit("[FATAL] Too few extractions.")

    # ------- Evaluate -------
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

        # --- Level 1: Binary DFG-in vs DFG-out ---
        acc1 = loo_knn_accuracy(keys, dfg_labels, S, k=1)
        acc3 = loo_knn_accuracy(keys, dfg_labels, S, k=3)
        d, p, gap, w, b = within_between_stats(keys, dfg_labels, S)

        print(f"\n  DFG STATE classification:")
        print(f"    1-NN accuracy: {acc1:.3f}")
        print(f"    3-NN accuracy: {acc3:.3f}")
        print(f"    Cohen's d:     {d:.3f}")
        print(f"    p-value:       {p:.2e}")
        print(f"    Gap:           {gap:.4f}")

        # --- Level 2: Kinase identity classification ---
        acc1_kin = loo_knn_accuracy(keys, kinase_labels, S, k=1)
        d_kin, p_kin, gap_kin, _, _ = within_between_stats(keys, kinase_labels, S)

        print(f"\n  KINASE IDENTITY classification:")
        print(f"    1-NN accuracy: {acc1_kin:.3f}")
        print(f"    Cohen's d:     {d_kin:.3f}")
        print(f"    p-value:       {p_kin:.2e}")

        # --- Level 3: Paired analysis ---
        paired = paired_analysis(emb, dfg_labels, kinase_labels)

        print(f"\n  PAIRED ANALYSIS ({paired.get('n_paired_kinases', 0)} kinases):")
        if "error" not in paired:
            print(f"    Same kinase, diff state: {paired['same_kinase_diff_state_mean']:.3f}")
            print(f"    Diff kinase, same state: {paired['diff_kinase_same_state_mean']:.3f}")
            print(f"    Diff kinase, diff state: {paired['diff_kinase_diff_state_mean']:.3f}")
            print(f"    >> {paired['interpretation']}")

            for pair in paired["same_kinase_diff_state_pairs"]:
                print(f"      {pair['kinase']:12s}: {pair['similarity']:.3f}")
        else:
            print(f"    {paired['error']}")

        result = {
            "method": method_name,
            "dfg_1nn": acc1,
            "dfg_3nn": acc3,
            "dfg_cohen_d": d,
            "dfg_p_value": p,
            "dfg_gap": gap,
            "kinase_1nn": acc1_kin,
            "kinase_cohen_d": d_kin,
            "kinase_p_value": p_kin,
            "paired_analysis": paired,
        }
        all_results.append(result)

        # --- Plots ---
        plot_pca_dfg(emb, dfg_labels, kinase_labels,
                     os.path.join(args.out, f"fig_pca_dfg_{method_name}.png"),
                     f"{method_name}: PCA colored by DFG state")

        plot_paired_distances(paired,
                              os.path.join(args.out, f"fig_paired_{method_name}.png"),
                              f"{method_name}: Paired distance analysis")

        plot_per_kinase_pairs(paired,
                              os.path.join(args.out, f"fig_per_kinase_{method_name}.png"),
                              f"{method_name}: Per-kinase DFG-in vs DFG-out similarity")

    # ------- Summary -------
    print(f"\n{'='*70}")
    print("DFG EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':20s} {'DFG 1-NN':>10s} {'DFG d':>10s} {'Kin 1-NN':>10s} {'Kin d':>10s} {'Identity?':>12s}")
    print("-" * 75)
    for r in all_results:
        pa = r["paired_analysis"]
        ident = "YES" if pa.get("identity_dominates", True) else "NO"
        print(f"{r['method']:20s} {r['dfg_1nn']:10.3f} {r['dfg_cohen_d']:10.3f} "
              f"{r['kinase_1nn']:10.3f} {r['kinase_cohen_d']:10.3f} {ident:>12s}")

    n_in = sum(1 for k in keys if dfg_labels[k] == "DFG-in")
    n_out = sum(1 for k in keys if dfg_labels[k] == "DFG-out")
    print(f"\nDFG-in: {n_in}, DFG-out: {n_out}")
    print(f"Random baseline: 0.500")
    majority = max(n_in, n_out) / len(keys)
    print(f"Majority baseline: {majority:.3f}")

    # Save results
    csv_rows = []
    for r in all_results:
        csv_rows.append({
            "method": r["method"],
            "dfg_1nn": r["dfg_1nn"],
            "dfg_3nn": r["dfg_3nn"],
            "dfg_cohen_d": r["dfg_cohen_d"],
            "dfg_p_value": r["dfg_p_value"],
            "kinase_1nn": r["kinase_1nn"],
            "kinase_cohen_d": r["kinase_cohen_d"],
        })
    pd.DataFrame(csv_rows).to_csv(os.path.join(args.out, "dfg_results.csv"), index=False)

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(args.out, "dfg_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nAll results saved to {args.out}")


if __name__ == "__main__":
    main()
