#!/usr/bin/env python3
"""
compare_paf_vs_baselines_v1.py

Head-to-head comparison on the SAME extracted pockets:
  - PAF v1 (wave+FFT) via pocket_to_embedding()  -> flattened to 1D
  - FP-A v1: mean(30-d residue features)          (no wave)
  - FP-B v1: radial histogram (n_shells x 30)     (no wave)
  - FP-C v1: sorted concat (max_res x 30)         (no wave)

Outputs (in --out):
  comparison_results.csv
  summary_<method>.json
  fig_comparison_bar.png
  fig_within_between_<method>.png
  fig_pca_<method>.png (unless --no_pca)
  extraction_failures.csv (if any)
  emb_*.npz, labels.json, meta.json

Run (example, best ablation setting you found):
  python compare_paf_vs_baselines_v1.py \
    --kinase_csv data/kinase_pdbs/kinase_list.csv \
    --out results/compare_v1_best/ \
    --radius 8 --gamma_fm 0.15 --sigma_t 0.06 --a_hyd 1 --a_charge 1 --a_vol 0
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paf_core_v1 import PAFParams, extract_pocket, pocket_to_embedding


# ----------------------------
# Similarity + metrics
# ----------------------------

def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    du = float(np.dot(u, u))
    dv = float(np.dot(v, v))
    if du < eps or dv < eps:
        return 0.0
    return float(np.dot(u, v) / (math.sqrt(du) * math.sqrt(dv)))


def pairwise_similarity(keys: List[str], emb: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(keys)
    S = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        S[i, i] = 1.0
    for i in range(n):
        ui = emb[keys[i]]
        for j in range(i + 1, n):
            sj = _cosine(ui, emb[keys[j]])
            S[i, j] = sj
            S[j, i] = sj
    return S


def loo_knn_accuracy(keys: List[str], labels: Dict[str, str], S: np.ndarray, k: int = 1) -> float:
    n = len(keys)
    if n == 0:
        return 0.0
    correct = 0
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -1e9  # exclude self
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
    within = []
    between = []
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

    mw = float(within.mean())
    mb = float(between.mean())
    sw = float(within.std(ddof=1))
    sb = float(between.std(ddof=1))

    nw, nb = len(within), len(between)
    pooled = math.sqrt(
        ((nw - 1) * sw * sw + (nb - 1) * sb * sb) / (nw + nb - 2)
    ) if (nw + nb - 2) > 0 else 0.0

    d = (mw - mb) / pooled if pooled > 1e-12 else 0.0
    gap = mw - mb

    # Welch t-test approximation (normal approx for p-value; good enough for large nw/nb)
    denom = math.sqrt((sw * sw) / nw + (sb * sb) / nb) if (nw > 0 and nb > 0) else 1e9
    t = (mw - mb) / denom if denom > 1e-12 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))

    return d, p, gap, within, between


# ----------------------------
# Baseline feature builders
# ----------------------------

def residue_feature30(r) -> np.ndarray:
    """
    30-d vector:
      physchem: charge, hyd, hb, aro, vol (5)
      flex (1)
      contact (1)
      ss_onehot: helix, sheet, coil (3)
      blosum62 row (20)
    Total = 30
    """
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
    if out.shape[0] != 30:
        raise ValueError(f"feature30 wrong shape: {out.shape}")
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
    return np.concatenate(shell_feats, axis=0).astype(np.float32).reshape(-1)  # (n_shells*30,)


def fp_c_sorted_concat(pocket, max_res: int = 60) -> np.ndarray:
    rs = sorted(pocket.residues, key=lambda r: float(r.radial1_A))
    rs = rs[:max_res]
    blocks = [residue_feature30(r) for r in rs]
    X = np.concatenate(blocks, axis=0) if blocks else np.zeros((0,), dtype=np.float32)
    need = max_res * 30
    if X.shape[0] < need:
        X = np.concatenate([X, np.zeros((need - X.shape[0],), dtype=np.float32)], axis=0)
    return X.astype(np.float32).reshape(-1)


# ----------------------------
# Plot helpers
# ----------------------------

def plot_bar(df: pd.DataFrame, out_png: str):
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    w = 0.25

    plt.figure()
    plt.bar(x - w, df["1nn"].values, width=w, label="1-NN")
    plt.bar(x,     df["3nn"].values, width=w, label="3-NN")
    plt.bar(x + w, df["cohen_d"].values, width=w, label="Cohen's d")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.title("PAF vs non-wave baselines (same pockets)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_within_between(within: np.ndarray, between: np.ndarray, title: str, out_png: str):
    plt.figure()
    plt.boxplot([within, between], tick_labels=["within-family", "between-family"], showfliers=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pca(emb: Dict[str, np.ndarray], labels: Dict[str, str], out_png: str, title: str):
    keys = list(emb.keys())
    X = np.stack([np.asarray(emb[k], dtype=np.float32).reshape(-1) for k in keys], axis=0)
    X = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pc = U[:, :2] * S[:2]

    fams = [labels[k] for k in keys]
    uniq = sorted(set(fams))

    plt.figure()
    for fam in uniq:
        idx = [i for i, f in enumerate(fams) if f == fam]
        plt.scatter(pc[idx, 0], pc[idx, 1], label=fam, s=20)
    plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinase_csv", required=True)
    ap.add_argument("--out", required=True)

    # PAF params
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--gamma_fm", type=float, default=0.15)
    ap.add_argument("--sigma_t", type=float, default=0.06)
    ap.add_argument("--a_hyd", type=float, default=1.0)
    ap.add_argument("--a_charge", type=float, default=1.0)
    ap.add_argument("--a_vol", type=float, default=0.0)

    # baselines params
    ap.add_argument("--n_shells", type=int, default=16)
    ap.add_argument("--max_res", type=int, default=60)
    ap.add_argument("--no_pca", action="store_true")

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

    df = pd.read_csv(args.kinase_csv)

    labels: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    failures = []

    emb_paf: Dict[str, np.ndarray] = {}
    emb_fpa: Dict[str, np.ndarray] = {}
    emb_fpb: Dict[str, np.ndarray] = {}
    emb_fpc: Dict[str, np.ndarray] = {}

    for _, r in df.iterrows():
        pdb_path = str(r["pdb_path"])
        chain_id = str(r["chain_id"])
        ligand_resname = ""
        if "ligand_resname" in r and not pd.isna(r["ligand_resname"]):
            ligand_resname = str(r["ligand_resname"]).strip()
        family = str(r["family"])

        key = f"{os.path.basename(pdb_path)}|{chain_id}"

        try:
            pocket = extract_pocket(
                pdb_path=pdb_path,
                chain_id=chain_id,
                ligand_resname=ligand_resname,
                params=params,
            )

            # PAF: IMPORTANT -> flatten to 1D vector
            spec, _ = pocket_to_embedding(pocket, params)
            emb_paf[key] = np.asarray(spec, dtype=np.float32).reshape(-1)

            # baselines (already 1D, keep consistent)
            emb_fpa[key] = fp_a_mean(pocket).reshape(-1)
            emb_fpb[key] = fp_b_radial_hist(pocket, radius_A=args.radius, n_shells=args.n_shells).reshape(-1)
            emb_fpc[key] = fp_c_sorted_concat(pocket, max_res=args.max_res).reshape(-1)

            labels[key] = family
            meta[key] = {
                "pdb_path": pdb_path,
                "chain_id": chain_id,
                "ligand_resname": ligand_resname,
                "family": family,
                "center_method": pocket.meta.get("center_method", ""),
                "anchor2_method": pocket.meta.get("anchor2_method", ""),
                "n_residues": len(pocket.residues),
            }

            print(f"[OK] {key} fam={family} nres={len(pocket.residues)} center={meta[key]['center_method']}")

        except Exception as e:
            failures.append({"key": key, "pdb_path": pdb_path, "chain_id": chain_id, "err": str(e)})
            print(f"[FAIL] {key}: {e}")

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(args.out, "extraction_failures.csv"), index=False)

    keys = sorted(list(emb_paf.keys()))
    if len(keys) < 3:
        raise SystemExit(f"[FATAL] Only {len(keys)} pockets extracted. See extraction_failures.csv")

    # Ensure all methods have same keys
    for name, m in [("FP-A", emb_fpa), ("FP-B", emb_fpb), ("FP-C", emb_fpc)]:
        missing = set(keys) - set(m.keys())
        if missing:
            raise SystemExit(f"[FATAL] Missing keys in {name}: {sorted(list(missing))[:5]}")

    methods: Dict[str, Dict[str, np.ndarray]] = {
        "PAF_v1": emb_paf,
        "FP-A_mean": emb_fpa,
        "FP-B_radial": emb_fpb,
        "FP-C_sorted": emb_fpc,
    }

    rows = []
    for name, emb in methods.items():
        S = pairwise_similarity(keys, emb)
        acc1 = loo_knn_accuracy(keys, labels, S, k=1)
        acc3 = loo_knn_accuracy(keys, labels, S, k=3)
        d, p, gap, within, between = within_between_stats(keys, labels, S)

        summary = {
            "method": name,
            "n": len(keys),
            "1nn": acc1,
            "3nn": acc3,
            "cohen_d": d,
            "p_value": p,
            "within_between_gap": gap,
            "params": {
                "radius": args.radius,
                "gamma_fm": args.gamma_fm,
                "sigma_t": args.sigma_t,
                "a_hyd": args.a_hyd,
                "a_charge": args.a_charge,
                "a_vol": args.a_vol,
                "n_shells": args.n_shells,
                "max_res": args.max_res,
            },
        }
        with open(os.path.join(args.out, f"summary_{name}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        rows.append(summary)

        plot_within_between(
            within, between,
            f"{name}: within vs between",
            os.path.join(args.out, f"fig_within_between_{name}.png"),
        )

        if not args.no_pca:
            plot_pca(
                emb, labels,
                os.path.join(args.out, f"fig_pca_{name}.png"),
                title=f"{name} PCA (PC1 vs PC2)",
            )

    out_csv = os.path.join(args.out, "comparison_results.csv")
    out_df = pd.DataFrame([{
        "method": r["method"],
        "n": r["n"],
        "1nn": r["1nn"],
        "3nn": r["3nn"],
        "cohen_d": r["cohen_d"],
        "p_value": r["p_value"],
        "within_between_gap": r["within_between_gap"],
    } for r in rows]).sort_values(["1nn", "cohen_d"], ascending=False)

    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}\n")
    print(out_df.to_string(index=False))

    plot_bar(out_df, os.path.join(args.out, "fig_comparison_bar.png"))

    # Save embeddings/labels/meta for reuse
    np.savez_compressed(os.path.join(args.out, "emb_paf_v1.npz"), **emb_paf)
    np.savez_compressed(os.path.join(args.out, "emb_fp_a_mean.npz"), **emb_fpa)
    np.savez_compressed(os.path.join(args.out, "emb_fp_b_radial.npz"), **emb_fpb)
    np.savez_compressed(os.path.join(args.out, "emb_fp_c_sorted.npz"), **emb_fpc)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
