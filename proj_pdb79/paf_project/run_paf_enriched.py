# run_paf_enriched.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

from paf_core_v1 import PAFParams, extract_pocket, pocket_to_embedding
from evaluate_metrics import pairwise_similarity, loo_knn_accuracy, within_between_stats

_3TO1 = {
 "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
 "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
 "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
 "MSE":"M"
}
def three_to_one_safe(resname: str) -> str:
    r = resname.strip().upper()
    if r in _3TO1:
        return _3TO1[r]
    raise KeyError(r)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinase_csv", required=True, help="path to kinase_list.csv")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--gamma_fm", type=float, default=0.15)
    ap.add_argument("--sigma_t", type=float, default=0.040)
    ap.add_argument("--a_hyd", type=float, default=1.0)
    ap.add_argument("--a_charge", type=float, default=1.0)
    ap.add_argument("--a_vol", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    params = PAFParams(
        pocket_radius_A=args.radius,
        gamma_fm=args.gamma_fm,
        sigma_t_s=args.sigma_t,
        a_hyd=args.a_hyd,
        a_charge=args.a_charge,
        a_vol=args.a_vol
    )

    df = pd.read_csv(args.kinase_csv)

    emb = {}
    labels = {}
    meta = {}
    failures = []

    chan_names_ref = None

    for _, r in df.iterrows():
        pdb_path = str(r["pdb_path"])
        chain_id = str(r["chain_id"])
        ligand_resname = str(r["ligand_resname"]) if "ligand_resname" in r and not pd.isna(r["ligand_resname"]) else ""
        pocket_label = str(r["pocket_label"])
        family = str(r["family"])

        key = f"{os.path.basename(pdb_path)}|{chain_id}"

        try:
            pocket = extract_pocket(pdb_path=pdb_path, chain_id=chain_id, ligand_resname=ligand_resname, params=params)
            spec, chan_names = pocket_to_embedding(pocket, params)
            if chan_names_ref is None:
                chan_names_ref = chan_names
            emb[key] = spec
            labels[key] = family
            meta[key] = {
                "pdb_path": pdb_path,
                "chain_id": chain_id,
                "ligand_resname": ligand_resname,
                "pocket_label": pocket_label,
                "family": family,
                "center_method": pocket.meta.get("center_method", ""),
                "anchor2_method": pocket.meta.get("anchor2_method", ""),
                "n_residues": len(pocket.residues),
            }
            print(f"[OK] {key} fam={family} nres={len(pocket.residues)} center={meta[key]['center_method']}")
        except Exception as e:
            failures.append({"key": key, "pdb_path": pdb_path, "chain_id": chain_id, "err": str(e)})
            print(f"[FAIL] {key}: {e}")

    # Save
    np.savez_compressed(os.path.join(args.out, "embeddings_paf_enriched.npz"), **emb)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out, "channels.json"), "w") as f:
        json.dump(chan_names_ref or [], f, indent=2)
    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(args.out, "extraction_failures.csv"), index=False)

    # Evaluate
    keys = list(emb.keys())
    if len(keys) < 3:
        print(f"[FATAL] Only {len(keys)} embeddings extracted. Check errors above.")
        return
    S = pairwise_similarity(keys, emb)
    acc1 = loo_knn_accuracy(keys, labels, S, k=1)
    acc3 = loo_knn_accuracy(keys, labels, S, k=3)
    d, p, gap = within_between_stats(keys, labels, S)

    summary = {
        "n": len(keys),
        "1nn": acc1,
        "3nn": acc3,
        "cohen_d": d,
        "p_value": p,
        "mean_within_minus_between": gap,
        "params": {
            "radius": args.radius,
            "gamma_fm": args.gamma_fm,
            "sigma_t": args.sigma_t,
            "a_hyd": args.a_hyd,
            "a_charge": args.a_charge,
            "a_vol": args.a_vol
        }
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
