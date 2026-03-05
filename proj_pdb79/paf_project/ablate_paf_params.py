# ablate_paf_params.py
from __future__ import annotations
import os
import itertools
import pandas as pd
import subprocess
import json

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinase_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Small, interpretable ablation grid (fast)
    radii = [8.0, 10.0]
    gammas = [0.0, 0.10, 0.15]          # 0 = no second-anchor FM (tests whether angular info helps)
    sigmas = [0.020, 0.040, 0.060]
    weights = [
        (1.0, 1.0, 0.5),                # default
        (1.0, 0.0, 0.5),                # no charge
        (0.0, 1.0, 0.5),                # no hydro
        (1.0, 1.0, 0.0),                # no volume
    ]

    rows = []
    for radius, gamma, sigma, (a_hyd, a_charge, a_vol) in itertools.product(radii, gammas, sigmas, weights):
        tag = f"R{radius}_G{gamma}_S{sigma}_W{a_hyd}-{a_charge}-{a_vol}".replace(".", "p")
        out_dir = os.path.join(args.out, tag)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "python", "run_paf_enriched.py",
            "--kinase_csv", args.kinase_csv,
            "--out", out_dir,
            "--radius", str(radius),
            "--gamma_fm", str(gamma),
            "--sigma_t", str(sigma),
            "--a_hyd", str(a_hyd),
            "--a_charge", str(a_charge),
            "--a_vol", str(a_vol),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)

        summ_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summ_path):
            with open(summ_path, "r") as f:
                summ = json.load(f)
            rows.append({
                "tag": tag,
                "n": summ["n"],
                "1nn": summ["1nn"],
                "3nn": summ["3nn"],
                "cohen_d": summ["cohen_d"],
                "p_value": summ["p_value"],
                "within_between_gap": summ["mean_within_minus_between"],
                "radius": radius,
                "gamma_fm": gamma,
                "sigma_t": sigma,
                "a_hyd": a_hyd,
                "a_charge": a_charge,
                "a_vol": a_vol,
            })

    df = pd.DataFrame(rows).sort_values(["1nn", "cohen_d"], ascending=[False, False])
    out_csv = os.path.join(args.out, "ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
    print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
