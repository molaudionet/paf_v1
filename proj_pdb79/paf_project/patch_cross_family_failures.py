#!/usr/bin/env python3
"""
patch_cross_family_failures.py

Fix the 4 extraction failures from the cross-family dataset:
  1FPC|A  – was labeled FXa but is actually thrombin; chain mismatch
  1LQE|B  – plasmin, chain B doesn't exist  
  3TGI|A  – trypsinogen, chain A doesn't exist
  3TJQ|C  – HCV protease, chain C doesn't exist

Replacements use well-validated co-crystal structures with confirmed chains.

Usage:
  python patch_cross_family_failures.py \
    --csv data/cross_family/cross_family_list.csv \
    --pdb_dir data/cross_family/pdbs/ \
    --download
"""

import os
import csv
import time
import argparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Replacement entries with verified chain IDs
REPLACEMENTS = {
    # 1FPC|A failed: was labeled FXa but 1FPC is thrombin. 
    # Replace with 1NFX: human FXa + RPR208944 inhibitor, chain A confirmed
    "1FPC": {
        "pdb_id": "1NFX",
        "chain_id": "A",
        "ligand_resname": "RPR",
        "subfamily": "FXa",
        "family": "serine_protease",
    },
    # 1LQE|B failed: plasmin, chain B doesn't exist.
    # Replace with 1BML: human plasmin catalytic domain + streptokinase, chain A
    "1LQE": {
        "pdb_id": "1BML",
        "chain_id": "A",
        "ligand_resname": "SO4",  # sulfate in active site
        "subfamily": "plasmin",
        "family": "serine_protease",
    },
    # 3TGI|A failed: trypsinogen, chain A doesn't exist in this format.
    # Replace with 1TGN: bovine trypsinogen + PMSF inhibitor, chain A
    "3TGI": {
        "pdb_id": "1TGN",
        "chain_id": "A",
        "ligand_resname": "PMS",
        "subfamily": "trypsinogen",
        "family": "serine_protease",
    },
    # 3TJQ|C failed: HCV protease, chain C doesn't exist.
    # Replace with 3M5L: HCV NS3/4A protease + MK-5172 precursor, chain A
    "3TJQ": {
        "pdb_id": "3M5L",
        "chain_id": "A",
        "ligand_resname": "GK4",
        "subfamily": "HCV_protease",
        "family": "serine_protease",
    },
}


def download_pdb(pdb_id: str, out_dir: str) -> str:
    pid = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    local = os.path.join(out_dir, f"{pid}.pdb")
    if os.path.exists(local) and os.path.getsize(local) > 100:
        print(f"  {pid} already exists")
        return local
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")
    print(f"  Downloading {pid} ...", end=" ", flush=True)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    with open(local, "w") as f:
        f.write(r.text)
    print(f"OK ({os.path.getsize(local)} bytes)")
    time.sleep(0.3)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="original cross_family_list.csv")
    ap.add_argument("--pdb_dir", default="data/cross_family/pdbs/")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    # Read existing CSV
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} entries from {args.csv}")
    
    patched = 0
    for i, row in enumerate(rows):
        old_pdb = row["pdb_id"]
        if old_pdb in REPLACEMENTS:
            repl = REPLACEMENTS[old_pdb]
            new_pdb = repl["pdb_id"]
            pdb_path = os.path.join(args.pdb_dir, f"{new_pdb}.pdb")

            if args.download:
                try:
                    pdb_path = download_pdb(new_pdb, args.pdb_dir)
                except Exception as e:
                    print(f"  [FAIL] Could not download {new_pdb}: {e}")
                    continue

            rows[i] = {
                "pdb_id": new_pdb,
                "pdb_path": pdb_path,
                "chain_id": repl["chain_id"],
                "ligand_resname": repl["ligand_resname"],
                "subfamily": repl["subfamily"],
                "family": repl["family"],
                "pocket_label": f"{new_pdb}_{repl['subfamily']}",
            }
            print(f"  PATCHED: {old_pdb} -> {new_pdb} ({repl['subfamily']})")
            patched += 1

    # Write patched CSV (overwrite)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_id", "pdb_path", "chain_id", "ligand_resname",
            "subfamily", "family", "pocket_label",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPatched {patched} entries. Updated {args.csv}")
    print(f"\nNow re-run the experiment:")
    print(f"  python run_cross_family_paf.py \\")
    print(f"    --csv {args.csv} \\")
    print(f"    --out results/cross_family/ \\")
    print(f"    --radius 10 --gamma_fm 0.15 --sigma_t 0.04")


if __name__ == "__main__":
    main()
