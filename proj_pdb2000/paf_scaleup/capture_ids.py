#!/usr/bin/env python3
"""
capture_ids.py -- write out the successful+kept PDB IDs WITHOUT editing
run_experiments.py. Reuses your existing encoder and applies the SAME
>=min_size per-family mask, so the count matches your reported N exactly.

Place this file in the SAME folder as run_experiments.py and run:

    python3 capture_ids.py \
        --manifest data/cross_family_manifest.csv \
        --pdbdir   data/pdbs \
        --out      successful_ids_cross_family.txt

Then freeze:
    python3 paf_freeze.py freeze \
        --successful successful_ids_cross_family.txt \
        --manifest   data/cross_family_manifest.csv \
        --out        data/cross_family_FROZEN.csv
"""
import argparse, csv, os, sys
import numpy as np
from collections import Counter

# reuse YOUR encoder exactly as run_experiments.py does
from spectral_encoder import encode_all_methods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdbdir",   required=True)
    ap.add_argument("--out",      required=True)
    ap.add_argument("--min_size", type=int, default=5,
                    help="min structures per family to keep (match run_experiments.py)")
    a = ap.parse_args()

    # load manifest exactly like run_experiments.py (csv.DictReader)
    entries = []
    with open(a.manifest) as f:
        for row in csv.DictReader(f):
            entries.append(row)
    print(f"  loaded {len(entries)} manifest entries")

    # encode with the REAL encoder (same call run_experiments.py uses)
    encoded = encode_all_methods(entries, a.pdbdir, verbose=True)
    if not encoded:
        print("  ERROR: nothing encoded."); sys.exit(1)

    valid_entries = encoded["valid_entries"]
    print(f"  encoded successfully: {len(valid_entries)}")

    # apply the SAME >=min_size per-family mask
    fam_counts = Counter(e["family"] for e in valid_entries)
    valid_families = {f for f, n in fam_counts.items() if n >= a.min_size}
    mask = np.array([e["family"] in valid_families for e in valid_entries])

    # figure out the pdb-id field name robustly
    sample = valid_entries[0]
    id_key = None
    for cand in ("pdb_id", "pdb", "pdbid", "id", "PDB", "pdbId"):
        if cand in sample:
            id_key = cand; break
    if id_key is None:
        print("  ERROR: could not find a pdb-id field in entries. Keys are:")
        print("   ", list(sample.keys()))
        sys.exit(1)
    print(f"  using id field: '{id_key}'")

    kept_ids = np.array([e[id_key] for e in valid_entries])[mask]
    with open(a.out, "w") as fh:
        fh.write("\n".join(str(x) for x in kept_ids))

    print(f"\n  kept (successful + family>={a.min_size}): {len(kept_ids)}")
    print(f"  >>> THIS is your reproducible N: {len(kept_ids)} <<<")
    print(f"  wrote -> {a.out}")

if __name__ == "__main__":
    main()
