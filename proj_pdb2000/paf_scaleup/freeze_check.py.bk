#!/usr/bin/env python3
"""
freeze_check.py  --  READ-ONLY diagnostic. Changes nothing.

Purpose: before we "freeze" the dataset, find out what we actually have:
  1. Does any results file already record WHICH PDB IDs encoded successfully
     (not just the counts)?
  2. How many structures are in each manifest vs. how many PDB files exist
     vs. how many actually encoded?
  3. Surface the exact reproducible N so the manuscript can be set to match it.

Usage (from your proj_pdb2000/paf_scaleup directory, or pass --root):
    python3 freeze_check.py
    python3 freeze_check.py --root /Users/jzhou/bk/bioai/paf_v1_freeze/proj_pdb2000/paf_scaleup
"""
import argparse, json, os, glob, csv, sys

def find(root, *names):
    hits = []
    for n in names:
        hits += glob.glob(os.path.join(root, "**", n), recursive=True)
    return sorted(set(hits))

def count_csv_ids(path):
    try:
        with open(path) as f:
            rows = [r for r in csv.reader(f) if r and r[0].strip()]
        # assume header present
        return max(0, len(rows) - 1), rows[0] if rows else []
    except Exception as e:
        return None, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root to scan")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    print(f"Scanning: {root}\n")

    # 1) manifests
    print("== MANIFESTS (curated input lists) ==")
    for m in find(root, "cross_family_manifest.csv", "kinase_manifest.csv",
                  "*manifest*.csv", "*frozen*.csv"):
        n, hdr = count_csv_ids(m)
        print(f"  {os.path.relpath(m, root):55}  ids={n}")
    print()

    # 2) results JSONs -- do they list successful IDs?
    print("== RESULTS FILES (looking for saved successful-ID lists) ==")
    saved_ids_found = False
    for j in find(root, "*results*.json", "*.json"):
        try:
            with open(j) as f:
                data = json.load(f)
        except Exception:
            continue
        # look for any list of PDB-like IDs inside
        def scan(obj, pathkey=""):
            nonlocal saved_ids_found
            found = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    found += scan(v, f"{pathkey}.{k}")
            elif isinstance(obj, list):
                # PDB IDs are 4-char alphanumeric
                idlike = [x for x in obj if isinstance(x, str) and len(x) == 4 and x.isalnum()]
                if len(idlike) >= 10:
                    found.append((pathkey, len(idlike)))
            return found
        hits = scan(data)
        if hits:
            saved_ids_found = True
            print(f"  {os.path.relpath(j, root)}")
            for key, cnt in hits:
                print(f"      -> list at '{key.lstrip('.')}'  ({cnt} ids)  <-- usable for freezing")
    if not saved_ids_found:
        print("  No saved successful-ID lists found in any JSON.")
        print("  (Means: the run printed counts but did not persist WHICH ids encoded.)")
    print()

    # 3) PDB files physically present
    pdbdirs = find(root, "pdbs")
    print("== PDB FILES ON DISK ==")
    for d in pdbdirs:
        if os.path.isdir(d):
            n = len(glob.glob(os.path.join(d, "*.pdb"))) + len(glob.glob(os.path.join(d, "*.cif")))
            print(f"  {os.path.relpath(d, root):55}  files={n}")
    print()

    # 4) verdict
    print("== WHAT THIS MEANS ==")
    if saved_ids_found:
        print("  GOOD: a successful-ID list exists -> we can freeze EXACTLY that set,")
        print("  set the manuscript N to its count, and reproduce identically forever.")
    else:
        print("  The successful-ID set was not persisted. To freeze cleanly we should")
        print("  modify the encoder to WRITE OUT the ids it successfully encodes on the")
        print("  next run, then freeze that file. (One small, safe addition -- I can")
        print("  give you the exact code to add.)")
    print()
    print("  Next: whatever the frozen count turns out to be, set the paper's N to it")
    print("  so the repository and manuscript agree exactly.")

if __name__ == "__main__":
    main()
