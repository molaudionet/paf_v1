#!/usr/bin/env python3
"""
paf_freeze.py  --  make the PAF dataset reproducible at a FIXED N.

The problem this solves:
  Your pipeline re-curates from the live PDB every run (Step 1 -> 1683),
  re-downloads, and re-encodes; the ~200 that fail to encode vary slightly
  run-to-run, so N lands on 1482 / 1483 / ... and never matches a fixed
  number. The successful-ID set was never persisted, so nothing is frozen.

The fix (two modes):

  MODE 1 -- CAPTURE (run once):
    After (or during) an encoding run, record the EXACT PDB IDs that
    encoded successfully into a frozen manifest. That frozen file is your
    permanent dataset. Its row count IS your reproducible N.

  MODE 2 -- VERIFY (run anytime):
    Check that a frozen manifest still matches what encodes now, and report
    the exact N to put in the manuscript.

This file does NOT modify your encoder. It only reads results / writes a
frozen list. Integrate the 6-line snippet at the bottom into your encoder
loop if you want capture to happen automatically on the next run.
"""

import argparse, csv, json, os, sys

# ----------------------------------------------------------------------
# MODE 1: build a frozen manifest from a list of successful IDs.
# You can supply the successful IDs either as a text file (one ID per line)
# or let your encoder write them (see snippet at bottom).
# ----------------------------------------------------------------------
def freeze(successful_ids_path, source_manifest, out_path):
    with open(successful_ids_path) as f:
        ok_ids = [ln.strip() for ln in f if ln.strip()]
    ok_set = set(i.lower() for i in ok_ids)

    # carry over the family/label columns from the source manifest,
    # keeping ONLY the rows whose PDB id encoded successfully.
    kept = []
    with open(source_manifest) as f:
        reader = csv.reader(f)
        header = next(reader)
        # find the column holding the PDB id (first 4-char alnum column)
        id_col = 0
        for row in reader:
            if not row:
                continue
            pid = row[id_col].strip().lower()
            if pid in ok_set:
                kept.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    print(f"  Frozen manifest written: {out_path}")
    print(f"  Successful IDs supplied : {len(ok_set)}")
    print(f"  Rows kept (frozen N)    : {len(kept)}")
    if len(kept) != len(ok_set):
        print(f"  NOTE: {len(ok_set)-len(kept)} successful IDs were not found in the")
        print(f"        source manifest (id-format mismatch?). Inspect if the gap is large.")
    print(f"\n  >>> Put N = {len(kept)} in the manuscript for this dataset. <<<")
    return len(kept)

# ----------------------------------------------------------------------
# MODE 2: verify a frozen manifest and report N.
# ----------------------------------------------------------------------
def verify(frozen_path, pdb_dir=None):
    with open(frozen_path) as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    n = len(rows) - 1
    print(f"  Frozen manifest: {frozen_path}")
    print(f"  Frozen N       : {n}")
    if pdb_dir:
        have = 0; missing = []
        for r in rows[1:]:
            pid = r[0].strip().lower()
            p1 = os.path.join(pdb_dir, pid + ".pdb")
            p2 = os.path.join(pdb_dir, pid + ".cif")
            if os.path.exists(p1) or os.path.exists(p2):
                have += 1
            else:
                missing.append(pid)
        print(f"  PDB files present for frozen set: {have}/{n}")
        if missing:
            print(f"  Missing ({len(missing)}): {', '.join(missing[:10])}{' ...' if len(missing)>10 else ''}")
    print(f"\n  >>> Manuscript N for this dataset = {n} <<<")
    return n

def main():
    ap = argparse.ArgumentParser(description="Freeze / verify the PAF dataset to a fixed N.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="build a frozen manifest from successful IDs")
    f.add_argument("--successful", required=True, help="text file: one successful PDB id per line")
    f.add_argument("--manifest", required=True, help="source manifest csv (e.g. cross_family_manifest.csv)")
    f.add_argument("--out", required=True, help="output frozen csv (e.g. cross_family_FROZEN.csv)")

    v = sub.add_parser("verify", help="report N of a frozen manifest")
    v.add_argument("--frozen", required=True)
    v.add_argument("--pdbdir", default=None, help="optional: check pdb files exist")

    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.successful, a.manifest, a.out)
    elif a.cmd == "verify":
        verify(a.frozen, a.pdbdir)

if __name__ == "__main__":
    main()

# ======================================================================
# SNIPPET to add to your encoder loop (in paf_core_v1.py or the experiment
# script) so the NEXT run records exactly which ids encoded. ~6 lines.
#
# Find the loop where each pocket is encoded. Where you currently count a
# success, also append the id to a list, and dump it at the end:
#
#     successful_ids = []                      # <-- before the loop
#     ...
#     for entry in entries:
#         try:
#             emb = encode_pocket(entry)        # your existing call
#             ...                               # your existing success path
#             successful_ids.append(entry_pdb_id)   # <-- ADD: record the id
#         except Exception:
#             ...                               # your existing fail path
#     ...
#     with open("successful_ids_cross_family.txt", "w") as fh:   # <-- after loop
#         fh.write("\n".join(successful_ids))
#
# Then run once, and:
#     python3 paf_freeze.py freeze \
#         --successful successful_ids_cross_family.txt \
#         --manifest data/cross_family_manifest.csv \
#         --out data/cross_family_FROZEN.csv
#
# Finally, point Step 1 of run_all.sh to the FROZEN csv instead of
# re-curating, and N is locked forever.
# ======================================================================
