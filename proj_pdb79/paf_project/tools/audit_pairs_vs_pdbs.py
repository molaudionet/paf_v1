#!/usr/bin/env python3
import csv
import os
import sys
from collections import Counter

def het_resnames_in_pdb(pdb_path: str):
    res = []
    try:
        with open(pdb_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("HETATM") and len(line) >= 20:
                    # PDB fixed columns: resname is cols 18-20 (1-indexed)
                    resname = line[17:20].strip()
                    if resname:
                        res.append(resname)
    except FileNotFoundError:
        return None, "missing_pdb"
    except Exception as e:
        return None, f"read_error:{e}"
    return sorted(set(res)), None

def main():
    if len(sys.argv) != 4:
        print("Usage: audit_pairs_vs_pdbs.py <pairs_csv> <pdb_dir> <out_csv>")
        sys.exit(1)

    pairs_csv, pdb_dir, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]

    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Heuristic: find likely column names
    def pick_col(cands):
        for c in cands:
            if c in rows[0]:
                return c
        return None

    col_pdb = pick_col(["pdb", "pdb_file", "pdb_path", "pdb_id"])
    col_chain = pick_col(["chain", "chain_id"])
    col_lig = pick_col(["ligand", "ligand_resname", "resname", "lig_resname"])

    if not col_pdb or not col_lig:
        print("Could not find required columns in CSV. Need pdb_path and ligand_resname columns.")
        print("Columns found:", list(rows[0].keys()))
        sys.exit(2)

    ok = 0
    miss = 0
    mism = 0
    missing_pdb = 0

    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) + ["status", "pdb_fullpath", "ligand_found", "het_resnames_sample"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            pdb_rel = r[col_pdb]
            lig = r[col_lig].strip()
            pdb_full = os.path.join(pdb_dir, pdb_rel)

            hets, err = het_resnames_in_pdb(pdb_full)
            out = dict(r)
            out["pdb_fullpath"] = pdb_full

            if err == "missing_pdb":
                out["status"] = "MISSING_PDB"
                out["ligand_found"] = ""
                out["het_resnames_sample"] = ""
                missing_pdb += 1
                w.writerow(out)
                continue
            if err:
                out["status"] = f"READ_ERROR:{err}"
                out["ligand_found"] = ""
                out["het_resnames_sample"] = ""
                w.writerow(out)
                continue

            # ligand present as HETATM?
            found = lig in hets
            out["ligand_found"] = "YES" if found else "NO"
            out["het_resnames_sample"] = ",".join(hets[:30])

            if found:
                out["status"] = "OK"
                ok += 1
            else:
                # If there are HETATMs but not that ligand, it's a mismatch
                if len(hets) == 0:
                    out["status"] = "NO_HETATM_IN_PDB"
                    miss += 1
                else:
                    out["status"] = "LIGAND_RESNAME_MISMATCH"
                    mism += 1

            w.writerow(out)

    print("Wrote:", out_csv)
    print("Summary:",
          f"OK={ok}",
          f"MISSING_PDB={missing_pdb}",
          f"NO_HETATM_IN_PDB={miss}",
          f"MISMATCH={mism}",
          sep=" | ")

if __name__ == "__main__":
    main()
