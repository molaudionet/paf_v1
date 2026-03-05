#!/usr/bin/env python3
"""
redownload_pdbs.py

Re-downloads PDB files from RCSB with full HETATM records.
This fixes the #1 failure mode: "No HETATM coords found for ligand_resname=XXX"

Usage:
  python redownload_pdbs.py --csv data/cocrystal_pairs.csv --outdir .

It reads pdb_path from the CSV, extracts the 4-letter PDB ID, and downloads
the full PDB file (with HETATM) from RCSB.
"""

import argparse
import os
import re
import sys
import time
import urllib.request
import urllib.error
import pandas as pd


def extract_pdb_id(pdb_path: str) -> str:
    """Extract 4-letter PDB ID from a filename like '1M17.pdb' or 'data/pdbs/1M17.pdb'."""
    base = os.path.basename(pdb_path)
    # Match 4-char alphanumeric PDB ID
    m = re.match(r"^([A-Za-z0-9]{4})\.", base)
    if m:
        return m.group(1).upper()
    # fallback: first 4 chars
    return base[:4].upper()


def download_pdb(pdb_id: str, outpath: str, max_retries: int = 3) -> bool:
    """Download a PDB file from RCSB. Returns True on success."""
    pdb_id_lower = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PAF-Pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            # Verify it has HETATM records
            text = data.decode("utf-8", errors="ignore")
            hetatm_count = text.count("\nHETATM")
            atom_count = text.count("\nATOM  ")

            with open(outpath, "wb") as f:
                f.write(data)

            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  [404] {pdb_id} not found on RCSB (may be obsolete)")
                return False
            print(f"  [HTTP {e.code}] Retry {attempt+1}/{max_retries}...")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERR] {e}, retry {attempt+1}/{max_retries}...")
            time.sleep(2)

    return False


def check_hetatm_in_pdb(filepath: str, ligand_resname: str) -> dict:
    """Check what HETATM records exist in a PDB file."""
    info = {"exists": os.path.exists(filepath), "has_hetatm": False,
            "hetatm_resnames": set(), "ligand_found": False, "ligand_atoms": 0}
    if not info["exists"]:
        return info

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("HETATM"):
                info["has_hetatm"] = True
                resn = line[17:20].strip()
                info["hetatm_resnames"].add(resn)
                if resn == ligand_resname:
                    info["ligand_found"] = True
                    elem = line[76:78].strip().upper()
                    if elem != "H":
                        info["ligand_atoms"] += 1

    info["hetatm_resnames"] = sorted(info["hetatm_resnames"])
    return info


def main():
    ap = argparse.ArgumentParser(description="Re-download PDB files from RCSB with HETATM records")
    ap.add_argument("--csv", required=True, help="Path to cocrystal_pairs.csv")
    ap.add_argument("--outdir", default=".", help="Output directory for PDB files")
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists and has HETATM")
    ap.add_argument("--diagnose-only", action="store_true", help="Only check existing files, don't download")
    ap.add_argument("--pdb_dir", default=".", help="Where existing PDBs live (for diagnosis)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    total = len(df)
    ok_before = 0
    ok_after = 0
    downloaded = 0
    failed_download = 0

    print(f"\n{'='*70}")
    print(f"  PDB HETATM Diagnosis & Download")
    print(f"  CSV: {args.csv} ({total} entries)")
    print(f"{'='*70}\n")

    for i, row in df.iterrows():
        pdb_path = str(row["pdb_path"])
        lig = str(row["ligand_resname"]).strip()
        chain = str(row.get("protein_chain", "A"))
        pdb_id = extract_pdb_id(pdb_path)

        # Check existing file
        existing_path = os.path.join(args.pdb_dir, pdb_path) if not os.path.isabs(pdb_path) else pdb_path
        if not os.path.exists(existing_path):
            existing_path = os.path.join(args.pdb_dir, os.path.basename(pdb_path))

        info_before = check_hetatm_in_pdb(existing_path, lig)
        status_before = "OK" if info_before["ligand_found"] else "MISSING"
        if info_before["ligand_found"]:
            ok_before += 1

        if args.diagnose_only:
            het_list = ", ".join(info_before["hetatm_resnames"][:10]) if info_before["hetatm_resnames"] else "(none)"
            print(f"  [{status_before:7s}] {pdb_id}|{chain}|{lig}  "
                  f"HETATM={info_before['has_hetatm']}  "
                  f"lig_atoms={info_before['ligand_atoms']}  "
                  f"het_resnames=[{het_list}]")
            continue

        # Download if needed
        outpath = os.path.join(args.outdir, f"{pdb_id}.pdb")
        need_download = args.force or not info_before["ligand_found"]

        if not need_download:
            ok_after += 1
            print(f"  [SKIP] {pdb_id}|{chain}|{lig} - already has ligand HETATM")
            continue

        print(f"  [DL]   {pdb_id}|{chain}|{lig} - downloading from RCSB...", end=" ")
        success = download_pdb(pdb_id, outpath)

        if success:
            info_after = check_hetatm_in_pdb(outpath, lig)
            if info_after["ligand_found"]:
                ok_after += 1
                downloaded += 1
                print(f"OK ({info_after['ligand_atoms']} heavy atoms)")
            else:
                het_list = ", ".join(info_after["hetatm_resnames"][:8])
                print(f"Downloaded but ligand '{lig}' not found. Available: [{het_list}]")
                failed_download += 1
        else:
            failed_download += 1
            print("FAILED")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Total entries:        {total}")
    print(f"  Had ligand before:    {ok_before}")
    if not args.diagnose_only:
        print(f"  Have ligand after:    {ok_after}")
        print(f"  Newly downloaded:     {downloaded}")
        print(f"  Failed/not found:     {failed_download}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
