#!/usr/bin/env python3
"""
fix_cocrystal_csv.py

Fixes cocrystal_pairs.csv by replacing wrong ligand_resname values
with the actual largest non-solvent HETATM residue found in each PDB.

Usage:
  python fix_cocrystal_csv.py \
    --csv data/cocrystal_pairs.csv \
    --pdb_dir . \
    --out data/cocrystal_pairs_fixed.csv
"""

import argparse
import os
import pandas as pd

# Molecules that are solvents/ions/additives — never the real ligand
_SOLVENT_BLACKLIST = {
    "HOH", "WAT", "DOD", "H2O",
    "NA", "K", "CL", "BR", "IOD",
    "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI",
    "SO4", "PO4", "NO3",
    "GOL", "EDO", "PEG", "DMS", "MPD",
    "MES", "TRS", "HEP", "EPE", "FMT",
    "ACT", "ACE", "BOG",
    "SEP", "TPO", "PTR",  # phosphorylated residues (modifications, not ligands)
}


def find_best_ligand(pdb_path: str, target_chain: str = "A") -> dict:
    """
    Parse a PDB file and find the best (largest) non-solvent HETATM ligand.
    Returns dict with resname, heavy_atom_count, all_het_resnames.
    """
    het_atoms = {}  # resname -> count of heavy atoms
    
    if not os.path.exists(pdb_path):
        return {"resname": None, "heavy_atoms": 0, "all_het": [], "error": "file_not_found"}

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resn = line[17:20].strip().upper()
            if resn in {"HOH", "WAT", "DOD"}:
                continue
            
            # Element check (heavy atoms only)
            elem = line[76:78].strip().upper()
            if not elem:
                atom_name = line[12:16].strip()
                elem = "".join(c for c in atom_name if c.isalpha())[:2].upper()
            if elem in ("H", "D"):
                continue
            
            het_atoms[resn] = het_atoms.get(resn, 0) + 1
    
    all_het = sorted(het_atoms.keys())
    
    # Filter out solvents/ions
    candidates = {r: n for r, n in het_atoms.items() if r not in _SOLVENT_BLACKLIST}
    
    if not candidates:
        # If everything was filtered, try with a looser filter (only remove tiny ions)
        tiny_ions = {"NA", "K", "CL", "BR", "IOD", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI"}
        candidates = {r: n for r, n in het_atoms.items() 
                      if r not in tiny_ions and r not in {"HOH", "WAT", "DOD"}}
    
    if not candidates:
        return {"resname": None, "heavy_atoms": 0, "all_het": all_het, "error": "no_ligand_candidates"}
    
    # Pick the one with most heavy atoms
    best = max(candidates, key=lambda r: candidates[r])
    return {"resname": best, "heavy_atoms": candidates[best], "all_het": all_het, "error": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pdb_dir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    
    fixed_resnames = []
    changes = []
    
    print(f"\n{'='*80}")
    print(f"  Fixing ligand_resname in CSV ({len(df)} entries)")
    print(f"{'='*80}\n")
    
    for i, row in df.iterrows():
        pdb_file = str(row["pdb_path"])
        chain = str(row.get("protein_chain", "A"))
        old_lig = str(row["ligand_resname"]).strip()
        
        pdb_path = os.path.join(args.pdb_dir, pdb_file) if not os.path.isabs(pdb_file) else pdb_file
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(args.pdb_dir, os.path.basename(pdb_file))
        
        result = find_best_ligand(pdb_path, chain)
        
        # Check if old resname is actually in the file
        old_is_valid = old_lig in (result.get("all_het") or [])
        
        if old_is_valid:
            # Original was correct
            fixed_resnames.append(old_lig)
            print(f"  [OK  ] {pdb_file}|{chain}|{old_lig} — present in file")
        elif result["resname"]:
            # Replace with best candidate
            new_lig = result["resname"]
            fixed_resnames.append(new_lig)
            changes.append((pdb_file, old_lig, new_lig, result["heavy_atoms"]))
            print(f"  [FIX ] {pdb_file}|{chain}|{old_lig} → {new_lig} "
                  f"({result['heavy_atoms']} heavy atoms) "
                  f"[available: {result['all_het']}]")
        else:
            # Can't fix — keep original
            fixed_resnames.append(old_lig)
            print(f"  [FAIL] {pdb_file}|{chain}|{old_lig} — no valid ligand in file "
                  f"[het: {result['all_het']}]")
    
    df["ligand_resname_original"] = df["ligand_resname"]
    df["ligand_resname"] = fixed_resnames
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"  Total entries:    {len(df)}")
    print(f"  Already correct:  {len(df) - len(changes)}")
    print(f"  Fixed:            {len(changes)}")
    print(f"  Wrote: {args.out}")
    print(f"{'='*80}\n")
    
    if changes:
        print("  Changes made:")
        for pdb, old, new, ha in changes:
            print(f"    {pdb}: {old} → {new} ({ha} atoms)")
        print()


if __name__ == "__main__":
    main()
