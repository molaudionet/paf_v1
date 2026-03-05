#!/usr/bin/env python3
"""
run_real_kinases.py
====================
Bridge script: reads real PDB files → extracts pockets → runs PAF vs baselines.

This connects:
  - PDB parsing + pocket extraction (from protein_sonification_scoping.py)
  - Head-to-head comparison (from head_to_head.py)

Usage:
  python run_real_kinases.py --pdb_list data/kinase_pdbs/kinase_list.csv \
                             --out results/kinase_v0/
"""

import os
import sys
import argparse
import traceback

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB.NeighborSearch import NeighborSearch


def three_to_one(resname: str) -> str:
    """Convert 3-letter amino acid code to 1-letter. Raises KeyError if unknown."""
    return protein_letters_3to1[resname]

# Import the comparison pipeline
from head_to_head import (
    Pocket, ResidueRecord, aa_features, run_head_to_head,
)


# ============================================================
# Pocket extraction from real PDB files
# ============================================================

def extract_pocket_from_pdb(
    pdb_path: str,
    chain_id: str,
    ligand_resname: str = None,
    pocket_radius: float = 10.0,
    family: str = "",
    kinase_name: str = "",
) -> Pocket:
    """
    Extract binding pocket from a PDB file.

    Strategy:
      1. If ligand_resname is given → find it → use its centroid as pocket center
      2. Else → try to find largest HETATM group as ligand
      3. Select protein residues within pocket_radius of center
      4. Compute features + flexibility for each residue

    Returns a Pocket object compatible with head_to_head.py
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    # Get the target chain
    if chain_id not in model:
        available = [c.id for c in model.get_chains()]
        raise ValueError(
            f"Chain '{chain_id}' not found in {pdb_path}. "
            f"Available chains: {available}"
        )
    chain = model[chain_id]

    # --- Find ligand center ---
    ligand_atoms = []

    if ligand_resname:
        # Look for specified ligand in ALL chains (sometimes ligand is in a different chain)
        for ch in model.get_chains():
            for res in ch.get_residues():
                hetflag = res.get_id()[0]
                if hetflag == " ":
                    continue
                rname = res.get_resname().strip()
                if rname == ligand_resname.strip():
                    for atom in res.get_atoms():
                        if atom.element != "H":
                            ligand_atoms.append(atom.get_coord())

    if not ligand_atoms:
        # Fallback: find largest HETATM group that isn't water/ion
        skip = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "SO4", "PO4",
                "GOL", "EDO", "ACE", "NH2", "DMS", "PEG", "MPD", "BME", "TRS"}
        best_group = []
        for ch in model.get_chains():
            for res in ch.get_residues():
                hetflag = res.get_id()[0]
                if hetflag == " ":
                    continue
                rname = res.get_resname().strip()
                if rname in skip:
                    continue
                atoms = [a.get_coord() for a in res.get_atoms() if a.element != "H"]
                if len(atoms) > len(best_group):
                    best_group = atoms

        if best_group:
            ligand_atoms = best_group

    if not ligand_atoms:
        raise ValueError(f"No ligand found in {pdb_path} (tried '{ligand_resname}' + fallback)")

    center = np.mean(np.array(ligand_atoms, dtype=np.float32), axis=0)

    # --- Extract pocket residues ---
    residues = []
    bfactors = []

    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ":
            continue
        if "CA" not in res:
            continue

        resname = res.get_resname()
        try:
            aa = three_to_one(resname)
        except KeyError:
            continue

        ca_coord = np.array(res["CA"].get_coord(), dtype=np.float32)
        dist = float(np.linalg.norm(ca_coord - center))

        if dist > pocket_radius:
            continue

        # Features
        feats = aa_features(aa)

        # B-factor (average of backbone atoms)
        bb_atoms = ["N", "CA", "C", "O"]
        bvals = [float(res[a].get_bfactor()) for a in bb_atoms if a in res]
        bfactor = float(np.mean(bvals)) if bvals else None
        if bfactor is not None:
            bfactors.append(bfactor)

        residues.append(ResidueRecord(
            aa=aa,
            ca_xyz=ca_coord,
            radial_A=dist,
            features=feats,
            flex=0.5,  # will normalize below
        ))

    if len(residues) < 5:
        raise ValueError(
            f"Only {len(residues)} pocket residues found in {pdb_path} "
            f"chain {chain_id} within {pocket_radius}Å. Too few."
        )

    # Normalize B-factors → flex (0..1)
    if bfactors:
        bmean = float(np.mean(bfactors))
        bstd = float(np.std(bfactors)) + 1e-8
        for rr in residues:
            # Find matching bfactor (approximate — we stored them in order)
            # For simplicity, re-extract
            pass

        # Re-extract with proper bfactor assignment
        idx = 0
        for res in chain.get_residues():
            hetflag, resseq, icode = res.get_id()
            if hetflag != " " or "CA" not in res:
                continue
            ca = np.array(res["CA"].get_coord(), dtype=np.float32)
            dist = float(np.linalg.norm(ca - center))
            if dist > pocket_radius:
                continue

            bb_atoms = ["N", "CA", "C", "O"]
            bvals = [float(res[a].get_bfactor()) for a in bb_atoms if a in res]
            if bvals and idx < len(residues):
                b = float(np.mean(bvals))
                z = (b - bmean) / bstd
                z = float(np.clip(z, -2.0, 2.0))
                residues[idx].flex = float((z + 2.0) / 4.0)
            idx += 1

    pocket_id = f"{kinase_name}|{os.path.basename(pdb_path)}"

    return Pocket(
        pocket_id=pocket_id,
        center_xyz=center,
        residues=residues,
        family=family,
    )


# ============================================================
# Main: load CSV → extract pockets → run comparison
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run PAF vs baselines on real kinase PDB structures"
    )
    parser.add_argument(
        "--pdb_list", required=True,
        help="CSV file with columns: pdb_path, chain_id, pocket_label, family, ligand_resname"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for results and figures"
    )
    parser.add_argument(
        "--radius", type=float, default=10.0,
        help="Pocket radius in Angstroms (default: 10.0)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.pdb_list)

    print(f"{'='*60}")
    print(f"Loading kinase pockets from {len(df)} entries...")
    print(f"{'='*60}\n")

    pockets = {}
    failed = []

    for i, row in df.iterrows():
        pdb_path = str(row["pdb_path"])
        chain_id = str(row["chain_id"])
        label = str(row.get("pocket_label", f"kinase_{i}"))
        family = str(row.get("family", "unknown"))
        ligand = str(row.get("ligand_resname", "")) if pd.notna(row.get("ligand_resname")) else None

        if not os.path.exists(pdb_path):
            print(f"  SKIP  {label:20s}  file not found: {pdb_path}")
            failed.append({"label": label, "reason": "file_not_found"})
            continue

        try:
            pocket = extract_pocket_from_pdb(
                pdb_path=pdb_path,
                chain_id=chain_id,
                ligand_resname=ligand,
                pocket_radius=args.radius,
                family=family,
                kinase_name=label,
            )
            n_res = len(pocket.residues)
            pockets[pocket.pocket_id] = pocket
            print(f"  OK    {label:20s}  residues={n_res:3d}  family={family}")

        except Exception as e:
            print(f"  FAIL  {label:20s}  {e}")
            failed.append({"label": label, "reason": str(e)})

    print(f"\n{'='*60}")
    print(f"Successfully extracted: {len(pockets)} pockets")
    print(f"Failed: {len(failed)} entries")
    print(f"{'='*60}\n")

    if len(pockets) < 10:
        print("ERROR: Too few pockets extracted (<10). Cannot run meaningful comparison.")
        print("Check PDB files and ligand names in your CSV.")
        sys.exit(1)

    # Save failure log
    os.makedirs(args.out, exist_ok=True)
    if failed:
        fail_df = pd.DataFrame(failed)
        fail_path = os.path.join(args.out, "extraction_failures.csv")
        fail_df.to_csv(fail_path, index=False)
        print(f"Failure log: {fail_path}\n")

    # Run the head-to-head comparison
    results = run_head_to_head(pockets, args.out)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Pockets analyzed: {len(pockets)}")
    print(f"Results saved to: {args.out}/")
    print(f"\nKey files:")
    print(f"  {args.out}/comparison_results.csv  ← Spearman rho table")
    print(f"  {args.out}/fig_comparison_bar.png   ← Main result figure")
    print(f"  {args.out}/fig_pca_paf.png          ← Family clustering")
    print(f"\nNext: open comparison_results.csv and check PAF vs FP-B Spearman rho.")
    print(f"See execution_guide.py for interpretation.")


if __name__ == "__main__":
    main()
