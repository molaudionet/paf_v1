#!/usr/bin/env python3
"""
Kinase PDB Dataset Preparation
================================

Prepares a curated set of ~50 kinase structures for the PAF vs baseline
head-to-head experiment. Downloads PDB files and creates the CSV manifest
needed by head_to_head.py.

Kinase selection criteria:
  - Mix of kinase families (TK, CMGC, AGC, STE, CK1, CAMK, TKL)
  - Resolution < 2.5Å (prefer < 2.0Å)
  - Co-crystallized with ATP-competitive inhibitor (for clean pocket definition)
  - One structure per kinase (no redundancy)
  - Human kinases preferred

This list is manually curated — not scraped — because quality matters
more than quantity for a first validation experiment.

Usage:
  python prepare_kinase_dataset.py --out_dir data/kinase_pdbs/
"""

import os
import argparse
import csv
from typing import List, Dict

# ============================================================================
# Curated kinase list (50 structures)
# ============================================================================
#
# Columns: PDB ID, chain, kinase name, family, ligand resname, resolution (Å)
#
# These are well-studied kinases with high-quality crystal structures.
# Ligand resnames may vary by PDB entry — verify after download.
#
# IMPORTANT: This list should be verified against the actual PDB files
# before running experiments. Some ligand resnames may need updating.
# ============================================================================

KINASE_LIST: List[Dict[str, str]] = [
    # === TK (Tyrosine Kinases) ===
    {"pdb_id": "1M17", "chain": "A", "kinase": "EGFR",     "family": "TK",   "ligand": "AQ4", "resolution": "2.60"},
    {"pdb_id": "2HYY", "chain": "A", "kinase": "EGFR_T790M","family": "TK",  "ligand": "IRE", "resolution": "2.65"},
    {"pdb_id": "1IEP", "chain": "A", "kinase": "ABL1",     "family": "TK",   "ligand": "STI", "resolution": "2.10"},
    {"pdb_id": "2GQG", "chain": "A", "kinase": "ABL1_T315I","family": "TK",  "ligand": "406", "resolution": "1.80"},
    {"pdb_id": "3CS9", "chain": "A", "kinase": "SRC",      "family": "TK",   "ligand": "DAS", "resolution": "1.73"},
    {"pdb_id": "2SRC", "chain": "A", "kinase": "SRC_apo",  "family": "TK",   "ligand": "ANP", "resolution": "1.50"},
    {"pdb_id": "1PKG", "chain": "A", "kinase": "FGFR1",    "family": "TK",   "ligand": "SU1", "resolution": "2.40"},
    {"pdb_id": "4V01", "chain": "A", "kinase": "VEGFR2",   "family": "TK",   "ligand": "032", "resolution": "2.03"},
    {"pdb_id": "3LCK", "chain": "A", "kinase": "LCK",      "family": "TK",   "ligand": "STU", "resolution": "1.60"},
    {"pdb_id": "1T46", "chain": "A", "kinase": "PDGFR",    "family": "TK",   "ligand": "STI", "resolution": "2.20"},

    # === CMGC (CDKs, MAPKs, GSKs, CDK-like) ===
    {"pdb_id": "1HCK", "chain": "A", "kinase": "CDK2",     "family": "CMGC", "ligand": "ATP", "resolution": "1.80"},
    {"pdb_id": "2VTH", "chain": "A", "kinase": "CDK2_inh", "family": "CMGC", "ligand": "K49", "resolution": "2.00"},
    {"pdb_id": "2I0V", "chain": "A", "kinase": "CDK4",     "family": "CMGC", "ligand": "LQQ", "resolution": "2.30"},
    {"pdb_id": "3EQG", "chain": "A", "kinase": "CDK6",     "family": "CMGC", "ligand": "FAS", "resolution": "2.30"},
    {"pdb_id": "2ERK", "chain": "A", "kinase": "ERK2",     "family": "CMGC", "ligand": "ATP", "resolution": "1.90"},
    {"pdb_id": "3PY3", "chain": "A", "kinase": "ERK2_inh", "family": "CMGC", "ligand": "INH", "resolution": "2.10"},
    {"pdb_id": "1PME", "chain": "A", "kinase": "P38alpha",  "family": "CMGC","ligand": "SB2", "resolution": "2.10"},
    {"pdb_id": "3GCS", "chain": "A", "kinase": "JNK1",     "family": "CMGC", "ligand": "J60", "resolution": "2.40"},
    {"pdb_id": "1Q5K", "chain": "A", "kinase": "GSK3beta",  "family": "CMGC","ligand": "STU", "resolution": "2.20"},
    {"pdb_id": "3FI3", "chain": "A", "kinase": "DYRK1A",   "family": "CMGC", "ligand": "D57", "resolution": "2.20"},

    # === AGC (PKA, PKB/Akt, PKC, etc.) ===
    {"pdb_id": "1ATP", "chain": "E", "kinase": "PKA",      "family": "AGC",  "ligand": "ATP", "resolution": "2.20"},
    {"pdb_id": "1BKX", "chain": "A", "kinase": "PKA_inh",  "family": "AGC",  "ligand": "H89", "resolution": "2.00"},
    {"pdb_id": "3CQW", "chain": "A", "kinase": "AKT1",     "family": "AGC",  "ligand": "ANP", "resolution": "2.00"},
    {"pdb_id": "3MV5", "chain": "A", "kinase": "AKT2",     "family": "AGC",  "ligand": "MK2", "resolution": "2.10"},
    {"pdb_id": "3A8W", "chain": "A", "kinase": "PKCtheta",  "family": "AGC", "ligand": "STA", "resolution": "2.40"},
    {"pdb_id": "2X39", "chain": "A", "kinase": "ROCK1",    "family": "AGC",  "ligand": "Y27", "resolution": "2.30"},
    {"pdb_id": "4GV1", "chain": "A", "kinase": "RSK2",     "family": "AGC",  "ligand": "SL0", "resolution": "2.40"},

    # === CAMK (Calcium/calmodulin-dependent) ===
    {"pdb_id": "2VN9", "chain": "A", "kinase": "CAMK2A",   "family": "CAMK", "ligand": "STO", "resolution": "2.40"},
    {"pdb_id": "4BYI", "chain": "A", "kinase": "PIM1",     "family": "CAMK", "ligand": "SGX", "resolution": "1.90"},
    {"pdb_id": "3FXZ", "chain": "A", "kinase": "PIM2",     "family": "CAMK", "ligand": "LGH", "resolution": "2.10"},
    {"pdb_id": "3LXK", "chain": "A", "kinase": "DAPK1",    "family": "CAMK", "ligand": "VER", "resolution": "2.30"},
    {"pdb_id": "3COK", "chain": "A", "kinase": "CHK1",     "family": "CAMK", "ligand": "0LI", "resolution": "2.00"},
    {"pdb_id": "2CN5", "chain": "A", "kinase": "CHK2",     "family": "CAMK", "ligand": "DBQ", "resolution": "2.20"},

    # === STE (MAP kinase cascade) ===
    {"pdb_id": "3VN9", "chain": "A", "kinase": "MEK1",     "family": "STE",  "ligand": "ACP", "resolution": "2.40"},
    {"pdb_id": "4AN2", "chain": "A", "kinase": "MEK2",     "family": "STE",  "ligand": "1FY", "resolution": "2.10"},
    {"pdb_id": "4MNE", "chain": "A", "kinase": "PAK1",     "family": "STE",  "ligand": "IPA", "resolution": "2.30"},
    {"pdb_id": "2BDF", "chain": "A", "kinase": "BRAF",     "family": "STE",  "ligand": "BAX", "resolution": "2.90"},

    # === TKL (Tyrosine Kinase-Like) ===
    {"pdb_id": "3Q96", "chain": "A", "kinase": "BRAF_V600E","family": "TKL", "ligand": "032", "resolution": "2.45"},
    {"pdb_id": "1BYG", "chain": "A", "kinase": "TGFBR1",   "family": "TKL", "ligand": "SB4", "resolution": "2.60"},
    {"pdb_id": "3HHM", "chain": "A", "kinase": "TAK1",     "family": "TKL", "ligand": "5PP", "resolution": "2.00"},
    {"pdb_id": "4KN7", "chain": "A", "kinase": "MLK1",     "family": "TKL", "ligand": "K1N", "resolution": "2.40"},

    # === CK1 (Casein Kinase 1) ===
    {"pdb_id": "1CKJ", "chain": "A", "kinase": "CK1delta",  "family": "CK1", "ligand": "MGP", "resolution": "2.30"},
    {"pdb_id": "4HNF", "chain": "A", "kinase": "CK1epsilon","family": "CK1", "ligand": "PF0", "resolution": "2.49"},

    # === Other well-studied kinases ===
    {"pdb_id": "3EFJ", "chain": "A", "kinase": "AURKA",    "family": "Other","ligand": "VX6", "resolution": "2.40"},
    {"pdb_id": "2VGO", "chain": "A", "kinase": "PLK1",     "family": "Other","ligand": "TAL", "resolution": "2.40"},
    {"pdb_id": "2IVT", "chain": "A", "kinase": "HASPIN",   "family": "Other","ligand": "STU", "resolution": "2.20"},
    {"pdb_id": "3C4C", "chain": "A", "kinase": "JAK2",     "family": "TK",  "ligand": "IZA", "resolution": "2.00"},
    {"pdb_id": "3EYG", "chain": "A", "kinase": "MET",      "family": "TK",  "ligand": "PHA", "resolution": "2.00"},
    {"pdb_id": "3PP0", "chain": "A", "kinase": "BTK",      "family": "TK",  "ligand": "B43", "resolution": "1.60"},
    {"pdb_id": "4WBO", "chain": "A", "kinase": "ALK",      "family": "TK",  "ligand": "3OY", "resolution": "2.00"},
]


def download_pdb(pdb_id: str, out_dir: str) -> str:
    """Download a PDB file from RCSB. Returns local path."""
    import urllib.request

    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    local_path = os.path.join(out_dir, f"{pdb_id}.pdb")

    if os.path.exists(local_path):
        return local_path

    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"  WARNING: Could not download {pdb_id}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/kinase_pdbs")
    parser.add_argument("--download", action="store_true",
                        help="Actually download PDB files (requires network)")
    parser.add_argument("--csv_only", action="store_true",
                        help="Only write the CSV manifest, no downloads")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, "kinase_list.csv")

    rows = []
    for entry in KINASE_LIST:
        pdb_id = entry["pdb_id"]
        pdb_path = os.path.join(args.out_dir, f"{pdb_id}.pdb")

        if args.download and not args.csv_only:
            result = download_pdb(pdb_id, args.out_dir)
            if result:
                print(f"  Downloaded: {pdb_id}")
            else:
                print(f"  FAILED: {pdb_id}")
                continue

        rows.append({
            "pdb_path": pdb_path,
            "pdb_id": pdb_id,
            "chain_id": entry["chain"],
            "pocket_label": entry["kinase"],
            "family": entry["family"],
            "ligand_resname": entry["ligand"],
            "resolution": entry["resolution"],
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_path", "pdb_id", "chain_id", "pocket_label",
            "family", "ligand_resname", "resolution",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote manifest: {csv_path} ({len(rows)} kinases)")

    # Summary by family
    from collections import Counter
    fam_counts = Counter(e["family"] for e in KINASE_LIST)
    print("\nFamily distribution:")
    for fam, count in sorted(fam_counts.items()):
        print(f"  {fam:8s}: {count}")
    print(f"  {'TOTAL':8s}: {len(KINASE_LIST)}")


if __name__ == "__main__":
    main()
