#!/usr/bin/env python3
"""
curate_cross_family_dataset.py

Curate a multi-family protein pocket dataset for PAF cross-family
generalization testing.

Families:
  1. Kinases        – reuse existing + expand (TK, CMGC, AGC, STE, CK1)
  2. Serine proteases – trypsin-like, subtilisin-like, chymotrypsin-like
  3. Metalloproteases – MMPs, ADAMs, thermolysin-like
  4. Nuclear receptors – ER, AR, PPAR, RAR, RXR

For each entry we need:
  - A ligand-bound PDB structure (co-crystal with small molecule)
  - Chain ID of the protein
  - Ligand residue name (3-letter code)
  - Subfamily label (for within-family testing)
  - Family label (for between-family testing)

Usage:
  python curate_cross_family_dataset.py --out data/cross_family/ [--download]

Without --download: writes cross_family_list.csv with PDB IDs only
With    --download: also fetches PDB files from RCSB
"""

from __future__ import annotations
import os
import csv
import json
import time
import argparse
from typing import List, Dict

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ============================================================
# Curated dataset: manually selected high-quality co-crystals
# ============================================================
# Selection criteria:
#   - Resolution ≤ 2.5 Å
#   - Co-crystallized with a drug-like small molecule
#   - Human or well-studied model organism
#   - Diverse subfamily coverage within each family
#   - One structure per protein where possible (avoid bias)

DATASET: List[Dict] = [
    # -----------------------------------------------------------
    # KINASES  (subfamily = kinase group)
    # -----------------------------------------------------------
    # TK group
    {"pdb_id": "1M17", "chain_id": "A", "ligand_resname": "AQ4", "subfamily": "EGFR",      "family": "kinase"},
    {"pdb_id": "3ETA", "chain_id": "A", "ligand_resname": "AEE", "subfamily": "ABL",       "family": "kinase"},
    {"pdb_id": "2HYY", "chain_id": "A", "ligand_resname": "BAX", "subfamily": "VEGFR2",    "family": "kinase"},
    {"pdb_id": "3GQL", "chain_id": "A", "ligand_resname": "J15", "subfamily": "FGFR1",     "family": "kinase"},
    {"pdb_id": "1PKG", "chain_id": "A", "ligand_resname": "STI", "subfamily": "KIT",       "family": "kinase"},
    {"pdb_id": "2SRC", "chain_id": "A", "ligand_resname": "ANP", "subfamily": "SRC",       "family": "kinase"},
    {"pdb_id": "1YWN", "chain_id": "A", "ligand_resname": "SU1", "subfamily": "PDGFR",     "family": "kinase"},

    # CMGC group
    {"pdb_id": "1H1S", "chain_id": "A", "ligand_resname": "STU", "subfamily": "CDK2",      "family": "kinase"},
    {"pdb_id": "2VTA", "chain_id": "A", "ligand_resname": "3JY", "subfamily": "CDK4",      "family": "kinase"},
    {"pdb_id": "3ELJ", "chain_id": "A", "ligand_resname": "ID8", "subfamily": "GSK3B",     "family": "kinase"},
    {"pdb_id": "1PME", "chain_id": "A", "ligand_resname": "SB4", "subfamily": "p38alpha",  "family": "kinase"},
    {"pdb_id": "2ERK", "chain_id": "A", "ligand_resname": "FR6", "subfamily": "ERK2",      "family": "kinase"},
    {"pdb_id": "3NPC", "chain_id": "A", "ligand_resname": "N30", "subfamily": "CLK1",      "family": "kinase"},

    # AGC group
    {"pdb_id": "1O6K", "chain_id": "A", "ligand_resname": "MR4", "subfamily": "PKA",       "family": "kinase"},
    {"pdb_id": "3CQW", "chain_id": "A", "ligand_resname": "38E", "subfamily": "AKT1",      "family": "kinase"},
    {"pdb_id": "2F9G", "chain_id": "A", "ligand_resname": "BOG", "subfamily": "PKCtheta",  "family": "kinase"},
    {"pdb_id": "3A7H", "chain_id": "A", "ligand_resname": "BI2", "subfamily": "ROCK1",     "family": "kinase"},

    # STE group
    {"pdb_id": "3WZE", "chain_id": "A", "ligand_resname": "4WZ", "subfamily": "MEK1",      "family": "kinase"},
    {"pdb_id": "4MNE", "chain_id": "A", "ligand_resname": "20S", "subfamily": "PAK1",      "family": "kinase"},

    # CK1 / CAMK / other
    {"pdb_id": "1CKJ", "chain_id": "A", "ligand_resname": "B96", "subfamily": "CK1",       "family": "kinase"},
    {"pdb_id": "2JAM", "chain_id": "A", "ligand_resname": "STU", "subfamily": "DAPK1",     "family": "kinase"},
    {"pdb_id": "3COK", "chain_id": "A", "ligand_resname": "LYB", "subfamily": "BRAF",      "family": "kinase"},
    {"pdb_id": "4JVS", "chain_id": "A", "ligand_resname": "0WM", "subfamily": "ALK",       "family": "kinase"},
    {"pdb_id": "3LCK", "chain_id": "A", "ligand_resname": "STU", "subfamily": "LCK",       "family": "kinase"},

    # -----------------------------------------------------------
    # SERINE PROTEASES  (subfamily = protease name)
    # -----------------------------------------------------------
    # Trypsin-like
    {"pdb_id": "1C5T", "chain_id": "A", "ligand_resname": "BEN", "subfamily": "trypsin",        "family": "serine_protease"},
    {"pdb_id": "1FPC", "chain_id": "A", "ligand_resname": "BZA", "subfamily": "FXa",            "family": "serine_protease"},
    {"pdb_id": "1DWC", "chain_id": "H", "ligand_resname": "AER", "subfamily": "thrombin",       "family": "serine_protease"},
    {"pdb_id": "2ZDK", "chain_id": "A", "ligand_resname": "D4C", "subfamily": "FVIIa",          "family": "serine_protease"},
    {"pdb_id": "1LQE", "chain_id": "B", "ligand_resname": "7HP", "subfamily": "plasmin",        "family": "serine_protease"},
    {"pdb_id": "1GVK", "chain_id": "A", "ligand_resname": "NFP", "subfamily": "FXIa",           "family": "serine_protease"},

    # Chymotrypsin-like
    {"pdb_id": "4CHA", "chain_id": "A", "ligand_resname": "FMF", "subfamily": "chymotrypsin",   "family": "serine_protease"},
    {"pdb_id": "1ELT", "chain_id": "A", "ligand_resname": "ACT", "subfamily": "elastase",       "family": "serine_protease"},
    {"pdb_id": "1HJ9", "chain_id": "A", "ligand_resname": "GS2", "subfamily": "granzyme_B",     "family": "serine_protease"},

    # Subtilisin-like / other
    {"pdb_id": "1SCA", "chain_id": "A", "ligand_resname": "DFP", "subfamily": "subtilisin",     "family": "serine_protease"},
    {"pdb_id": "1SGT", "chain_id": "A", "ligand_resname": "IPH", "subfamily": "proteinase_K",   "family": "serine_protease"},
    {"pdb_id": "3TGI", "chain_id": "A", "ligand_resname": "AIP", "subfamily": "trypsinogen",    "family": "serine_protease"},

    # Kallikreins / uPA
    {"pdb_id": "2QXI", "chain_id": "A", "ligand_resname": "ROC", "subfamily": "KLK5",           "family": "serine_protease"},
    {"pdb_id": "1SQA", "chain_id": "A", "ligand_resname": "EPX", "subfamily": "uPA",            "family": "serine_protease"},
    {"pdb_id": "1GI1", "chain_id": "A", "ligand_resname": "TOC", "subfamily": "kallikrein",     "family": "serine_protease"},

    # Additional serine proteases
    {"pdb_id": "3TJQ", "chain_id": "C", "ligand_resname": "10E", "subfamily": "HCV_protease",   "family": "serine_protease"},
    {"pdb_id": "1A0H", "chain_id": "A", "ligand_resname": "TFA", "subfamily": "DPP4",           "family": "serine_protease"},
    {"pdb_id": "1MTS", "chain_id": "A", "ligand_resname": "APC", "subfamily": "matriptase",     "family": "serine_protease"},

    # -----------------------------------------------------------
    # METALLOPROTEASES  (subfamily = MMP/ADAM type)
    # -----------------------------------------------------------
    {"pdb_id": "1CGL", "chain_id": "A", "ligand_resname": "HAP", "subfamily": "MMP1",      "family": "metalloprotease"},
    {"pdb_id": "1QIB", "chain_id": "A", "ligand_resname": "BBS", "subfamily": "MMP2",      "family": "metalloprotease"},
    {"pdb_id": "1B3D", "chain_id": "A", "ligand_resname": "LHS", "subfamily": "MMP3",      "family": "metalloprotease"},
    {"pdb_id": "1MMP", "chain_id": "A", "ligand_resname": "FHB", "subfamily": "MMP7",      "family": "metalloprotease"},
    {"pdb_id": "1MMB", "chain_id": "A", "ligand_resname": "BHH", "subfamily": "MMP8",      "family": "metalloprotease"},
    {"pdb_id": "1GKC", "chain_id": "A", "ligand_resname": "MAR", "subfamily": "MMP9",      "family": "metalloprotease"},
    {"pdb_id": "1Y93", "chain_id": "A", "ligand_resname": "HCI", "subfamily": "MMP12",     "family": "metalloprotease"},
    {"pdb_id": "1RMZ", "chain_id": "A", "ligand_resname": "G2N", "subfamily": "MMP13",     "family": "metalloprotease"},
    {"pdb_id": "3AYU", "chain_id": "A", "ligand_resname": "GIY", "subfamily": "MMP14",     "family": "metalloprotease"},

    # ADAM / ACE / other zinc metalloproteases
    {"pdb_id": "1O86", "chain_id": "A", "ligand_resname": "IH1", "subfamily": "ACE",       "family": "metalloprotease"},
    {"pdb_id": "1R1H", "chain_id": "A", "ligand_resname": "LIS", "subfamily": "ACE2",      "family": "metalloprotease"},
    {"pdb_id": "4IGE", "chain_id": "A", "ligand_resname": "2FE", "subfamily": "ADAM17",    "family": "metalloprotease"},
    {"pdb_id": "1THL", "chain_id": "A", "ligand_resname": "ZAF", "subfamily": "thermolysin","family": "metalloprotease"},
    {"pdb_id": "1EZM", "chain_id": "A", "ligand_resname": "NEP", "subfamily": "neprilysin", "family": "metalloprotease"},

    # Additional metalloproteases
    {"pdb_id": "1SLN", "chain_id": "A", "ligand_resname": "PPL", "subfamily": "thermolysin_B","family": "metalloprotease"},
    {"pdb_id": "1HFS", "chain_id": "A", "ligand_resname": "AHA", "subfamily": "MMP3_B",    "family": "metalloprotease"},
    {"pdb_id": "1BQO", "chain_id": "A", "ligand_resname": "AAN", "subfamily": "MMP1_B",    "family": "metalloprotease"},
    {"pdb_id": "1JAP", "chain_id": "A", "ligand_resname": "RO4", "subfamily": "MMP8_B",    "family": "metalloprotease"},

    # -----------------------------------------------------------
    # NUCLEAR RECEPTORS  (subfamily = receptor type)
    # -----------------------------------------------------------
    # Estrogen receptors
    {"pdb_id": "1ERE", "chain_id": "A", "ligand_resname": "EST", "subfamily": "ERalpha",   "family": "nuclear_receptor"},
    {"pdb_id": "3ERT", "chain_id": "A", "ligand_resname": "OHT", "subfamily": "ERalpha_B", "family": "nuclear_receptor"},
    {"pdb_id": "1QKN", "chain_id": "A", "ligand_resname": "GEN", "subfamily": "ERbeta",    "family": "nuclear_receptor"},
    {"pdb_id": "1X7R", "chain_id": "A", "ligand_resname": "EST", "subfamily": "ERbeta_B",  "family": "nuclear_receptor"},

    # Androgen receptor
    {"pdb_id": "1E3G", "chain_id": "A", "ligand_resname": "DHT", "subfamily": "AR",        "family": "nuclear_receptor"},
    {"pdb_id": "2AXA", "chain_id": "A", "ligand_resname": "TES", "subfamily": "AR_B",      "family": "nuclear_receptor"},

    # PPARs
    {"pdb_id": "2PRG", "chain_id": "A", "ligand_resname": "RGZ", "subfamily": "PPARgamma", "family": "nuclear_receptor"},
    {"pdb_id": "1I7G", "chain_id": "A", "ligand_resname": "544", "subfamily": "PPARgamma_B","family": "nuclear_receptor"},
    {"pdb_id": "1K7L", "chain_id": "A", "ligand_resname": "GW0", "subfamily": "PPARdelta", "family": "nuclear_receptor"},
    {"pdb_id": "1KKQ", "chain_id": "A", "ligand_resname": "GW6", "subfamily": "PPARalpha", "family": "nuclear_receptor"},

    # RXR / RAR
    {"pdb_id": "1FBY", "chain_id": "A", "ligand_resname": "REA", "subfamily": "RXRalpha",  "family": "nuclear_receptor"},
    {"pdb_id": "3KMR", "chain_id": "A", "ligand_resname": "9RA", "subfamily": "RARgamma",  "family": "nuclear_receptor"},

    # Glucocorticoid / mineralocorticoid
    {"pdb_id": "1M2Z", "chain_id": "A", "ligand_resname": "DEX", "subfamily": "GR",        "family": "nuclear_receptor"},
    {"pdb_id": "2OAX", "chain_id": "A", "ligand_resname": "DOC", "subfamily": "MR",        "family": "nuclear_receptor"},

    # Progesterone / VDR / TR
    {"pdb_id": "1A28", "chain_id": "A", "ligand_resname": "STR", "subfamily": "PR",        "family": "nuclear_receptor"},
    {"pdb_id": "1DB1", "chain_id": "A", "ligand_resname": "MC1", "subfamily": "VDR",       "family": "nuclear_receptor"},
    {"pdb_id": "1NAV", "chain_id": "A", "ligand_resname": "IH5", "subfamily": "TRbeta",    "family": "nuclear_receptor"},

    # LXR / FXR
    {"pdb_id": "1PQC", "chain_id": "A", "ligand_resname": "EOH", "subfamily": "LXRbeta",   "family": "nuclear_receptor"},
    {"pdb_id": "1OSV", "chain_id": "A", "ligand_resname": "FXR", "subfamily": "FXR",       "family": "nuclear_receptor"},
]


def summarize_dataset():
    """Print dataset summary."""
    from collections import Counter
    fam_counts = Counter(d["family"] for d in DATASET)
    sub_counts = Counter((d["family"], d["subfamily"]) for d in DATASET)

    print("=" * 60)
    print("CROSS-FAMILY DATASET SUMMARY")
    print("=" * 60)
    print(f"Total structures: {len(DATASET)}")
    print()
    for fam in sorted(fam_counts.keys()):
        subs = sorted(set(d["subfamily"] for d in DATASET if d["family"] == fam))
        print(f"  {fam:25s}: {fam_counts[fam]:3d} structures, {len(subs)} subfamilies")
        for s in subs:
            n = sum(1 for d in DATASET if d["subfamily"] == s)
            print(f"    {s:25s}: {n}")
    print("=" * 60)


def download_pdb(pdb_id: str, out_dir: str) -> str:
    """Download PDB file from RCSB. Returns local path."""
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    pid = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    local = os.path.join(out_dir, f"{pid}.pdb")

    if os.path.exists(local) and os.path.getsize(local) > 100:
        return local

    print(f"  Downloading {pid} ...", end=" ", flush=True)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    with open(local, "w") as f:
        f.write(r.text)
    print(f"OK ({os.path.getsize(local)} bytes)")
    time.sleep(0.3)  # be polite to RCSB
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cross_family/", help="output directory")
    ap.add_argument("--download", action="store_true",
                    help="actually download PDB files from RCSB")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pdb_dir = os.path.join(args.out, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    summarize_dataset()

    rows = []
    failed = []

    for entry in DATASET:
        pdb_id = entry["pdb_id"].upper()

        if args.download:
            try:
                local_path = download_pdb(pdb_id, pdb_dir)
            except Exception as e:
                print(f"  [FAIL] {pdb_id}: {e}")
                failed.append({"pdb_id": pdb_id, "error": str(e)})
                continue
        else:
            local_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

        rows.append({
            "pdb_id":         pdb_id,
            "pdb_path":       local_path,
            "chain_id":       entry["chain_id"],
            "ligand_resname": entry["ligand_resname"],
            "subfamily":      entry["subfamily"],
            "family":         entry["family"],
            "pocket_label":   f"{pdb_id}_{entry['subfamily']}",
        })

    # Write CSV
    csv_path = os.path.join(args.out, "cross_family_list.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_id", "pdb_path", "chain_id", "ligand_resname",
            "subfamily", "family", "pocket_label",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} entries to {csv_path}")

    if failed:
        fail_path = os.path.join(args.out, "download_failures.csv")
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pdb_id", "error"])
            writer.writeheader()
            writer.writerows(failed)
        print(f"Download failures: {len(failed)} (see {fail_path})")

    # Write summary JSON
    summary = {
        "total": len(rows),
        "families": {},
        "failed": len(failed),
    }
    for entry in rows:
        fam = entry["family"]
        if fam not in summary["families"]:
            summary["families"][fam] = {"count": 0, "subfamilies": set()}
        summary["families"][fam]["count"] += 1
        summary["families"][fam]["subfamilies"].add(entry["subfamily"])
    # convert sets to lists for JSON
    for fam in summary["families"]:
        summary["families"][fam]["subfamilies"] = sorted(summary["families"][fam]["subfamilies"])

    with open(os.path.join(args.out, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- NEXT STEPS ---")
    if not args.download:
        print(f"1. Run again with --download to fetch PDB files from RCSB")
    print(f"2. Run the cross-family PAF experiment:")
    print(f"   python run_cross_family_paf.py \\")
    print(f"     --csv {csv_path} \\")
    print(f"     --out results/cross_family/ \\")
    print(f"     --radius 10 --gamma_fm 0.15 --sigma_t 0.04")


if __name__ == "__main__":
    main()
