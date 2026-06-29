#!/usr/bin/env python3
"""
curate_datasets.py — Build curated datasets for large-scale PAF experiments.

Three data sources:
  1. PDBbind refined set — ~5,000 high-quality protein-ligand complexes with family labels
  2. KLIFS kinase database — ~6,000 kinase structures with DFG/αC annotations
  3. RCSB PDB advanced search — custom queries for diverse protein families

Outputs CSV manifests that the downloader and pipeline consume.

Usage:
  python curate_datasets.py --task cross_family --out data/cross_family_manifest.csv
  python curate_datasets.py --task kinase_klifs --out data/kinase_manifest.csv
  python curate_datasets.py --task all --out_dir data/
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


# ── RCSB PDB Search API ─────────────────────────────────────────────────────

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

def rcsb_search(query_json: dict, max_results: int = 500) -> List[str]:
    """Execute RCSB search API query, return list of PDB IDs."""
    query_json["request_options"] = {
        "return_all_hits": False,
        "results_content_type": ["experimental"],
        "paginate": {"start": 0, "rows": max_results},
    }
    query_json["return_type"] = "entry"

    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query_json, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [r["identifier"] for r in data.get("result_set", [])]
    except Exception as e:
        print(f"  RCSB search error: {e}")
        return []


def build_family_query(enzyme_class: str, has_ligand: bool = True,
                       resolution_max: float = 3.0) -> dict:
    """Build RCSB query for a protein family with ligand + resolution filter."""
    nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity.pdbx_description",
                "operator": "contains_phrase",
                "value": enzyme_class,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": resolution_max,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "X-RAY DIFFRACTION",
            },
        },
    ]

    if has_ligand:
        nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater",
                "value": 0,
            },
        })

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
    }


# ── Cross-Family Dataset (Expanded) ─────────────────────────────────────────

PROTEIN_FAMILIES = {
    # Original 4 families (expanded)
    "kinase": {
        "search_terms": ["protein kinase"],
        "target_n": 300,
    },
    "serine_protease": {
        "search_terms": ["serine protease", "trypsin", "chymotrypsin", "elastase", "thrombin"],
        "target_n": 200,
    },
    "metalloprotease": {
        "search_terms": ["metalloprotease", "metalloproteinase", "MMP"],
        "target_n": 200,
    },
    "nuclear_receptor": {
        "search_terms": ["nuclear receptor", "estrogen receptor", "androgen receptor",
                         "glucocorticoid receptor", "PPAR"],
        "target_n": 200,
    },
    # New families for expanded validation
    "phosphodiesterase": {
        "search_terms": ["phosphodiesterase"],
        "target_n": 100,
    },
    "proteasome": {
        "search_terms": ["proteasome"],
        "target_n": 80,
    },
    "carbonic_anhydrase": {
        "search_terms": ["carbonic anhydrase"],
        "target_n": 100,
    },
    "hsp90": {
        "search_terms": ["heat shock protein 90", "HSP90"],
        "target_n": 100,
    },
    "cyclooxygenase": {
        "search_terms": ["cyclooxygenase", "COX-2", "prostaglandin"],
        "target_n": 80,
    },
    "histone_deacetylase": {
        "search_terms": ["histone deacetylase", "HDAC"],
        "target_n": 80,
    },
    "gpcr": {
        "search_terms": ["G protein-coupled receptor"],
        "target_n": 150,
    },
    "aspartyl_protease": {
        "search_terms": ["aspartyl protease", "HIV protease", "renin", "BACE"],
        "target_n": 150,
    },
    "phosphatase": {
        "search_terms": ["protein phosphatase", "PTP"],
        "target_n": 100,
    },
    "bromodomain": {
        "search_terms": ["bromodomain"],
        "target_n": 100,
    },
    "dihydrofolate_reductase": {
        "search_terms": ["dihydrofolate reductase", "DHFR"],
        "target_n": 80,
    },
}


def curate_cross_family(out_path: str, resolution: float = 2.5) -> int:
    """
    Query RCSB for diverse protein families with co-crystallized ligands.
    Writes CSV manifest.
    """
    print("\n" + "="*70)
    print("  Curating Cross-Family Dataset")
    print("="*70)

    all_entries = []
    seen_pdb_ids = set()

    for family, config in PROTEIN_FAMILIES.items():
        family_ids = []

        for term in config["search_terms"]:
            query = build_family_query(term, has_ligand=True,
                                       resolution_max=resolution)
            ids = rcsb_search(query, max_results=config["target_n"])
            family_ids.extend(ids)
            time.sleep(0.5)  # Rate limit

        # Deduplicate within family
        family_ids = list(dict.fromkeys(family_ids))
        # Remove cross-family duplicates
        new_ids = [pid for pid in family_ids if pid not in seen_pdb_ids]
        new_ids = new_ids[:config["target_n"]]
        seen_pdb_ids.update(new_ids)

        for pid in new_ids:
            all_entries.append({
                "pdb_id": pid,
                "family": family,
                "subfamily": "",
                "chain": "A",
                "ligand_resname": "",
                "dfg_state": "",
            })

        print(f"  {family:30s}: {len(new_ids):4d} structures")

    # Write CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pdb_id", "family", "subfamily", "chain", "ligand_resname", "dfg_state"
        ])
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\n  Total: {len(all_entries)} structures → {out_path}")
    return len(all_entries)


# ── KLIFS Kinase Dataset ─────────────────────────────────────────────────────

KLIFS_API = "https://klifs.net/api"


def curate_kinase_klifs(out_path: str) -> int:
    """
    Pull kinase structures from KLIFS with DFG/αC annotations.
    KLIFS provides pre-curated binding pocket definitions for kinases.
    """
    print("\n" + "="*70)
    print("  Curating KLIFS Kinase Dataset")
    print("="*70)

    all_entries = []

    # KLIFS kinase families
    try:
        resp = requests.get(f"{KLIFS_API}/kinase_names", timeout=30)
        resp.raise_for_status()
        kinase_list = resp.json()
        print(f"  Found {len(kinase_list)} kinases in KLIFS")
    except Exception as e:
        print(f"  KLIFS API error: {e}")
        print("  Falling back to manual kinase list...")
        kinase_list = None

    if kinase_list:
        # Get structures for each kinase
        for kin in kinase_list[:200]:  # Process top 200 kinases
            kinase_id = kin.get("kinase_ID", kin.get("id"))
            kinase_name = kin.get("name", "")
            family = kin.get("family", kin.get("group", ""))
            subfamily = kin.get("subfamily", kin.get("group", ""))

            try:
                resp = requests.get(
                    f"{KLIFS_API}/structures_list",
                    params={"kinase_ID": kinase_id},
                    timeout=15,
                )
                resp.raise_for_status()
                structures = resp.json()
            except Exception:
                continue

            for s in structures[:10]:  # Max 10 per kinase
                pdb_id = s.get("pdb", "")
                if not pdb_id or len(pdb_id) != 4:
                    continue

                dfg = s.get("DFG", s.get("dfg", ""))
                ac_helix = s.get("aC_helix", s.get("ac_helix", ""))
                chain = s.get("chain", "A")
                ligand = s.get("ligand", "")

                all_entries.append({
                    "pdb_id": pdb_id,
                    "family": family or "kinase",
                    "subfamily": subfamily or kinase_name,
                    "chain": chain,
                    "ligand_resname": ligand,
                    "dfg_state": dfg,
                    "ac_helix": ac_helix,
                    "kinase_name": kinase_name,
                })

            time.sleep(0.3)

    else:
        # Fallback: query RCSB for well-known kinase subfamilies
        kinase_subfamilies = {
            "TK": ["EGFR kinase", "ABL kinase", "SRC kinase", "VEGFR kinase",
                    "FGFR kinase", "MET kinase", "KIT kinase", "JAK kinase"],
            "TKL": ["BRAF kinase", "RAF kinase", "ALK kinase"],
            "CMGC": ["CDK2 kinase", "CDK4 kinase", "ERK kinase", "GSK3 kinase",
                      "CK2 kinase", "DYRK kinase"],
            "AGC": ["PKA kinase", "PKC kinase", "AKT kinase", "RSK kinase"],
            "CAMK": ["CAMK2 kinase", "DAPK kinase", "CHK1 kinase", "CHK2 kinase"],
            "STE": ["MEK kinase", "MAP kinase", "PAK kinase"],
            "CK1": ["CK1 kinase", "casein kinase 1"],
        }

        for subfamily, terms in kinase_subfamilies.items():
            for term in terms:
                query = build_family_query(term, has_ligand=True, resolution_max=2.5)
                ids = rcsb_search(query, max_results=50)
                for pid in ids:
                    all_entries.append({
                        "pdb_id": pid,
                        "family": "kinase",
                        "subfamily": subfamily,
                        "chain": "A",
                        "ligand_resname": "",
                        "dfg_state": "",
                        "ac_helix": "",
                        "kinase_name": term.replace(" kinase", ""),
                    })
                time.sleep(0.5)

    # Deduplicate by PDB ID
    seen = set()
    unique = []
    for e in all_entries:
        if e["pdb_id"] not in seen:
            seen.add(e["pdb_id"])
            unique.append(e)
    all_entries = unique

    print(f"  Total unique kinase structures: {len(all_entries)}")

    # Count by subfamily
    from collections import Counter
    subfam_counts = Counter(e["subfamily"] for e in all_entries)
    for sf, n in subfam_counts.most_common(20):
        print(f"    {sf:20s}: {n:4d}")

    # Write CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["pdb_id", "family", "subfamily", "chain",
                  "ligand_resname", "dfg_state", "ac_helix", "kinase_name"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in all_entries:
            writer.writerow({k: e.get(k, "") for k in fieldnames})

    print(f"  Wrote: {out_path}")
    return len(all_entries)


# ── PDBbind Refined Set ──────────────────────────────────────────────────────

def curate_pdbbind_index(index_path: str, out_path: str) -> int:
    """
    Parse PDBbind refined set index file.
    You need to download PDBbind from http://www.pdbbind.org.cn/

    The index file (INDEX_refined_data.2020) has format:
    PDB_code  resolution  release_year  -logKd/Ki  Kd/Ki  reference  ligand_name
    """
    print("\n" + "="*70)
    print("  Parsing PDBbind Index")
    print("="*70)

    if not os.path.exists(index_path):
        print(f"  Index file not found: {index_path}")
        print("  Download PDBbind from http://www.pdbbind.org.cn/")
        print("  Then point --pdbbind_index to the INDEX file")
        return 0

    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0]
            if len(pdb_id) != 4:
                continue
            try:
                resolution = float(parts[1])
                affinity = float(parts[3])
            except ValueError:
                continue

            entries.append({
                "pdb_id": pdb_id,
                "family": "pdbbind",
                "subfamily": "",
                "chain": "A",
                "ligand_resname": "",
                "dfg_state": "",
                "resolution": resolution,
                "affinity_pKd": affinity,
            })

    print(f"  Found {len(entries)} entries in PDBbind index")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["pdb_id", "family", "subfamily", "chain",
                  "ligand_resname", "dfg_state", "resolution", "affinity_pKd"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    print(f"  Wrote: {out_path}")
    return len(entries)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Curate datasets for PAF experiments")
    ap.add_argument("--task", choices=["cross_family", "kinase_klifs",
                                        "pdbbind", "all"],
                    default="all")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--out", help="Output CSV (for single-task mode)")
    ap.add_argument("--pdbbind_index", help="Path to PDBbind INDEX file")
    ap.add_argument("--resolution", type=float, default=2.5,
                    help="Max resolution cutoff (Å)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0

    if args.task in ("cross_family", "all"):
        out = args.out or os.path.join(args.out_dir, "cross_family_manifest.csv")
        total += curate_cross_family(out, resolution=args.resolution)

    if args.task in ("kinase_klifs", "all"):
        out = args.out if args.task != "all" else os.path.join(args.out_dir, "kinase_manifest.csv")
        total += curate_kinase_klifs(out)

    if args.task in ("pdbbind", "all") and args.pdbbind_index:
        out = args.out if args.task != "all" else os.path.join(args.out_dir, "pdbbind_manifest.csv")
        total += curate_pdbbind_index(args.pdbbind_index, out)

    print(f"\n{'='*70}")
    print(f"  Grand total: {total} structures curated")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
