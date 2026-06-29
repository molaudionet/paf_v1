#!/usr/bin/env python3
"""
download_pdbs.py — Bulk download PDB files from RCSB.

Usage:
  python download_pdbs.py --manifest data/cross_family_manifest.csv --out_dir data/pdbs/
  python download_pdbs.py --pdb_ids 1ATP 2HYY 3CS9 --out_dir data/pdbs/
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, out_dir: str, overwrite: bool = False) -> bool:
    """Download a single PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    out_path = os.path.join(out_dir, f"{pdb_id.lower()}.pdb")

    if not overwrite and os.path.exists(out_path):
        return True  # Already exists

    url = RCSB_DOWNLOAD.format(pdb_id=pdb_id)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "w") as f:
                f.write(resp.text)
            return True
        else:
            return False
    except Exception:
        return False


def download_batch(
    pdb_ids: list,
    out_dir: str,
    overwrite: bool = False,
    max_workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Download PDB files in parallel.

    Returns dict with 'success', 'failed', 'skipped' counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    unique_ids = list(dict.fromkeys(pdb_ids))

    success = 0
    failed = 0
    skipped = 0
    failed_ids = []

    if verbose:
        print(f"\n  Downloading {len(unique_ids)} PDB files to {out_dir}")
        print(f"  Using {max_workers} parallel workers")
        print(f"  Overwrite: {overwrite}\n")

    def _download(pid):
        out_path = os.path.join(out_dir, f"{pid.lower()}.pdb")
        if not overwrite and os.path.exists(out_path):
            return pid, "skipped"
        ok = download_pdb(pid, out_dir, overwrite)
        return pid, "success" if ok else "failed"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, pid): pid for pid in unique_ids}

        for i, future in enumerate(as_completed(futures)):
            pid, status = future.result()

            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_ids.append(pid)

            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(unique_ids)} "
                      f"(ok={success}, skip={skipped}, fail={failed})")

    if verbose:
        print(f"\n  Done: {success} downloaded, {skipped} already existed, "
              f"{failed} failed")
        if failed_ids and len(failed_ids) <= 20:
            print(f"  Failed IDs: {', '.join(failed_ids)}")

    return {
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "failed_ids": failed_ids,
    }


def main():
    ap = argparse.ArgumentParser(description="Download PDB files from RCSB")
    ap.add_argument("--manifest", help="CSV manifest with pdb_id column")
    ap.add_argument("--pdb_ids", nargs="+", help="List of PDB IDs")
    ap.add_argument("--out_dir", default="data/pdbs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pdb_ids = []

    if args.manifest:
        with open(args.manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("pdb_id", "").strip()
                if pid and len(pid) == 4:
                    pdb_ids.append(pid)
        print(f"  Loaded {len(pdb_ids)} PDB IDs from {args.manifest}")

    if args.pdb_ids:
        pdb_ids.extend(args.pdb_ids)

    if not pdb_ids:
        print("Error: provide --manifest or --pdb_ids")
        sys.exit(1)

    download_batch(pdb_ids, args.out_dir, args.overwrite, args.workers)


if __name__ == "__main__":
    main()
