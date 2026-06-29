#!/usr/bin/env bash
#
# run_all.sh — Master script for PAF scale-up pipeline (v2).
#
# Prerequisites:
#   pip install numpy requests biopython scikit-learn
#   paf_core_v1.py must be in this directory (your real PAF encoder)
#
# Usage:
#   bash run_all.sh              # Run everything
#   bash run_all.sh --step 2     # Start from step 2 (download)
#   bash run_all.sh --step 3     # Start from step 3 (experiments only)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
PDB_DIR="${DATA_DIR}/pdbs"
RESULTS_DIR="${SCRIPT_DIR}/results"

START_STEP=${1:-1}
if [ "$1" = "--step" ]; then
    START_STEP=$2
fi

echo "============================================================"
echo "  PAF Scale-Up Pipeline v2 (real encoder)"
echo "  Sound of Molecules LLC"
echo "============================================================"
echo ""
echo "  Data dir:    ${DATA_DIR}"
echo "  PDB dir:     ${PDB_DIR}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Start step:  ${START_STEP}"
echo ""

# Check paf_core_v1.py exists
if [ ! -f "${SCRIPT_DIR}/paf_core_v1.py" ]; then
    echo "ERROR: paf_core_v1.py not found in ${SCRIPT_DIR}"
    echo "Copy your real PAF encoder here before running."
    exit 1
fi

mkdir -p "${DATA_DIR}" "${PDB_DIR}" "${RESULTS_DIR}"

# ── Step 1: Curate Datasets ──
if [ "${START_STEP}" -le 1 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1: Curating Datasets"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPT_DIR}/curate_datasets.py" \
        --task all \
        --out_dir "${DATA_DIR}" \
        --resolution 2.5
    echo ""
fi

# ── Step 2: Download PDB Files ──
if [ "${START_STEP}" -le 2 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2: Downloading PDB Files"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for manifest in "${DATA_DIR}"/*.csv; do
        if [ -f "$manifest" ]; then
            echo ""
            echo "  Downloading from: $(basename $manifest)"
            python "${SCRIPT_DIR}/download_pdbs.py" \
                --manifest "$manifest" \
                --out_dir "${PDB_DIR}" \
                --workers 8
        fi
    done
    PDB_COUNT=$(ls "${PDB_DIR}"/*.pdb 2>/dev/null | wc -l)
    echo ""
    echo "  Total PDB files: ${PDB_COUNT}"
    echo ""
fi

# ── Step 3: Run Experiments ──
if [ "${START_STEP}" -le 3 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 3: Running Experiments (real PAF encoder)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python "${SCRIPT_DIR}/run_experiments.py" \
        --experiment all \
        --manifest_dir "${DATA_DIR}" \
        --pdb_dir "${PDB_DIR}" \
        --out_dir "${RESULTS_DIR}" \
        --permutations 10000
    echo ""
fi

# ── Summary ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Results:"
ls -la "${RESULTS_DIR}"/*.json 2>/dev/null || echo "  (none)"
echo ""

for json_file in "${RESULTS_DIR}"/*.json; do
    if [ -f "$json_file" ]; then
        echo "  ── $(basename $json_file) ──"
        python -c "
import json
with open('$json_file') as f:
    data = json.load(f)
for key, val in data.items():
    if isinstance(val, dict):
        if 'accuracy' in val:
            bacc = val.get('balanced_accuracy', val['accuracy'])
            d = val.get('cohens_d', '?')
            p = val.get('p_value', '?')
            print(f'    {key:35s}: acc={val[\"accuracy\"]:.3f}  bacc={bacc:.3f}  d={d}  p={p}')
        elif 'ms_per_pocket' in val:
            print(f'    {key:35s}: {val[\"ms_per_pocket\"]:.1f} ms/pocket')
        elif 'us_per_pair' in val:
            print(f'    {key:35s}: {val[\"us_per_pair\"]:.2f} us/pair')
" 2>/dev/null || true
        echo ""
    fi
done

echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
