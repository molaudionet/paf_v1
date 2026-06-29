# Frozen Dataset — Protein Acoustic Fingerprinting (PAF)

This directory contains the **frozen dataset** used to produce the results reported in:

> Zhou ER, Zhou CJ. *Acoustic Encoding of Protein Binding Pockets: A Deterministic
> Representation for AI-Driven Selectivity Profiling in Drug Discovery.*
> Medicinal Chemistry Research (submitted).

The frozen PDB identifier lists below **fully determine** every value in the paper's
results table. Running the pipeline on these exact identifiers reproduces the reported
numbers to three decimal places.

---

## Why a frozen dataset

The curation step queries the live RCSB Protein Data Bank, whose contents change over
time, so re-curating from scratch yields a slightly different set of structures on each
run (and a small number fail to download or encode each time). To make the results
exactly reproducible, the precise set of structures that produced the published numbers
is captured here as fixed identifier lists. **Use these lists — do not re-curate — to
reproduce the paper.**

---

## Files

| File | Structures (N) | Description |
|------|----------------|-------------|
| `cross_family_FROZEN.csv` | 1,483 | Cross-family benchmark: 15 protein families |
| `kinase_manifest.csv`     | 273 / 280 | KLIFS kinase set (subfamily and DFG tasks) |

Each row lists a PDB identifier and its label column(s) (e.g. `family`, and for kinases
`subfamily` / DFG annotation). The first row is a header.

### Cross-family composition (N = 1,483)

| Family | N | Family | N | Family | N |
|--------|---|--------|---|--------|---|
| kinase | 290 | proteasome | 80 | aspartyl protease | 63 |
| nuclear receptor | 195 | DHFR | 80 | serine protease | 44 |
| metalloprotease | 128 | COX | 79 | HSP90 | 39 |
| phosphatase | 99 | HDAC | 78 | GPCR | 19 |
| phosphodiesterase | 98 | carbonic anhydrase | 94 | | |
| bromodomain | 97 | | | | |

**Total: 1,483 structures across 15 families.**

### Kinase tasks
- **Subfamily:** 273 kinase structures across 42 subfamilies (KLIFS; subfamilies with
  ≥3 representatives).
- **DFG conformation:** 280 kinase structures (253 DFG-in, 27 DFG-out).

---

## Reproducing the paper's results

From the project directory (where `run_experiments.py` and the encoder live), point the
experiments at the **frozen** manifests:

```bash
# cross-family (N = 1,483)
python3 run_experiments.py \
    --experiment cross_family \
    --manifest data/cross_family_FROZEN.csv \
    --pdbdir   data/pdbs

# kinase subfamily + DFG
python3 run_experiments.py \
    --experiment kinase \
    --manifest data/kinase_manifest.csv \
    --pdbdir   data/pdbs
```

PDB coordinate files for every identifier are downloaded automatically into
`data/pdbs/` if not already present. A small number of identifiers may fail to download
if they are later removed or superseded in the PDB; the frozen lists record the exact set
used, so any such structure can be retrieved from the PDB archive by its identifier.

### Expected results (frozen dataset)

| Task | Method | Acc. | Bal. Acc. | Cohen's *d* |
|------|--------|------|-----------|-------------|
| Cross-family (N=1,483, 15 fam.) | Spectral (PAF) | 0.857 | 0.805 | 1.419 |
|  | Mean aggregation | 0.819 | 0.769 | 0.471 |
|  | Radial histogram | 0.599 | 0.508 | 0.378 |
| Kinase subfamily (N=273, 42 subfam.) | Spectral (PAF) | 0.788 | 0.749 | 1.702 |
|  | Mean aggregation | 0.575 | 0.555 | 0.716 |
|  | Radial histogram | 0.227 | 0.211 | 0.171 |
| DFG conformation (N=280; 253:27) | Spectral (PAF) | 0.986 | 0.959 | — (DFG-out recall 92.6%) |
|  | Mean aggregation | 0.954 | 0.842 | — (recall 70.4%) |
|  | Radial histogram | 0.921 | 0.741 | — (recall 51.9%) |

Minor third-decimal variation is expected across machines (e.g. permutation-test sampling,
BLAS differences); the reported effect sizes and accuracies are stable.

---

## How the frozen lists were generated

The cross-family list is the exact set of structures that (a) downloaded successfully and
(b) encoded successfully under the real PAF encoder, after applying the ≥5-structures-per-family
filter. It was captured directly from an encoding run rather than re-curated, so it pins the
dataset permanently. See `capture_ids.py` and `paf_freeze.py` in the project for the capture
and freeze tooling.

---

## Data sources and licensing

Structural data are derived from the **RCSB Protein Data Bank** (https://www.rcsb.org) and
the **KLIFS** kinase database (https://klifs.net). These resources are freely available for
research; please cite them and observe their terms of use. This directory redistributes only
PDB **identifier lists** and labels, not the coordinate files themselves.
