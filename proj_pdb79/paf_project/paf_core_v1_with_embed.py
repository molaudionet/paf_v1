# paf_core_v1.py
# Protein Acoustic Field (PAF) v1:
# - Robust ligand centroid (uses ligand_resname, heavy-atom filtering, blacklist)
# - Kinase motif fallback (VAIK/HRD/DFG)
# - Two-anchor geometry: r1->time, r2->frequency modulation (FM)
# - Enriched residue features: physchem + flex + contact + DSSP(3) + BLOSUM62(20)
#
# Dependencies:
#   pip install biopython numpy pandas scipy matplotlib

from __future__ import annotations
import os, re, math, json, hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
#from Bio.PDB.Polypeptide import three_to_one

# --- 3-letter to 1-letter AA conversion (robust) ---
_3TO1 = {
 "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
 "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
 "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
 # common modified residues
 "MSE":"M",  # selenomethionine
}
def resname_to_aa(resname: str) -> str:
    r = (resname or "").strip().upper()
    if r in _3TO1:
        return _3TO1[r]
    raise KeyError(f"Unknown residue name: {r}")

_3TO1 = {
 "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
 "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
 "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
 "MSE":"M"
}
def three_to_one_safe(resname: str) -> str:
    r = resname.strip().upper()
    if r in _3TO1:
        return _3TO1[r]
    raise KeyError(r)

# Optional DSSP
try:
    from Bio.PDB.DSSP import DSSP  # requires mkdssp installed to work
    _HAS_DSSP = True
except Exception:
    DSSP = None
    _HAS_DSSP = False


# -----------------------------
# Params
# -----------------------------

@dataclass(frozen=True)
class PAFParams:
    sr: int = 16000
    duration_s: float = 6.0
    pocket_radius_A: float = 10.0

    fmin_hz: float = 150.0
    fmax_hz: float = 2400.0
    sigma_t_s: float = 0.040

    n_bins: int = 256
    eps: float = 1e-8

    # base frequency map weights
    a_hyd: float = 1.0
    a_charge: float = 1.0
    a_vol: float = 0.5

    # two-anchor FM strength
    gamma_fm: float = 0.15  # f *= (1 + gamma*(r2_norm-0.5))

    # contacts
    contact_cutoff_A: float = 6.0

    # ligand selection
    ligand_min_heavy_atoms: int = 10


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Feature tables
# -----------------------------

CHARGE = {"D": -1.0, "E": -1.0, "K": +1.0, "R": +1.0, "H": +0.5}

KD = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3
}
KD_MIN, KD_MAX = -4.5, 4.5

HB_RAW = {
    "D": 1, "E": 1,
    "N": 2, "Q": 2,
    "H": 2,
    "K": 1, "R": 1,
    "S": 2, "T": 2,
    "Y": 1, "W": 1,
    "C": 1
}
AROMATIC = set(["F", "Y", "W", "H"])

SIDECHAIN_HEAVY = {
    "A": 1, "C": 2, "D": 2, "E": 3, "F": 7,
    "G": 0, "H": 6, "I": 4, "K": 5, "L": 4,
    "M": 4, "N": 3, "P": 3, "Q": 4, "R": 6,
    "S": 2, "T": 3, "V": 3, "W": 10, "Y": 8
}

# BLOSUM62 20x20 ordered by AA_ORDER
AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_BLOSUM62 = {
"A":[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
"R":[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
"N":[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
"D":[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
"C":[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
"Q":[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
"E":[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
"G":[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
"H":[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
"I":[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
"L":[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
"K":[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
"M":[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
"F":[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
"P":[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
"S":[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
"T":[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
"W":[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
"Y":[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
"V":[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
}
BLOSUM_MIN, BLOSUM_MAX = -4.0, 11.0  # for normalization


def blosum62_row(aa: str) -> np.ndarray:
    aa = aa.upper()
    row = _BLOSUM62.get(aa, _BLOSUM62["A"])
    v = np.array(row, dtype=np.float32)
    # normalize to [0,1] amplitudes
    v = (v - BLOSUM_MIN) / (BLOSUM_MAX - BLOSUM_MIN)
    v = np.clip(v, 0.0, 1.0)
    return v


def aa_physchem(aa: str) -> Dict[str, float]:
    aa = aa.upper()
    q = float(CHARGE.get(aa, 0.0))
    hyd = float(np.clip((KD.get(aa, 0.0) - KD_MIN) / (KD_MAX - KD_MIN), 0.0, 1.0))
    hb = float(np.clip(HB_RAW.get(aa, 0) / 2.0, 0.0, 1.0))
    aro = 1.0 if aa in AROMATIC else 0.0
    vol = float(np.clip(SIDECHAIN_HEAVY.get(aa, 0) / 10.0, 0.0, 1.0))
    return {"charge": q, "hyd": hyd, "hb": hb, "aro": aro, "vol": vol}


# -----------------------------
# Structure loading
# -----------------------------

def load_structure(path: str, struct_id: str = "X"):
    if path.lower().endswith((".cif", ".mmcif")):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(struct_id, path)


def residue_ca_xyz(res) -> Optional[np.ndarray]:
    if res is None or "CA" not in res:
        return None
    return np.array(res["CA"].get_coord(), dtype=np.float32)


def residue_bfactor(res) -> Optional[float]:
    if res is None:
        return None
    b = []
    for atom_name in ["N", "CA", "C", "O"]:
        if atom_name in res:
            b.append(float(res[atom_name].get_bfactor()))
    return float(np.mean(b)) if b else None


def chain_atom_sequence_and_map(chain) -> Tuple[str, List[Tuple[str, int, str]]]:
    seq = []
    idx_map = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ":
            continue
        if "CA" not in res:
            continue
        resname = res.get_resname()
        try:
            #aa = three_to_one(resname)
            aa = resname_to_aa(resname)
        except KeyError:
            continue
        seq.append(aa)
        idx_map.append((chain.id, int(resseq), str(icode).strip() if icode else ""))
    return "".join(seq), idx_map


def get_residue_by_id(chain, resseq: int, icode: str = ""):
    icode = icode if icode != "" else " "
    key = (" ", int(resseq), icode)
    return chain[key] if key in chain else None


# -----------------------------
# Kinase motif detection
# -----------------------------

@dataclass(frozen=True)
class KinaseMotifs:
    pos_vaiK: int
    pos_hrd_D: int
    pos_dfg_D: int


def find_kinase_motifs(seq: str) -> Optional[KinaseMotifs]:
    hrd_hits = [m.start() for m in re.finditer("HRD", seq)]
    dfg_hits = [m.start() for m in re.finditer("DFG", seq)]
    if not hrd_hits or not dfg_hits:
        return None

    best_pair, best_dist = None, 10**9
    for h in hrd_hits:
        for d in dfg_hits:
            dist = abs(d - h)
            if 15 <= dist <= 120 and dist < best_dist:
                best_dist = dist
                best_pair = (h, d)
    h, d = best_pair if best_pair else (hrd_hits[0], dfg_hits[0])

    pos_hrd_D = h + 2
    pos_dfg_D = d

    vaik = seq.find("VAIK")
    if vaik != -1:
        pos_vaiK = vaik + 3
    else:
        m = re.search(r"([VIL])A([IVL])K", seq)
        if m:
            pos_vaiK = m.start() + 3
        else:
            candidates = [i for i, aa in enumerate(seq) if aa == "K" and (pos_hrd_D - 80) <= i <= (pos_hrd_D - 15)]
            if not candidates:
                return None
            pos_vaiK = min(candidates, key=lambda i: abs(pos_hrd_D - i))

    if not (pos_vaiK < pos_hrd_D < pos_dfg_D):
        return None
    return KinaseMotifs(pos_vaiK=pos_vaiK, pos_hrd_D=pos_hrd_D, pos_dfg_D=pos_dfg_D)


# -----------------------------
# Ligand centroid (robust)
# -----------------------------

_LIGAND_BLACKLIST = {
    # common additives / solvents
    "DMS", "DMSO", "GOL", "EDO", "PEG", "PG4", "PGE", "MPD", "ACT", "ACE", "IPA",
    "TRS", "MES", "HEP", "BME", "FMT", "EPE", "B3P", "NAG", "MAN", "FUC",
    # common ions/metals
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "CO", "NI", "CU"
}


def _heavy_atom_count(res) -> int:
    n = 0
    for atom in res.get_atoms():
        el = (atom.element or "").upper()
        if el and el != "H":
            n += 1
    return n


def ligand_centroid(chain, ligand_resname: Optional[str], params: PAFParams) -> Optional[np.ndarray]:
    """
    If ligand_resname provided: use only that HET residue name.
    Else: choose best HET residue by heavy atoms with blacklist and min heavy atom filter.
    Returns centroid of heavy atoms.
    """
    ligand_resname = (ligand_resname or "").strip()
    candidates = []

    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag == " ":
            continue
        resname = res.get_resname().strip()
        if resname in {"HOH", "WAT"}:
            continue
        if resname in _LIGAND_BLACKLIST:
            continue

        ha = _heavy_atom_count(res)
        if ha < params.ligand_min_heavy_atoms:
            continue

        if ligand_resname and resname != ligand_resname:
            continue

        coords = []
        for atom in res.get_atoms():
            el = (atom.element or "").upper()
            if el and el != "H":
                coords.append(atom.get_coord())
        if coords:
            candidates.append((resname, ha, np.mean(np.array(coords, dtype=np.float32), axis=0)))

    if not candidates:
        return None

    # If ligand_resname specified, take first match (or largest ha among matches)
    if ligand_resname:
        best = max(candidates, key=lambda x: x[1])
        return best[2]

    # Otherwise choose the residue with max heavy atoms
    best = max(candidates, key=lambda x: x[1])
    return best[2]


# -----------------------------
# Pocket extraction
# -----------------------------

@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    aa: str
    ca_xyz: np.ndarray
    radial1_A: float
    radial2_A: float
    bfactor: Optional[float]
    flex: float
    ss_onehot: Tuple[float, float, float]  # helix, sheet, coil
    contact: float
    physchem: Dict[str, float]
    blosum: np.ndarray  # (20,)


@dataclass
class Pocket:
    pocket_id: str
    pdb_path: str
    pdb_sha256: str
    chain_id: str
    center_xyz: np.ndarray
    anchor2_xyz: np.ndarray
    radius_A: float
    residues: List[ResidueRecord]
    meta: Dict[str, str]


def _get_ss_onehot_dssp(dssp_code: str) -> Tuple[float, float, float]:
    # DSSP secondary structure codes: H/G/I = helix; E/B = sheet; others coil
    dssp_code = (dssp_code or " ").strip()
    if dssp_code in {"H", "G", "I"}:
        return (1.0, 0.0, 0.0)
    if dssp_code in {"E", "B"}:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _compute_dssp_onehot(model, pdb_path: str, chain_id: str) -> Dict[Tuple[str, int, str], Tuple[float, float, float]]:
    """
    Returns dict mapping residue id (chain, resseq, icode) to (helix, sheet, coil).
    Requires mkdssp installed. If not available, returns empty dict.
    """
    if not _HAS_DSSP:
        return {}
    try:
        dssp = DSSP(model, pdb_path, dssp="mkdssp")  # may fail; many envs lack mkdssp
    except Exception:
        return {}

    out = {}
    for key in dssp.keys():
        ch, res_id = key[0], key[1]
        if ch != chain_id:
            continue
        hetflag, resseq, icode = res_id
        if hetflag != " ":
            continue
        dssp_code = dssp[key][2]
        out[(chain_id, int(resseq), str(icode).strip() if icode else "")] = _get_ss_onehot_dssp(dssp_code)
    return out


def kinase_center_and_anchor2(chain, ligand_resname: Optional[str], params: PAFParams) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    center: ligand centroid if available, else motif anchors centroid.
    anchor2: use HRD Asp CA if motifs available; else beta3K; else center itself.
    """
    # Ligand-based center
    lig_c = ligand_centroid(chain, ligand_resname, params)
    seq, idx_map = chain_atom_sequence_and_map(chain)
    motifs = find_kinase_motifs(seq)

    if lig_c is not None:
        meta = {"center_method": "ligand_centroid"}
        # anchor2 from motifs if possible (helps recover angular info)
        if motifs is not None:
            cid_h, resseq_h, icode_h = idx_map[motifs.pos_hrd_D]
            resH = get_residue_by_id(chain, resseq_h, icode_h)
            a2 = residue_ca_xyz(resH)
            if a2 is not None:
                meta["anchor2_method"] = "HRD_Asp_CA"
                meta["anchor2_res"] = f"{cid_h}:{resseq_h}{icode_h}"
                return lig_c, a2, meta
        meta["anchor2_method"] = "center_fallback"
        return lig_c, lig_c, meta

    # Motif fallback
    if motifs is None:
        raise RuntimeError("No ligand centroid and kinase motifs not found.")

    cid_k, resseq_k, icode_k = idx_map[motifs.pos_vaiK]
    cid_h, resseq_h, icode_h = idx_map[motifs.pos_hrd_D]
    cid_d, resseq_d, icode_d = idx_map[motifs.pos_dfg_D]

    resK = get_residue_by_id(chain, resseq_k, icode_k)
    resH = get_residue_by_id(chain, resseq_h, icode_h)
    resD = get_residue_by_id(chain, resseq_d, icode_d)

    xyzK = residue_ca_xyz(resK)
    xyzH = residue_ca_xyz(resH)
    xyzD = residue_ca_xyz(resD)
    if xyzK is None or xyzH is None or xyzD is None:
        raise RuntimeError("Missing CA coords for motif anchors.")

    center = (xyzK + xyzH + xyzD) / 3.0
    meta = {
        "center_method": "motif_anchors",
        "anchor_beta3K": f"{cid_k}:{resseq_k}{icode_k}",
        "anchor_hrdD": f"{cid_h}:{resseq_h}{icode_h}",
        "anchor_dfgD": f"{cid_d}:{resseq_d}{icode_d}",
        "anchor2_method": "HRD_Asp_CA",
    }
    return center, xyzH, meta


def extract_pocket(pdb_path: str, chain_id: str, ligand_resname: Optional[str], params: PAFParams) -> Pocket:
    structure = load_structure(pdb_path, struct_id=os.path.basename(pdb_path))
    model = next(structure.get_models())
    chain = model[chain_id]

    center, anchor2, meta = kinase_center_and_anchor2(chain, ligand_resname, params)

    # DSSP mapping (optional)
    dssp_map = _compute_dssp_onehot(model, pdb_path, chain_id)

    # First pass: collect residues within radius
    tmp = []
    bvals = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ":
            continue
        if "CA" not in res:
            continue
        try:
            #aa = three_to_one(res.get_resname())
            aa = resname_to_aa(res.get_resname())
            #aa = resname_to_aa(resname)
        except KeyError:
            continue

        ca = residue_ca_xyz(res)
        if ca is None:
            continue

        r1 = float(np.linalg.norm(ca - center))
        if r1 > params.pocket_radius_A:
            continue
        r2 = float(np.linalg.norm(ca - anchor2))

        b = residue_bfactor(res)
        if b is not None:
            bvals.append(b)

        ss = dssp_map.get((chain_id, int(resseq), str(icode).strip() if icode else ""), (0.0, 0.0, 1.0))

        pc = aa_physchem(aa)
        bl = blosum62_row(aa)

        tmp.append((res, aa, ca, r1, r2, b, ss, pc, bl, int(resseq), str(icode).strip() if icode else ""))

    if not tmp:
        raise RuntimeError("No pocket residues extracted (check center/radius/chain).")

    # Flex normalization from B-factors
    if bvals:
        bmean, bstd = float(np.mean(bvals)), float(np.std(bvals) + 1e-8)
    else:
        bmean, bstd = 0.0, 1.0

    # Contact count within pocket (CA-CA distances)
    cas = np.stack([x[2] for x in tmp], axis=0)  # (N,3)
    # pairwise distances
    dmat = np.linalg.norm(cas[:, None, :] - cas[None, :, :], axis=-1)
    contacts = (dmat < params.contact_cutoff_A).astype(np.float32)
    np.fill_diagonal(contacts, 0.0)
    contact_counts = contacts.sum(axis=1)  # (N,)
    # normalize contact to [0,1] using max in pocket (stable)
    cmax = float(np.max(contact_counts) + params.eps)
    contact_norm = (contact_counts / cmax).astype(np.float32)

    residues: List[ResidueRecord] = []
    for i, (res, aa, ca, r1, r2, b, ss, pc, bl, resseq, icode) in enumerate(tmp):
        if b is None or not bvals:
            flex = 0.5
        else:
            z = (float(b) - bmean) / bstd
            z = float(np.clip(z, -2.0, 2.0))
            flex = float((z + 2.0) / 4.0)

        residues.append(
            ResidueRecord(
                chain_id=chain_id,
                resseq=resseq,
                icode=icode,
                aa=aa,
                ca_xyz=ca,
                radial1_A=r1,
                radial2_A=r2,
                bfactor=b,
                flex=flex,
                ss_onehot=ss,
                contact=float(contact_norm[i]),
                physchem=pc,
                blosum=bl
            )
        )

    pocket_id = f"{os.path.basename(pdb_path)}|{chain_id}"
    return Pocket(
        pocket_id=pocket_id,
        pdb_path=pdb_path,
        pdb_sha256=sha256_file(pdb_path),
        chain_id=chain_id,
        center_xyz=center,
        anchor2_xyz=anchor2,
        radius_A=params.pocket_radius_A,
        residues=residues,
        meta=meta
    )


# -----------------------------
# PAF signal -> spectrum
# -----------------------------

def _freq_base(hyd: float, charge: float, vol: float, params: PAFParams) -> float:
    z = params.a_hyd * hyd + params.a_charge * charge + params.a_vol * vol
    s = sigmoid(z)
    return params.fmin_hz + (params.fmax_hz - params.fmin_hz) * s


def _gaussian_sine(t: np.ndarray, tau: float, f: float, phi: float, sigma_t: float) -> np.ndarray:
    x = t - tau
    win = np.exp(-(x * x) / (2.0 * sigma_t * sigma_t))
    return win * np.sin(2.0 * np.pi * f * x + phi)


def _logspace_edges(fmin: float, fmax: float, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def pocket_to_embedding(pocket: Pocket, params: PAFParams) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      spec: (K, B)
      channel_names: list[str]
    Channel set:
      6 physchem/dyn + contact + ss(3) + blosum(20) = 30 channels
    """
    n = int(params.sr * params.duration_s)
    t = np.arange(n, dtype=np.float32) / float(params.sr)
    R = float(params.pocket_radius_A)

    # Define channels
    chan_names = ["chg", "hyd", "hb", "aro", "vol", "dyn", "contact", "ss_helix", "ss_sheet", "ss_coil"]
    chan_names += [f"blosum_{aa}" for aa in AA_ORDER]  # 20

    K = len(chan_names)
    sig = np.zeros((K, n), dtype=np.float32)

    for rr in pocket.residues:
        # r1->time
        r1 = float(np.clip(rr.radial1_A, 0.0, R))
        tau = float(params.duration_s * (r1 / R))

        # r2->FM
        r2 = float(np.clip(rr.radial2_A, 0.0, 2 * R))
        r2n = float(np.clip(r2 / (2 * R), 0.0, 1.0))
        fm = 1.0 + params.gamma_fm * (r2n - 0.5)

        q = float(rr.physchem["charge"])
        hyd = float(rr.physchem["hyd"])
        hb = float(rr.physchem["hb"])
        aro = float(rr.physchem["aro"])
        vol = float(rr.physchem["vol"])
        dyn = float(rr.flex)

        f0 = _freq_base(hyd=hyd, charge=q, vol=vol, params=params)
        f = float(np.clip(f0 * fm, params.fmin_hz, params.fmax_hz))

        phi = 0.0 if q >= 0 else np.pi
        w = _gaussian_sine(t=t, tau=tau, f=f, phi=phi, sigma_t=params.sigma_t_s)

        # amplitudes (all nonnegative)
        amps = [
            abs(q), hyd, hb, aro, vol, dyn,
            float(rr.contact),
            float(rr.ss_onehot[0]), float(rr.ss_onehot[1]), float(rr.ss_onehot[2]),
            *[float(x) for x in rr.blosum.tolist()]
        ]

        for k in range(K):
            a = float(amps[k])
            if a != 0.0:
                sig[k, :] += a * w

    # Spectrum binning
    freqs = np.fft.rfftfreq(n, d=1.0 / params.sr)
    edges = _logspace_edges(params.fmin_hz, params.fmax_hz, params.n_bins)
    spec = np.zeros((K, params.n_bins), dtype=np.float32)

    for k in range(K):
        mag = np.abs(np.fft.rfft(sig[k, :])).astype(np.float32)
        for b in range(params.n_bins):
            lo, hi = edges[b], edges[b + 1]
            m = (freqs >= lo) & (freqs < hi)
            spec[k, b] = float(np.sum(mag[m])) if np.any(m) else 0.0
        s = float(np.sum(spec[k, :]) + params.eps)
        spec[k, :] /= s

    return spec, chan_names


# -----------------------------
# Public API wrappers (for downstream scripts)
# -----------------------------
def embed_pocket(
    pdb_path: str,
    chain: str,
    ligand_resname: str,
    radius: float = 10.0,
    gamma_fm: float = 0.15,
    sigma_t: float = 0.04,
    a_hyd: float = 1.0,
    a_charge: float = 1.0,
    a_vol: float = 0.5,
):
    """Compute a pocket PAF embedding from a PDB file.

    Signature matches downstream scripts (ligand_retrieval_v1.py, etc.).

    Returns:
      (E_mag, meta_dict)

    E_mag shape: (K, B) where K=30 channels (PAF v1) and B=params.n_bins (default 256).
    """
    params = PAFParams(
        pocket_radius_A=float(radius),
        gamma_fm=float(gamma_fm),
        sigma_t_s=float(sigma_t),
        a_hyd=float(a_hyd),
        a_charge=float(a_charge),
        a_vol=float(a_vol),
    )

    pocket = extract_pocket(
        pdb_path=str(pdb_path),
        chain_id=str(chain),
        ligand_resname=str(ligand_resname) if ligand_resname is not None else None,
        params=params,
    )
    spec, channel_names = pocket_to_embedding(pocket, params=params)
    E_mag = np.asarray(spec, dtype=np.float32)

    meta = {
        "pdb_path": str(pdb_path),
        "chain": str(chain),
        "ligand_resname": str(ligand_resname),
        "center_mode": pocket.center_mode,
        "n_residues": len(pocket.residues),
        "params": {
            "radius": float(radius),
            "gamma_fm": float(gamma_fm),
            "sigma_t": float(sigma_t),
            "a_hyd": float(a_hyd),
            "a_charge": float(a_charge),
            "a_vol": float(a_vol),
            "n_bins": int(params.n_bins),
        },
        "channel_names": channel_names,
    }
    return E_mag, meta

# Backward-compatible aliases some pipelines look for
def pocket_embedding_from_pdb(*args, **kwargs):
    return embed_pocket(*args, **kwargs)

def compute_pocket_embedding(*args, **kwargs):
    return embed_pocket(*args, **kwargs)

def embed_from_pdb(*args, **kwargs):
    return embed_pocket(*args, **kwargs)
