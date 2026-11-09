import hashlib
import os
from multiprocessing import Pool

import numpy as np
import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures, Crippen, Descriptors, Lipinski, rdMolDescriptors
from torch_geometric.data import Data, InMemoryDataset

_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

RDLogger.DisableLog("rdApp.*")

ATOM_LIST = list(range(1, 119))  
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
DEFAULT_BOND_LENGTH = 1.5  # mean bond length (A) used when 3D embedding fails
CIP_CODES = ["R", "S"]  # per-atom chirality (R/S); neither means "unspecified / not chiral"

NUM_ATOM_FEATURES = len(ATOM_LIST) + 1 + len(HYBRIDIZATIONS) + 1 + 5 + (len(CIP_CODES) + 1) + 2

# Whole-molecule, pharmacokinetic properties that GINE's local message
# passing may not fully capture; they feed a separate "global" branch. The
# scales are fixed numbers chosen for typical drug-like ranges. Nothing is
# learned from the data, and the backend uses the same constants, which keeps
# training and inference consistent.
GLOBAL_FEATURE_SCALES = {
    "molwt": 500.0,       # ExactMolWt -- typical drug MW range ~200-600
    "logp": 5.0,           # MolLogP -- typical range -2..6
    "tpsa": 150.0,          # TPSA -- typical range 0..200
    "hbd": 5.0,             # NumHDonors
    "hba": 10.0,            # NumHAcceptors
    "rotatable": 10.0,      # NumRotatableBonds
}
NUM_PHARMACOKINETIC_FEATURES = len(GLOBAL_FEATURE_SCALES)

# Molecule-target (protein/receptor) interaction features derived from ChEMBL
# bioactivity data: a signal that comes from real biological mechanism -- what
# the molecule binds -- rather than from structure. Coverage is incomplete
# (about 49% of SIDER molecules have data, and the other datasets have none);
# where it is missing the vector is all zeros with has_data=0 (see
# scripts/data/build_target_features.py).
#
# Each process must state the version and file path explicitly, from the
# config, through `configure_target_features()`. Without that call the ChEMBL
# target features are not used at all and a zero vector is returned. The older
# behaviour read a fixed path ("data/raw/chembl_target_features.json") once per
# process, and switching between V2 and V3 meant overwriting that file by hand,
# which risked silently training or predicting with the wrong version if a
# process started at the wrong moment.
#
# NUM_GLOBAL_FEATURES must be a constant at import time, because it fixes the
# models' nn.Linear dimension; it cannot be read from a .json file. Both V2 and
# V3 produce 101 target features, and configure_target_features verifies that
# and raises on a mismatch.
EXPECTED_NUM_TARGET_FEATURES = 101

_ACTIVE_TARGET_FEATURES = {
    "version": None, "path": None, "sha256": None,
    "num_features": 0, "vectors": {},
}


def configure_target_features(version: str, path: str) -> dict:
    """Explicitly configure the ChEMBL target feature file for this process.

    Every training and evaluation script must call this once before building a
    model (see src/train.py: load_config). Calling it again in the same process
    with a different version or hash raises: the "one ChEMBL feature file per
    process" rule is enforced by the code rather than left to the user's
    attention.
    """
    import hashlib
    import json

    with open(path, "rb") as f:
        raw = f.read()
    sha256 = hashlib.sha256(raw).hexdigest()

    if _ACTIVE_TARGET_FEATURES["sha256"] is not None:
        if _ACTIVE_TARGET_FEATURES["sha256"] != sha256:
            raise RuntimeError(
                f"the ChEMBL feature file was already configured in this process as "
                f"'{_ACTIVE_TARGET_FEATURES['version']}' ({_ACTIVE_TARGET_FEATURES['path']}); "
                f"'{version}' ({path}) cannot replace it "
                f"(one ChEMBL feature file per process). "
                f"Start a separate process for a different version."
            )
        return get_feature_provenance()

    data = json.loads(raw)
    num_features = data["num_features"]
    if num_features != EXPECTED_NUM_TARGET_FEATURES:
        raise RuntimeError(
            f"'{path}' holds {num_features} features, but "
            f"EXPECTED_NUM_TARGET_FEATURES={EXPECTED_NUM_TARGET_FEATURES} was expected. "
            f"The model dimension (NUM_GLOBAL_FEATURES) rests on that constant, and "
            f"activating an incompatible file would break the model, so this is refused."
        )
    _ACTIVE_TARGET_FEATURES.update(
        version=version, path=path, sha256=sha256,
        num_features=num_features, vectors=data["inchikey_to_vector"],
    )
    return get_feature_provenance()


def get_feature_provenance() -> dict:
    """The provenance information to embed in manifests and result JSONs.

    If `configure_target_features()` was never called -- for example by a
    script that does not use the ChEMBL target features -- this says so
    explicitly rather than pretending some "default" file was used.
    """
    if _ACTIVE_TARGET_FEATURES["version"] is None:
        return {"version": None, "path": None, "sha256": None, "note": "configure_target_features was not called"}
    return {k: _ACTIVE_TARGET_FEATURES[k] for k in ("version", "path", "sha256", "num_features")}


def _target_features(mol) -> np.ndarray:
    # NUM_GLOBAL_FEATURES (the model dimension) always assumes
    # PK + EXPECTED_NUM_TARGET_FEATURES, so when configure_target_features()
    # has not been called a zero vector is returned to keep the dimension
    # consistent. The same path is taken for an unrecognised molecule; check
    # get_feature_provenance()["version"] to tell the two apart.
    if _ACTIVE_TARGET_FEATURES["version"] is None:
        return np.zeros(EXPECTED_NUM_TARGET_FEATURES, dtype=np.float32)
    try:
        inchikey = Chem.MolToInchiKey(mol)
        vec = _ACTIVE_TARGET_FEATURES["vectors"].get(inchikey)
    except Exception:
        vec = None
    if vec is None:
        vec = [0] * EXPECTED_NUM_TARGET_FEATURES
    return np.array(vec, dtype=np.float32)


NUM_GLOBAL_FEATURES = NUM_PHARMACOKINETIC_FEATURES + EXPECTED_NUM_TARGET_FEATURES


# ---------------------------------------------------------------------------
# GIN-v2 ablation 2.3: RDKit 2D descriptor fusion
# (ablation.use_rdkit_descriptors).
#
# Chemprop's `rdkit_2d_normalized` feature produces exactly 200 curated
# descriptors via the `descriptastorus` package, which is not installed in this
# environment. Instead, all 2D descriptors in RDKit's own
# `Descriptors._descList` are used -- 217 of them here. That is the same
# category of information (molecular topology and pharmacokinetic descriptors)
# but not identical to Chemprop's curated subset, which should be kept in mind
# when interpreting the 2.3 result.
# ---------------------------------------------------------------------------
from rdkit.Chem import Descriptors as _RDKitDescriptors  # noqa: E402

_RDKIT_DESCRIPTOR_FUNCS = list(_RDKitDescriptors._descList)
NUM_RDKIT_DESCRIPTORS = len(_RDKIT_DESCRIPTOR_FUNCS)

_RDKIT_DESC_STATE = {"enabled": False, "mean": None, "std": None, "path": None}


def compute_rdkit_descriptors_raw(mol) -> np.ndarray:
    """The raw, unnormalised RDKit 2D descriptor vector. Any descriptor that
    cannot be computed -- which happens for some exotic or invalid molecules --
    becomes NaN and is cleaned up during normalisation; it is never silently
    turned into 0 here."""
    out = np.empty(NUM_RDKIT_DESCRIPTORS, dtype=np.float64)
    for i, (_name, fn) in enumerate(_RDKIT_DESCRIPTOR_FUNCS):
        try:
            v = fn(mol)
            out[i] = v if np.isfinite(v) else np.nan
        except Exception:
            out[i] = np.nan
    return out


Z_CLIP = 10.0


class RDKitDescriptorNormalizer:
    """Fits mean/std on the training-set SMILES only, so no validation/test
    statistics leak in. `transform` maps NaN/inf values to 0 (the mean in
    normalized space).
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, smiles_list):
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rows.append(compute_rdkit_descriptors_raw(mol))
        arr = np.array(rows, dtype=np.float64)
        self.mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std[std < 1e-8] = 1.0  # constant (zero-variance) descriptor: avoid division by zero
        self.std = std
        return self

    def transform(self, mol) -> np.ndarray:
        raw = compute_rdkit_descriptors_raw(mol)
        z = (raw - self.mean) / self.std
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        # Because the statistics are fit on SIDER's training set, the
        # auxiliary datasets (HIV, MUV) contain rare extreme values (|z| up to
        # 50, about 0.1% of values above 10). Clipping keeps one outlier from
        # derailing training.
        z = np.clip(z, -Z_CLIP, Z_CLIP)
        return z.astype(np.float32)

    def to_json(self, path):
        import json
        json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, open(path, "w"))

    @classmethod
    def from_json(cls, path):
        import json
        d = json.load(open(path))
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))


def configure_rdkit_descriptors(enabled: bool, normalizer_path: str = None):
    """Explicitly configure the `ablation.use_rdkit_descriptors` flag for this
    process, following the same pattern as `configure_target_features`. With
    `enabled=True`, `normalizer_path` is mandatory: rather than normalising
    with unfitted statistics (an arbitrary zero mean and unit variance), a
    missing file raises.
    """
    if not enabled:
        _RDKIT_DESC_STATE.update(enabled=False, mean=None, std=None, path=None)
        return
    if normalizer_path is None:
        raise RuntimeError(
            "ablation.use_rdkit_descriptors=true but no normalizer_path was given. "
            "First produce the training-set statistics with "
            "RDKitDescriptorNormalizer().fit(train_smiles).to_json(path)."
        )
    norm = RDKitDescriptorNormalizer.from_json(normalizer_path)
    _RDKIT_DESC_STATE.update(enabled=True, mean=norm.mean, std=norm.std, path=normalizer_path)


def num_global_features(include_rdkit: bool = None) -> int:
    """The dynamic global-feature dimension, for `build_model` to pass
    explicitly to GINEncoder. With `include_rdkit=None` it follows this
    process's current state, as set by `configure_rdkit_descriptors`."""
    enabled = _RDKIT_DESC_STATE["enabled"] if include_rdkit is None else include_rdkit
    base = NUM_PHARMACOKINETIC_FEATURES + EXPECTED_NUM_TARGET_FEATURES
    return base + NUM_RDKIT_DESCRIPTORS if enabled else base


def _global_features(mol) -> np.ndarray:
    """The molecule's whole-structure, pharmacokinetics-related RDKit
    descriptors (molecular weight, LogP, TPSA, H-bond donor/acceptor counts,
    rotatable bonds), plus the ChEMBL target-interaction features when
    available, plus -- when ablation.use_rdkit_descriptors=true -- the
    normalised 2D RDKit descriptor set. Normalised with fixed scales."""
    try:
        values = [
            Descriptors.ExactMolWt(mol) / GLOBAL_FEATURE_SCALES["molwt"],
            Crippen.MolLogP(mol) / GLOBAL_FEATURE_SCALES["logp"],
            rdMolDescriptors.CalcTPSA(mol) / GLOBAL_FEATURE_SCALES["tpsa"],
            Lipinski.NumHDonors(mol) / GLOBAL_FEATURE_SCALES["hbd"],
            Lipinski.NumHAcceptors(mol) / GLOBAL_FEATURE_SCALES["hba"],
            Descriptors.NumRotatableBonds(mol) / GLOBAL_FEATURE_SCALES["rotatable"],
        ]
    except Exception:
        values = [0.0] * NUM_PHARMACOKINETIC_FEATURES
    pk = np.array(values, dtype=np.float32)
    target = _target_features(mol)
    parts = [pk, target] if target.size else [pk]
    if _RDKIT_DESC_STATE["enabled"]:
        normalizer = RDKitDescriptorNormalizer(mean=_RDKIT_DESC_STATE["mean"], std=_RDKIT_DESC_STATE["std"])
        parts.append(normalizer.transform(mol))
    return np.concatenate(parts)
NUM_BOND_FEATURES = len(BOND_TYPES) + 1 + 2 + len(STEREO_TYPES) + 1 + 1


def _one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    encoding[choices.index(value) if value in choices else len(choices)] = 1
    return encoding


def _atom_features(atom, is_donor, is_acceptor):
    cip_code = atom.GetPropsAsDict().get("_CIPCode", None)
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_LIST)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
        + _one_hot(cip_code, CIP_CODES)
        + [int(is_donor), int(is_acceptor)]
    )


def _donor_acceptor_flags(mol):
    """Decide for each atom whether it is an H-bond donor or acceptor, using
    RDKit's standard pharmacophore feature factory: a direct donor/acceptor
    classification rather than an indirect count such as TotalNumHs."""
    n = mol.GetNumAtoms()
    is_donor = [False] * n
    is_acceptor = [False] * n
    for feat in _FEATURE_FACTORY.GetFeaturesForMol(mol):
        family = feat.GetFamily()
        if family == "Donor":
            for idx in feat.GetAtomIds():
                is_donor[idx] = True
        elif family == "Acceptor":
            for idx in feat.GetAtomIds():
                is_acceptor[idx] = True
    return is_donor, is_acceptor


def _bond_features(bond, bond_length):
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), STEREO_TYPES)
        + [bond_length]
    )


def _embed_conformer(mol):
    """Generate the molecule's 3D conformer, returning None on failure.

    For metal-containing compounds such as Gd/Ca contrast agents, the MMFF
    force field may find no parameters and refuse to optimise; the
    unoptimised embedding is kept in that case.
    """
    try:
        mol_h = Chem.AddHs(mol)
    except Exception:
        return None

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    params.maxIterations = 50
    params.timeout = 5  # seconds; keeps abnormally large molecules from hanging
    try:
        cid = AllChem.EmbedMolecule(mol_h, params)
    except Exception:
        return None
    if cid == -1:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        pass

    try:
        mol_h = Chem.RemoveHs(mol_h)
        if mol_h.GetNumAtoms() != mol.GetNumAtoms():
            # For some molecules (salts, multi-fragment structures) RemoveHs
            # can change the atom count relative to the original, so the bond
            # indices no longer match the conformer; treat the embedding as
            # failed.
            return None
        return mol_h.GetConformer()
    except Exception:
        return None


def augment_smiles(smiles: str, n: int) -> list:
    """Produce n different, non-canonical SMILES spellings of the same molecule.

    RDKit's `doRandom=True` mode randomises the order in which atoms are
    written, producing different but valid SMILES strings for the same
    molecule: the molecule is unchanged, only its spelling. On small datasets
    this is a cheap and effective augmentation. Duplicate spellings -- equal to
    the original or to each other -- are discarded.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    seen = {smiles}
    variants = []
    attempts = 0
    while len(variants) < n and attempts < n * 5:
        attempts += 1
        try:
            variant = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        except Exception:
            continue
        if variant not in seen:
            seen.add(variant)
            variants.append(variant)
    return variants


def _featurize_arrays(smiles: str, want_pos: bool = False):
    """Convert SMILES to plain numpy arrays, containing no torch tensors.

    This removes any sharing of torch tensors between multiprocessing.Pool
    workers, and with it the risk of exhausting /dev/shm: numpy arrays travel
    through the pipe as ordinary pickles.

    With `want_pos=True` the atom positions of the 3D conformer -- already
    generated for bond_length -- are returned as well, as an Nx3 array, for
    position-based architectures such as EGNN. The default (False) behaviour is
    unchanged: the existing callers (smiles_to_graph, _featurize_one) still
    receive a 4-tuple, so this does not affect the hash of the shared graph
    cache (data/processed*) and triggers no recomputation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass
    is_donor, is_acceptor = _donor_acceptor_flags(mol)

    x = np.array(
        [_atom_features(a, is_donor[a.GetIdx()], is_acceptor[a.GetIdx()]) for a in mol.GetAtoms()],
        dtype=np.float32,
    )

    conformer = _embed_conformer(mol)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if conformer is not None:
            bond_length = conformer.GetAtomPosition(i).Distance(conformer.GetAtomPosition(j))
        else:
            bond_length = DEFAULT_BOND_LENGTH
        feat = _bond_features(bond, bond_length)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if edge_indices:
        edge_index = np.array(edge_indices, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, NUM_BOND_FEATURES), dtype=np.float32)

    global_feats = _global_features(mol)

    if not want_pos:
        return x, edge_index, edge_attr, global_feats

    if conformer is not None:
        pos = np.array(
            [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
    else:
        pos = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    return x, edge_index, edge_attr, global_feats, pos


def _featurize_one_with_pos(args):
    """_featurize_one with positions, returning plain numpy (never torch
    tensors) for multiprocessing.Pool workers. Returning a PyG Data object,
    which contains torch tensors, directly through a Pool exhausts /dev/shm and
    raises MaybeEncodingError (see the _featurize_arrays docstring), so the Data
    object must always be built in the main process."""
    smiles, y, w = args
    arrays = _featurize_arrays(smiles, want_pos=True)
    if arrays is None:
        return None
    x, edge_index, edge_attr, global_feats, pos = arrays
    return x, edge_index, edge_attr, global_feats, pos, np.asarray(y, dtype=np.float32), np.asarray(w, dtype=np.float32), smiles


def smiles_to_graph_with_pos(smiles: str, y=None, w=None) -> "Data | None":
    """The same as smiles_to_graph but with `data.pos` (3D atom positions) added
    for EGNN. It uses a separate cache path (see scripts/legacy/train_egnn.py) and
    does not touch the existing 2D-only graph cache."""
    arrays = _featurize_arrays(smiles, want_pos=True)
    if arrays is None:
        return None
    x, edge_index, edge_attr, global_feats, pos = arrays

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        global_features=torch.from_numpy(global_feats).unsqueeze(0),
        pos=torch.from_numpy(pos),
    )
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)
    if w is not None:
        data.w = torch.tensor(w, dtype=torch.float).unsqueeze(0)
    data.smiles = smiles
    return data


def smiles_to_graph(smiles: str, y=None, w=None) -> Data | None:
    """Convert a SMILES string into a PyG Data object (a molecular graph)."""
    arrays = _featurize_arrays(smiles)
    if arrays is None:
        return None
    x, edge_index, edge_attr, global_feats = arrays

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        global_features=torch.from_numpy(global_feats).unsqueeze(0),
    )
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)
    if w is not None:
        data.w = torch.tensor(w, dtype=torch.float).unsqueeze(0)
    data.smiles = smiles
    return data


def _featurize_one(args):
    smiles, y, w = args
    arrays = _featurize_arrays(smiles)
    if arrays is None:
        return None
    x, edge_index, edge_attr, global_feats = arrays
    return x, edge_index, edge_attr, global_feats, np.asarray(y, dtype=np.float32), np.asarray(w, dtype=np.float32), smiles


class MoleculeGraphDataset(InMemoryDataset):
    """Convert a DeepChem Dataset (SMILES + labels + mask) into a list of PyG
    graphs.

    The results are written to disk under `cache_dir`: for the same SMILES set
    and the same feature dimensions, the 3D embedding is not redone but read
    back from disk.
    """

    def __init__(self, dc_dataset, cache_dir="data/processed", num_workers=None):
        super().__init__()

        cache_path = self._cache_path(dc_dataset, cache_dir)
        if os.path.exists(cache_path):
            self.data, self.slices = torch.load(cache_path, weights_only=False)
            return

        args = list(zip(dc_dataset.ids, dc_dataset.y, dc_dataset.w))
        # The host has 16 cores but only 15 GB of RAM. With 8 parallel
        # workers, each holding a full copy of its molecule chunk (3D
        # conformers included) in memory, the machine swapped and slowed down
        # badly, with a risk of OOM. Four workers reduce the memory pressure at
        # the cost of somewhat longer wall-clock time.
        num_workers = num_workers or min(os.cpu_count() or 1, 1)

        if num_workers > 1:
            with Pool(num_workers) as pool:
                results = pool.map(_featurize_one, args)
        else:
            results = [_featurize_one(a) for a in args]

        data_list = []
        for r in results:
            if r is None:
                continue
            x, edge_index, edge_attr, global_feats, y, w, smiles = r
            data = Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
                global_features=torch.from_numpy(global_feats).unsqueeze(0),
                y=torch.from_numpy(y).unsqueeze(0),
                w=torch.from_numpy(w).unsqueeze(0),
            )
            data.smiles = smiles
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save((self.data, self.slices), cache_path)

    @staticmethod
    def _cache_path(dc_dataset, cache_dir):
        # This uses num_global_features(), which is dynamic and changes with
        # the RDKit-descriptor setting, rather than the constant
        # NUM_GLOBAL_FEATURES. Otherwise a cache built with the same ChEMBL
        # version but a different descriptor setting would be shared by
        # mistake.
        key = (f"{'|'.join(dc_dataset.ids)}|{NUM_ATOM_FEATURES}|{NUM_BOND_FEATURES}|"
               f"{num_global_features()}|rdkit={_RDKIT_DESC_STATE['enabled']}")
        digest = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(cache_dir, f"graphs_{digest}.pt")
