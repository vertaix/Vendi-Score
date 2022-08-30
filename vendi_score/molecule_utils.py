import itertools

import numpy as np

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


from vendi_score.data_utils import Example, Group


def disable_rdkit_log():
    rdBase.DisableLog("rdApp.*")


def enable_rdkit_log():
    rdBase.EnableLog("rdApp.*")


def get_mol(smiles_or_mol):
    """
    moses: Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def load_molecules(fn, max_samples=2000, unique=False):
    with open(fn, "r") as f:
        lines = f.readlines()[1:]
    examples = []
    seen = set()
    disable_rdkit_log()
    for i, line in enumerate(lines):
        s = line.strip().split(",")[0]
        features = {"s": s}
        if unique and s in seen:
            continue
        seen.add(s)
        mol = get_mol(s)
        if mol is None:
            continue
        e = Example(x=mol, features=features)
        examples.append(e)
        if len(examples) >= max_samples:
            break
    enable_rdkit_log()
    if len(examples) < max_samples:
        print(f"len(examples) = {len(examples)}")
    return examples


def get_tanimoto_K(mols, fp="morgan"):
    N = len(mols)
    K = np.zeros((N, N))
    if fp == "rdk":
        fps = [Chem.RDKFingerprint(x) for x in mols]
    elif fp == "morgan":
        fps = [GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols]
    else:
        fps = [fp(x) for e in mols]
    for i in range(N):
        for j in range(i, N):
            K[i, j] = K[j, i] = DataStructs.FingerprintSimilarity(
                fps[i], fps[j]
            )
    return K
