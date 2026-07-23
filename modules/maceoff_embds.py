from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from mace.calculators.foundations_models import mace_off
from ase import Atoms
import torch
import pandas as pd
import os

"""Utilities for generating optimized 3D conformers from SMILES and computing MACE off embeddings."""

def generate_optimized_3d_conformers(
    smiles_list,
    num_confs=10,
    prune_rms_thresh=-1.0, #eg. no pruning when set to < 0. 
    max_iters=200,
    random_seed=42,
    use_etkdg=True,
    forcefield="MMFF",  # "MMFF" or "UFF"
):
    """Embed and optimize 3D conformers for each SMILES string.

    Parameters:
        smiles_list: iterable of SMILES strings.
        num_confs: number of conformers to generate per molecule.
        prune_rms_thresh: RMS threshold for pruning similar conformers.
        max_iters: maximum force-field minimization steps.
        random_seed: seed for embedding reproducibility.
        use_etkdg: use ETKDG v3 for embedding when True.
        forcefield: force field name, either "MMFF" or "UFF".

    Returns:
        list of RDKit molecules, each with one or more optimized conformers.
    """
    out = []
    params = AllChem.ETKDGv3() if use_etkdg else AllChem.EmbedParameters()
    params.randomSeed = int(random_seed)
    params.pruneRmsThresh = float(prune_rms_thresh)

    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            out.append(None)
            continue

        m = Chem.Mol(mol)
        if m.GetNumConformers():
            m.RemoveAllConformers()

        m = Chem.AddHs(m, addCoords=True)

        cids = list(AllChem.EmbedMultipleConfs(m, numConfs=int(num_confs), params=params))
        if not cids:
            out.append(None)
            continue

        ff = str(forcefield).upper()
        if ff == "MMFF" and AllChem.MMFFHasAllMoleculeParams(m):
            mp = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
            for cid in cids:
                ffobj = AllChem.MMFFGetMoleculeForceField(m, mp, confId=int(cid))
                if ffobj is not None:
                    ffobj.Initialize()
                    ffobj.Minimize(maxIts=int(max_iters))
        else:
            for cid in cids:
                ffobj = AllChem.UFFGetMoleculeForceField(m, confId=int(cid))
                if ffobj is not None:
                    ffobj.Initialize()
                    ffobj.Minimize(maxIts=int(max_iters))

        out.append(m)

    return out

def rdkit_mol_to_ase_atoms(rdkit_mol, conformer_id=0):
    """Convert an RDKit molecule conformer into an ASE Atoms object.

    Parameters:
        rdkit_mol: RDKit molecule with embedded conformers.
        conformer_id: conformer index to convert.

    Returns:
        ASE Atoms object with atomic symbols and positions.
    """
    if rdkit_mol.GetNumConformers() == 0:
        print("Warning: RDKit molecule has no conformers. Returning None.")
        return None  # No conformers available
    
    conf = rdkit_mol.GetConformer(conformer_id)
    
    # Get atomic symbols
    symbols = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
    
    # Get coordinates from conformer
    positions = conf.GetPositions()
    
    # Create ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms

def _append_if_same_dim(collected, vec, ref_dim, skipped_ids, current_id):
    """Append vec to collected if its length matches ref_dim.

    This prevents downstream np.vstack / np.array ragged-shape crashes.

    Parameters:
        collected: list[np.ndarray] to append to
        vec: 1D array-like
        ref_dim: expected length, or None to adopt vec's length
        skipped_ids: list[int] to record current_id when vec is skipped
        current_id: int identifier (e.g., molecule index)

    Returns:
        ref_dim (int): established/unchanged reference dimension
    """
    vec = np.asarray(vec).ravel()
    if vec.ndim != 1:
        skipped_ids.append(current_id)
        return ref_dim

    if ref_dim is None:
        collected.append(vec)
        return int(vec.shape[0])

    if int(vec.shape[0]) == int(ref_dim):
        collected.append(vec)
    else:
        skipped_ids.append(current_id)

    return ref_dim


def compute_mace_off_embeddings(
    mols,
    model_size: str = "medium",
    per_conformer: bool = False,
):
    """Compute MACE-off embeddings.

    If per_conformer is True:
        returns List[List[np.ndarray]] (molecule -> conformer vectors)
    else:
        returns np.ndarray of shape (n_molecules, dim) by averaging conformers.

    Any conformer vectors whose dimensionality doesn't match the first
    successfully computed vector are skipped (so vstack/array never crashes).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_off(model=model_size, device=device)

    embeddings = []
    failed = []

    dim_ref = None
    skipped_dim = []

    for midx, mol in enumerate(tqdm(mols)):
        if mol is None or mol.GetNumConformers() == 0:
            failed.append(midx)
            continue

        conf_vecs = []
        for cid in range(mol.GetNumConformers()):
            atoms = rdkit_mol_to_ase_atoms(mol, cid)
            if atoms is None:
                continue

            # calc.get_descriptors returns atom-wise descriptors: (n_atoms, dim)
            atom_desc = calc.get_descriptors(atoms)
            conf_vec = np.asarray(atom_desc).mean(axis=0)  # (dim,)

            dim_ref = _append_if_same_dim(conf_vecs, conf_vec, dim_ref, skipped_dim, midx)

        if not conf_vecs:
            failed.append(midx)
            continue

        if per_conformer:
            embeddings.append(conf_vecs)
        else:
            # safe: all kept vectors have the same length
            embeddings.append(np.vstack(conf_vecs).mean(axis=0))

    if skipped_dim:
        uniq = sorted(set(skipped_dim))
        print(
            f"Warning: skipped {len(skipped_dim)} conformer embeddings with mismatched "
            f"dimension. Affected molecule indices (unique): {uniq[:25]}" +
            (" ..." if len(uniq) > 25 else "")
        )

    return (embeddings if per_conformer else np.asarray(embeddings)), failed

def generate_maceoff_embedding(smiles: list, model_size: str ="medium", num_confs: int = 10, per_conformer: bool = False):
    """Generate MACE off embeddings for a list of SMILES strings. 
    Returns a list of embeddings and a list of indices for failed molecules.
    If multuple conformers are generated, the embedding for each conformer is returned as a list.
    Parameters:
    - smiles: list of SMILES strings to process.
    - model_size: size of the MACE off model to use ("small", "medium").
    - num_confs: number of conformers to generate for each molecule.
    - per_conformer: if True, returns a list of embeddings for each conformer; otherwise, returns a single embedding per molecule.
    Returns:
    tuple of (embeddings, failed) where:
    - embeddings: list of MACE off embeddings (either List[np.ndarray] or List[List[np.ndarray]] depending on num_confs).
    - failed: list of indices corresponding to SMILES strings that failed to process (e.g., invalid SMILES, embedding errors).
    """

    # generate optimized 3D conformers
    optimized_mols = generate_optimized_3d_conformers(smiles, num_confs=num_confs)
    
    # convert to ASE Atoms objects and compute MACE off embeddings
    embeddings, failed = compute_mace_off_embeddings(optimized_mols, model_size=model_size, per_conformer=per_conformer)
    # the embeddings is a list of lengths N conformers x N samples X embedding dimension
    # to pool comformer embeddings into one emb per molecule do

    if len(failed) > 0:
        print(f"Warning: {len(failed)} molecules failed to process. Indices: {failed}")

    return embeddings, failed

if __name__ == "__main__":
    # load test dataset
    datapath = Path("./01_DATASETS/CD/CD.csv")
    df = pd.read_csv(datapath)
    smiles_list = df["SMILES"].tolist()[:10]
    embeddings, failed_mols = generate_maceoff_embedding(smiles_list, model_size="large", num_confs=2)
    
    print("MACE off embeddings shape:", len(embeddings), "x", len(embeddings[0]) if embeddings else "N/A")

    pooled_embdings = [np.mean(emb, axis=0) if isinstance(emb, list) else emb for emb in embeddings]
    print("Pooled MACE off embeddings shape:", len(pooled_embdings), "x", len(pooled_embdings[0]) if pooled_embdings else "N/A")

    print("Failed molecules:", failed_mols)
    

