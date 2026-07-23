#!/usr/bin/env python3

from pathlib import Path
from typing import Iterable, Optional
import sys
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]       # .../01_FOUNDATION_MODELS
molclr_root = ROOT / "01_FOUNDATION_MODELS/04_FM_CPS"                    # .../01_FOUNDATION_MODELS/04_FM_CPS
sys.path.insert(0, str(molclr_root))
from MolCLR.models.ginet_molclr import GINet



# Feature vocabularies
ATOM_LIST = list(range(1, 119))
ATOM_MAP = {a: i for i, a in enumerate(ATOM_LIST)}
ATOM_UNKNOWN = len(ATOM_LIST)  # mask/unknown index (MolCLR uses 119 atom types incl mask)

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
CHIRALITY_MAP = {c: i for i, c in enumerate(CHIRALITY_LIST)}

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BOND_MAP = {b: i for i, b in enumerate(BOND_LIST)}

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
BONDDIR_MAP = {d: i for i, d in enumerate(BONDDIR_LIST)}


# SMILES → PyG graph
def smiles_to_data(smiles: str, add_hs: bool = False) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric molecular graph.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.
    add_hs : bool, optional
        Whether to explicitly add hydrogen atoms to the molecule.
        Default is False.

    Returns
    -------
    torch_geometric.data.Data
        A graph object containing:
        - x: Node feature tensor of shape (num_atoms, 2)
        - edge_index: Edge indices of shape (2, num_edges)
        - edge_attr: Edge feature tensor of shape (num_edges, 2)

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if add_hs:
        mol = Chem.AddHs(mol)

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(
            [
                ATOM_MAP.get(atom.GetAtomicNum(), ATOM_UNKNOWN),
                CHIRALITY_MAP.get(atom.GetChiralTag(), 0),  # map unknown/other -> unspecified
            ]
        )

    x = torch.tensor(atom_features, dtype=torch.long)

    #  Edge features 
    row, col, edge_attr = [], [], []

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        bt = BOND_MAP.get(bond.GetBondType(), 0)
        bd = BONDDIR_MAP.get(bond.GetBondDir(), 0)

        row.extend([a, b])
        col.extend([b, a])
        edge_attr.extend([[bt, bd], [bt, bd]])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# DataLoader builder
def build_molclr_dataloader(
    smiles: Iterable[str],
    y: Optional[Iterable] = None,
    batch_size: int = 64,
    task: Optional[str] = None,
    add_hs: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a PyTorch Geometric DataLoader for MolCLR inference or training.

    Converts SMILES strings into molecular graphs and optionally attaches
    labels for classification or regression tasks.

    Parameters
    ----------
    smiles : Iterable[str]
        Iterable of SMILES strings.
    y : Iterable, optional
        Target labels corresponding to each SMILES string.
    batch_size : int, optional
        Number of graphs per batch. Default is 64.
    task : {"classification", "regression"}, optional
        Type of supervised task. Required if `y` is provided.
    add_hs : bool, optional
        Whether to add explicit hydrogen atoms. Default is False.
    shuffle : bool, optional
        Whether to shuffle the dataset. Default is False.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 0.

    Returns
    -------
    torch_geometric.loader.DataLoader
        A DataLoader yielding batched molecular graphs.

    Raises
    ------
    ValueError
        If `task` is invalid or labels are provided without a task type.
    """
    graphs = []

    for i, s in enumerate(smiles):
        data = smiles_to_data(s, add_hs=add_hs)

        if y is not None:
            if task == "classification":
                data.y = torch.tensor(int(y[i]), dtype=torch.long).view(1, -1)
            elif task == "regression":
                data.y = torch.tensor(float(y[i]), dtype=torch.float).view(1, -1)
            else:
                raise ValueError("task must be 'classification' or 'regression'")

        graphs.append(data)

    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )


# MolCLR embedding
def mol_clr_embed(smiles: Iterable[str], ckpt_path: str | None = None) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = Path(ckpt_path)
    ckpt_path = ROOT / "01_FOUNDATION_MODELS" / ckpt_path
    print("ckpt path:", ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    model = GINet(
        num_layer=5,
        emb_dim=300,
        feat_dim=512,   # must match the checkpoint architecture
        drop_ratio=0.0,
        pool="mean",
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    dataloader = build_molclr_dataloader(smiles)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            h, _proj = model(batch)   # <-- for property prediction use h
            embeddings.append(h.cpu())

    emb = torch.cat(embeddings, dim=0)
    return emb

