from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from chemprop import data, featurizers, models, nn
from chemprop.data.collate import collate_batch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_CKPT_PATH = "04_FM_Embeddings/04_FM_CPS/chemeleon_mp.pt"


def chemeleon_embedding(
    smiles: Iterable[str],
    ys: Iterable[float],
    ckpt_path: str | None = None,
) -> np.ndarray:
    """Compute CheMeleon (Chemprop MPNN) fingerprints for a list of SMILES.

    Args:
        smiles: SMILES strings.
        ys: Dummy/target values required by Chemprop datapoints.
        ckpt_path: Path to the CheMeleon checkpoint.

    Returns:
        Array of shape (n_samples, embedding_dim).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or DEFAULT_CKPT_PATH

    # Materialize to avoid surprises if callers pass generators.
    smiles = list(smiles)
    if ys is None:
        ys = [0.0] * len(smiles)  # Dummy values if not provided.
    ys = list(ys)

    # Build dataset
    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, ys, strict=False)
    ]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer=featurizer)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # Load model checkpoint (support common Chemprop checkpoint layouts).
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CheMeleon checkpoint not found: {ckpt_path}") from e

    hyper_parameters = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    state_dict = (
        checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )

    mp = nn.BondMessagePassing(**hyper_parameters)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[nn.metrics.RMSE()])
    mpnn.load_state_dict(state_dict, strict=False)
    mpnn.to(device).float().eval()
    

    embeds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating CheMeleon embeddings"):
            bmg = batch.bmg
            bmg.to(device)

            V_d = batch.V_d.to(device) if isinstance(batch.V_d, torch.Tensor) else None
            X_d = batch.X_d.to(device) if isinstance(batch.X_d, torch.Tensor) else None
            
            z = mpnn.fingerprint(bmg, V_d, X_d)
            embeds.append(z.cpu())

    return torch.cat(embeds, dim=0).numpy()
