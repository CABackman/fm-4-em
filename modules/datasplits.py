"""Utilities for dataset splitting and split similarity analysis.

This module refactors the exploratory work from `06_NOTEBOOKS/testing.ipynb`
into reusable functions that can be called from preprocessing notebooks.

Main capabilities
-----------------
- Create splits:
  - extreme-value split on a descriptor column (e.g., bottom/top 10% of `kmf`)
  - random split
  - scaffold split using Bemis–Murcko scaffolds
- Quantify split difficulty by computing Train/Test Tanimoto similarities
  from Morgan fingerprints.
- Prepare/plot comparison data across multiple split strategies.

Notes
-----
- RDKit is required.
- Invalid SMILES can be optionally filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ArrayLike = np.ndarray


@dataclass(frozen=True)
class SimilaritySummary:
    """Summary statistics for a similarity matrix."""

    mean: float
    median: float
    min: float
    max: float


def descriptors_from_smiles(
    smiles_list: Iterable[str],
    descriptor_fn: Callable[[str], dict],
) -> pd.DataFrame:
    """Compute per-SMILES descriptors.

    Parameters
    ----------
    smiles_list
        Iterable of SMILES strings.
    descriptor_fn
        Function mapping a SMILES -> dict of descriptors.

    Returns
    -------
    pd.DataFrame
        One row per SMILES. Column names are descriptor keys.

    Raises
    ------
    ValueError
        If `descriptor_fn` returns non-dict values.
    """

    descs = []
    for s in tqdm(smiles_list, desc="Computing descriptors"):
        d = descriptor_fn(s)
        if not isinstance(d, dict):
            raise ValueError("descriptor_fn must return a dict per SMILES")
        descs.append(d)

    return pd.DataFrame(descs)


def split_by_extremes(
    df: pd.DataFrame,
    column: str,
    frac: float = 0.10,
    sort_kind: str = "mergesort",
) -> tuple[pd.Index, pd.Index]:
    """Split indices into Train/Test by taking bottom and top `frac` of `column`.

    This mirrors the notebook logic:
      test = bottom k union top k
      train = all others

    Parameters
    ----------
    df
        DataFrame containing the column.
    column
        Column to sort by.
    frac
        Fraction (0, 0.5] used for bottom and top selection.
    sort_kind
        Pandas sort kind.

    Returns
    -------
    (train_idx, test_idx)
        Indices into `df`.
    """

    if not (0 < frac <= 0.5):
        raise ValueError("frac must be in (0, 0.5]")

    s = df[column].sort_values(kind=sort_kind)
    n = len(s)
    k = int(np.floor(frac * n))
    if k < 1:
        raise ValueError("frac too small: results in empty test set")

    bottom_idx = s.iloc[:k].index
    top_idx = s.iloc[-k:].index

    test_idx = bottom_idx.union(top_idx)
    train_idx = df.index.difference(test_idx)
    return train_idx, test_idx



def random_split_indices(
    n_samples: int,
    test_size: int | float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random train/test indices using sklearn.model_selection.train_test_split.
    Parameters
    ----------
    n_samples
        Total number of samples.
    test_size
        If float, interpreted as fraction in (0, 1). If int, number of test samples.
    random_state
        Seed.

    Returns
    -------
    (train_idx, test_idx)
        Numpy integer arrays.
    """

    indices = np.arange(n_samples, dtype=int)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def scaffold_smiles(
    smiles: str,
    include_chirality: bool = False,
) -> str | None:
    """Compute Bemis–Murcko scaffold SMILES for a molecule."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )


def group_by_scaffold(
    smiles_list: Iterable[str],
    include_chirality: bool = False,
    drop_invalid: bool = True,
) -> dict[str, list[str]]:
    """Group SMILES by Bemis–Murcko scaffold."""

    groups: dict[str, list[str]] = {}
    for smi in smiles_list:
        scaff = scaffold_smiles(smi, include_chirality=include_chirality)
        if scaff is None:
            if drop_invalid:
                continue
            scaff = "<INVALID_SMILES>"
        groups.setdefault(scaff, []).append(smi)
    return groups


def split_by_scaffold_group(
    smiles_list: Iterable[str],
    pick: Literal["largest", "second_largest"] = "second_largest",
    include_chirality: bool = False,
    drop_invalid: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, list[int]]]:
    """Create a scaffold split by holding out one scaffold group as the test set.

    The notebook specifically held out the *second largest* scaffold group.

    Returns
    -------
    (train_idx, test_idx, heldout_scaffold, groups)
        train_idx/test_idx are numpy integer arrays of indices into the
        *input* smiles_list (after filtering invalids if drop_invalid=True).
        groups maps scaffold -> list of indices.
    """

    smiles = list(smiles_list)

    groups = {}
    for i, smi in enumerate(smiles):
        scaff = scaffold_smiles(smi, include_chirality=include_chirality)
        if scaff is None:
            if drop_invalid:
                continue
            scaff = "<INVALID_SMILES>"
        groups.setdefault(scaff, []).append(i)

    if len(groups) < 2:
        raise ValueError("Need at least 2 scaffold groups to do this split")

    items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    if pick == "largest":
        heldout_scaffold, test_idx_list = items[0]
    elif pick == "second_largest":
        heldout_scaffold, test_idx_list = items[1]
    else:
        raise ValueError("pick must be 'largest' or 'second_largest'")

    return heldout_scaffold, test_idx_list


def _valid_mols_from_smiles(
    smiles_list: Iterable[str],
    drop_invalid: bool = True,
) -> list[Chem.Mol]:
    mols: list[Chem.Mol] = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            if drop_invalid:
                continue
            raise ValueError(f"Invalid SMILES encountered: {s!r}")
        mols.append(m)
    return mols


def morgan_fingerprints_from_smiles(
    smiles_list: Iterable[str],
    radius: int = 3,
    fp_size: int = 2048,
    drop_invalid: bool = True,
):
    """Compute RDKit Morgan fingerprints for a SMILES list."""

    mols = _valid_mols_from_smiles(smiles_list, drop_invalid=drop_invalid)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return [gen.GetFingerprint(m) for m in mols]


def tanimoto_matrix(
    fps1: list,
    fps2: list,
) -> np.ndarray:
    """Compute an (n1 x n2) matrix of Tanimoto similarities."""

    n1 = len(fps1)
    n2 = len(fps2)
    sims = np.zeros((n1, n2), dtype=float)
    for i in range(n1):
        sims[i] = np.array([DataStructs.TanimotoSimilarity(fps1[i], f) for f in fps2])
    return sims


def summarize_similarity(sim_matrix: np.ndarray) -> SimilaritySummary:
    """Compute summary statistics for a similarity matrix."""

    return SimilaritySummary(
        mean=float(sim_matrix.mean()),
        median=float(np.median(sim_matrix)),
        min=float(sim_matrix.min()),
        max=float(sim_matrix.max()),
    )


def evaluate_split_similarity(
    train_smiles: Iterable[str],
    test_smiles: Iterable[str],
    radius: int = 3,
    fp_size: int = 2048,
    drop_invalid: bool = True,
) -> tuple[np.ndarray, SimilaritySummary]:
    """Compute Train/Test similarity matrix and summary statistics."""

    train_fps = morgan_fingerprints_from_smiles(
        train_smiles,
        radius=radius,
        fp_size=fp_size,
        drop_invalid=drop_invalid,
    )
    test_fps = morgan_fingerprints_from_smiles(
        test_smiles,
        radius=radius,
        fp_size=fp_size,
        drop_invalid=drop_invalid,
    )

    sims = tanimoto_matrix(train_fps, test_fps)
    return sims, summarize_similarity(sims)


def top_n_similarities_per_train(
    sim_matrix: np.ndarray,
    n: int = 5,
    flatten: bool = True,
) -> np.ndarray:
    """For each train point, take the top-n similarities to the test set.

    This reproduces the notebook computation used for plotting.

    Parameters
    ----------
    sim_matrix
        Train x Test similarity matrix.
    n
        Top-n count per train row.
    flatten
        If True, returns a flat array of length (n_train*n). If False, returns
        an (n_train x n) array.
    """

    if sim_matrix.ndim != 2:
        raise ValueError("sim_matrix must be 2D")
    if n < 1:
        raise ValueError("n must be >= 1")

    n = min(n, sim_matrix.shape[1])
    top_n = np.sort(np.partition(sim_matrix, -n, axis=1)[:, -n:], axis=1)
    return top_n.flatten() if flatten else top_n


def build_similarity_comparison_frame(
    *,
    random_top_n: np.ndarray,
    extreme_top_n: np.ndarray,
    scaffold_top_n: np.ndarray,
    labels: tuple[str, str, str] = ("Random Split", "Extreme Split", "Scaffold Split"),
) -> pd.DataFrame:
    """Create a long-form dataframe for comparing similarity distributions."""

    a, b, c = labels

    df = pd.DataFrame(
        {
            a: pd.Series(random_top_n, dtype=float),
            b: pd.Series(extreme_top_n, dtype=float),
            c: pd.Series(scaffold_top_n, dtype=float),
        }
    )

    melt = df.melt(var_name="Split Method", value_name="Tanimoto Similarity")
    melt = melt.dropna()
    return melt
