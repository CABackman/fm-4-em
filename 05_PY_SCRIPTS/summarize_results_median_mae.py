"""Summarize embedding/model performance by median MAE.

Reads:
  - 07_OUTPUTS/20260321/compiled/clean_results.csv
  - 07_OUTPUTS/20260321/compiled/dummy_baselines.csv

Produces:
  - Printed markdown tables to stdout

Notes
-----
- Primary metric: test_MAE (lower is better).
- Computes relative MAE = test_MAE / dummy_mae for each dataset+split.
- Provides two complementary views:
  1) winners by (dataset × split), aggregating across models
  2) winners by (dataset × split × model), i.e., split out by model and median across seeds
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "07_OUTPUTS/20260321/compiled/clean_results.csv"
DEFAULT_DUMMY = ROOT / "07_OUTPUTS/20260321/compiled/dummy_baselines.csv"


def load(results_path: Path, dummy_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)

    # Normalize split naming (some dummy rows include random_seedXX)
    df["split_method"] = df["split_method"].replace({"RAND": "random", "BUTINA": "butina"})

    dummy = pd.read_csv(dummy_path)
    dummy["split"] = dummy["split"].replace({"RAND": "random", "BUTINA": "butina"})
    # Collapse random_seedXX to random
    dummy["split"] = dummy["split"].str.replace(r"^random_seed\d+$", "random", regex=True)

    # Keep one dummy per (dataset, split)
    dummy = (
        dummy.groupby(["dataset", "split"], as_index=False)["dummy_mae"]
        .median()
        .rename(columns={"split": "split_method"})
    )

    out = df.merge(dummy, on=["dataset", "split_method"], how="left")
    if out["dummy_mae"].isna().any():
        missing = out.loc[out["dummy_mae"].isna(), ["dataset", "split_method"]].drop_duplicates()
        raise ValueError(
            "Missing dummy_mae for some dataset/split pairs:\n" + missing.to_string(index=False)
        )

    out["rel_mae"] = out["test_MAE"] / out["dummy_mae"]
    return out


def best_by_dataset_split_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset+split+embedding: median MAE across (seed, model)."""
    agg = (
        df.groupby(["dataset", "split_method", "embedding"], as_index=False)
        .agg(
            median_mae=("test_MAE", "median"),
            median_rel_mae=("rel_mae", "median"),
            n=("test_MAE", "size"),
        )
    )
    return agg


def best_by_dataset_split_model_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset+split+model+embedding: median MAE across seeds."""
    agg = (
        df.groupby(["dataset", "split_method", "model", "embedding"], as_index=False)
        .agg(
            median_mae=("test_MAE", "median"),
            median_rel_mae=("rel_mae", "median"),
            n=("test_MAE", "size"),
        )
    )
    return agg


def pick_winners(agg: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Pick the best embedding per group using median MAE.

    Parameters
    ----------
    agg:
        A dataframe that contains at least: embedding, median_mae, median_rel_mae.
    group_cols:
        Columns that define the grouping (e.g., [dataset, split_method] or
        [dataset, split_method, model]).
    """

    def _winner(g: pd.DataFrame) -> pd.Series:
        g2 = g.sort_values(["median_mae", "median_rel_mae", "embedding"], ascending=[True, True, True])
        best = g2.iloc[0]
        second = g2.iloc[1] if len(g2) > 1 else None

        gap = None
        gap_rel = None
        if second is not None:
            gap = float(second["median_mae"] - best["median_mae"])
            gap_rel = float(second["median_rel_mae"] - best["median_rel_mae"])

        # Crude confidence heuristic: relative gap threshold
        # (no CI available here; this is just a prioritization hint)
        conf = "mixed"
        if second is None:
            conf = "strong"
        else:
            if best["median_rel_mae"] > 0:
                ratio = float(second["median_rel_mae"] / best["median_rel_mae"])
                if ratio >= 1.05:
                    conf = "strong"
                elif ratio <= 1.02:
                    conf = "weak"

        return pd.Series(
            {
                "winner_embedding": best["embedding"],
                "winner_median_mae": best["median_mae"],
                "winner_median_rel_mae": best["median_rel_mae"],
                "runnerup_embedding": None if second is None else second["embedding"],
                "runnerup_median_mae": None if second is None else second["median_mae"],
                "gap_mae": gap,
                "gap_rel_mae": gap_rel,
                "confidence": conf,
            }
        )

    winners = agg.groupby(group_cols, as_index=False).apply(_winner)
    winners = winners.reset_index(drop=True)
    return winners


def embedding_ranking(agg: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Rank embeddings within each group; summarize top1 count and median rank.

    Examples
    --------
    - group_cols=["dataset", "split_method"]
    - group_cols=["dataset", "split_method", "model"]
    """
    ranked = agg.copy()
    ranked["rank"] = ranked.groupby(group_cols)["median_mae"].rank(method="average")

    summary = (
        ranked.groupby("embedding", as_index=False)
        .agg(
            top1_count=("rank", lambda s: int((s == 1).sum())),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            n_cases=("rank", "size"),
        )
        .sort_values(["top1_count", "median_rank", "mean_rank"], ascending=[False, True, True])
    )
    return summary


def model_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each dataset+split; aggregate across embeddings+seeds by median."""
    # First aggregate per dataset/split/model: median MAE across embeddings+seeds
    m = (
        df.groupby(["dataset", "split_method", "model"], as_index=False)
        .agg(median_mae=("test_MAE", "median"), median_rel_mae=("rel_mae", "median"), n=("test_MAE", "size"))
    )
    m["rank"] = m.groupby(["dataset", "split_method"])["median_mae"].rank(method="average")

    summary = (
        m.groupby("model", as_index=False)
        .agg(
            top1_count=("rank", lambda s: int((s == 1).sum())),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            n_cases=("rank", "size"),
        )
        .sort_values(["top1_count", "median_rank", "mean_rank"], ascending=[False, True, True])
    )
    return summary


def flips(winners: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Find entities where winner embedding changes across splits.

    Parameters
    ----------
    winners:
        Output of pick_winners with at least columns:
        - split_method
        - winner_embedding
    index_cols:
        Columns to define the entity to check for flips (e.g., ["dataset"] or
        ["dataset", "model"]).
    """
    piv = winners.pivot(index=index_cols, columns="split_method", values="winner_embedding")
    out_rows: list[dict] = []

    for idx, row in piv.iterrows():
        vals = [v for v in row.dropna().tolist()]
        if len(set(vals)) > 1:
            if not isinstance(idx, tuple):
                idx = (idx,)
            rec = {col: val for col, val in zip(index_cols, idx)}
            rec.update(row.to_dict())
            out_rows.append(rec)

    return pd.DataFrame(out_rows)


def to_markdown_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    # Use pandas markdown if available
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        # fallback
        return df.to_string(index=False)


def main(results_path: Path = DEFAULT_RESULTS, dummy_path: Path = DEFAULT_DUMMY) -> None:
    df = load(results_path, dummy_path)

    # Aggregations
    agg_emb = best_by_dataset_split_embedding(df)
    agg_emb_model = best_by_dataset_split_model_embedding(df)

    # Winners
    winners_ds_split = pick_winners(agg_emb, group_cols=["dataset", "split_method"])
    winners_ds_split_model = pick_winners(agg_emb_model, group_cols=["dataset", "split_method", "model"])

    # Rankings
    emb_rank_overall = embedding_ranking(agg_emb, group_cols=["dataset", "split_method"])
    emb_rank_by_model = (
        agg_emb_model.assign(rank=lambda d: d.groupby(["dataset", "split_method", "model"])["median_mae"].rank(method="average"))
        .groupby(["model", "embedding"], as_index=False)
        .agg(
            top1_count=("rank", lambda s: int((s == 1).sum())),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            n_cases=("rank", "size"),
        )
        .sort_values(["model", "top1_count", "median_rank", "mean_rank"], ascending=[True, False, True, True])
    )

    mod_rank = model_ranking(df)

    # Flips
    flip_ds = flips(winners_ds_split, index_cols=["dataset"])
    flip_ds_model = flips(winners_ds_split_model, index_cols=["dataset", "model"])

    # Pretty ordering
    split_order = pd.CategoricalDtype(["random", "butina", "OB"], ordered=True)
    winners_ds_split["split_method"] = winners_ds_split["split_method"].astype(split_order)
    winners_ds_split = winners_ds_split.sort_values(["dataset", "split_method"])  # type: ignore

    winners_ds_split_model["split_method"] = winners_ds_split_model["split_method"].astype(split_order)
    winners_ds_split_model = winners_ds_split_model.sort_values(["dataset", "model", "split_method"])  # type: ignore

    print("# Winners by dataset × split (median over seeds+models; MAE lower is better)\n")
    print(to_markdown_table(winners_ds_split[[
        "dataset",
        "split_method",
        "winner_embedding",
        "winner_median_mae",
        "winner_median_rel_mae",
        "runnerup_embedding",
        "runnerup_median_mae",
        "gap_mae",
        "confidence",
    ]]))

    print("\n# Winners by dataset × split × model (median over seeds; MAE lower is better)\n")
    print(to_markdown_table(winners_ds_split_model[[
        "dataset",
        "split_method",
        "model",
        "winner_embedding",
        "winner_median_mae",
        "winner_median_rel_mae",
        "runnerup_embedding",
        "runnerup_median_mae",
        "gap_mae",
        "confidence",
    ]]))

    print("\n# Embedding ranking (across dataset×split cases; based on median MAE ranks)\n")
    print(to_markdown_table(emb_rank_overall))

    print("\n# Embedding ranking split by model (across dataset×split×model cases; based on median MAE ranks)\n")
    print(to_markdown_table(emb_rank_by_model))

    print("\n# Model ranking (across dataset×split cases; based on median MAE ranks)\n")
    print(to_markdown_table(mod_rank))

    print("\n# Datasets with ranking flips across splits (winner differs; aggregated over models)\n")
    if len(flip_ds) == 0:
        print("(none)")
    else:
        cols = ["dataset"] + [c for c in ["random", "butina", "OB"] if c in flip_ds.columns]
        print(to_markdown_table(flip_ds[cols]))

    print("\n# Dataset×model pairs with ranking flips across splits (winner differs)\n")
    if len(flip_ds_model) == 0:
        print("(none)")
    else:
        cols = ["dataset", "model"] + [c for c in ["random", "butina", "OB"] if c in flip_ds_model.columns]
        print(to_markdown_table(flip_ds_model[cols]))


if __name__ == "__main__":
    main()
