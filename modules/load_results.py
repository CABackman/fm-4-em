from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


_SPLIT_RE = re.compile(r"^split=(?P<method>[^_]+)_seed=(?P<seed>\d+)$")

def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _flatten_metrics(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    # Turns {"MAE": 0.1, "RMSE": 0.2} into {"train_MAE": 0.1, ...}
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[f"{prefix}_{k}"] = v
    return out


def _parse_summary_path(summary_path: Path) -> Dict[str, Any]:
    """
    Expected structure:
      07_OUTPUTS/<date>/<dataset>/<model>/<embedding>/<split_dir>/summary.json
    """
    # Make it resilient by searching relative to "07_OUTPUTS"
    parts = summary_path.parts
    try:
        i = parts.index("07_OUTPUTS")
    except ValueError:
        # fallback: assume last 6 parts match the layout
        i = max(0, len(parts) - 7)

    # Minimal safety: ensure enough parts exist
    # indices: i+1=date, i+2=dataset, i+3=model, i+4=embedding, i+5=split_dir
    meta = {
        "date": parts[i + 1] if len(parts) > i + 1 else None,
        "dataset_from_path": parts[i + 2] if len(parts) > i + 2 else None,
        "model_from_path": parts[i + 3] if len(parts) > i + 3 else None,
        "embedding_from_path": parts[i + 4] if len(parts) > i + 4 else None,
        "split_dir": parts[i + 5] if len(parts) > i + 5 else None,
        "summary_path": str(summary_path),
    }

    split_dir = meta.get("split_dir") or ""
    m = _SPLIT_RE.match(split_dir)
    if m:
        meta["split_method_from_path"] = m.group("method")
        meta["split_seed_from_path"] = _safe_int(m.group("seed"))
    else:
        meta["split_method_from_path"] = None
        meta["split_seed_from_path"] = None

    return meta


def load_results(
    outputs_root: str | Path = "07_OUTPUTS",
    pattern: str = "**/summary.json",
    date: str | int = "20260319",
) -> pd.DataFrame:
    outputs_root = Path(outputs_root) / str(date)
    rows: list[dict[str, Any]] = []

    for summary_path in outputs_root.glob(pattern):
        meta = _parse_summary_path(summary_path)

        with open(summary_path, "r") as f:
            try: 
                data = json.load(f)
            except:
                print(f"loading failed for {f}")
                continue

        row: Dict[str, Any] = {}
        row.update(meta)

        # Prefer JSON keys when present (they should be authoritative)
        row["dataset"] = data.get("dataset", row.get("dataset_from_path"))
        row["model"] = data.get("model", row.get("model_from_path"))
        row["embedding"] = data.get("embedding", row.get("embedding_from_path"))
        row["embedding_dim"] = data.get("embedding_dim")

        # Split info: JSON might store split="random" and split_seed=42
        row["split_method"] = data.get("split", row.get("split_method_from_path"))
        row["split_seed"] = data.get("split_seed", row.get("split_seed_from_path"))

        row["seed"] = data.get("seed")
        row["timestamp"] = data.get("timestamp")

        # Optional provenance
        for k in ["dataset_path", "embedding_path", "test_idx_path", "outdir"]:
            if k in data:
                row[k] = data[k]

        # Flatten metrics
        row.update(_flatten_metrics(data.get("train_metrics", {}), "train"))
        row.update(_flatten_metrics(data.get("test_metrics", {}), "test"))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Helpful derived identifiers
    if not df.empty:
        df["run_id"] = (
            df["date"].astype(str)
            + "|"
            + df["dataset"].astype(str)
            + "|"
            + df["model"].astype(str)
            + "|"
            + df["embedding"].astype(str)
            + "|"
            + df["split_method"].astype(str)
            + "|"
            + df["split_seed"].astype(str)
            + "|seed="
            + df["seed"].astype(str)
        )

    return df


def save_cache(
    df: pd.DataFrame,
    outputs_root: str | Path = "07_OUTPUTS",
    date: str | int = "20260319",
) -> Path:
    """
    Saves to: 07_OUTPUTS/<date>/_compiled/results.parquet
    """
    outdir = Path(outputs_root) / str(date) / "compiled"
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / "results.parquet"
    df.to_parquet(outpath, index=False)

    return outpath


if __name__ == "__main__":
    df = load_results("07_OUTPUTS", date="20260319")
    print(df.shape)
    outpath = save_cache(df, "07_OUTPUTS", date="20260319", filename="results")
    print("Saved:", outpath)