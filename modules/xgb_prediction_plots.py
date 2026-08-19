"""Notebook-friendly plots for pooled XGBoost test-set predictions."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATASETS = (r"$\rho_{cryst}$", "HOF", "IS")
DEFAULT_EMBEDDINGS = ("QU", "MACE OFF")
POSTER_FONT_SIZE = 36
UNITS_MAP = {
    r"$\rho_{cryst}$": r"g/cm$^3$",
    "HOF": "kJ/mol",
    "IS": r"$\ln{(J)}$",
    }


def aggregate_seed_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated seed predictions to one row per dataset/embedding/compound."""
    required = {"dataset", "embedding", "compound_id", "y_true", "y_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    group_cols = ["dataset", "embedding", "compound_id"]
    grouped = df.groupby(group_cols, as_index=False, sort=False, observed=True)

    agg_df = grouped.agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        y_pred_std=("y_pred", "std"),
        n_predictions=("y_pred", "size"),
    )
    agg_df["y_pred_std"] = agg_df["y_pred_std"].fillna(0.0)

    extra_cols = [
        col
        for col in df.columns
        if col not in set(group_cols + ["split_seed", "model_seed", "y_true", "y_pred"])
    ]
    for col in extra_cols:
        first_values = grouped[col].first()[col].to_numpy()
        nunique = grouped[col].nunique(dropna=False)[col].to_numpy()
        if np.all(nunique <= 1):
            agg_df[col] = first_values

    return agg_df


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 for paired regression predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_residual_grid(
    df: pd.DataFrame,
    *,
    use_aggregated: bool = True,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    embeddings: Sequence[str] = DEFAULT_EMBEDDINGS,
    figsize: tuple[float, float] = (24, 16),
    smooth_frac: float = 0.18,
    show_pred_std: bool = False,
    point_size: float = 18,
    point_alpha: float = 0.35,
):
    """Plot residuals versus true values with embeddings as rows and datasets as columns."""
    plot_df = aggregate_seed_predictions(df) if use_aggregated else df.copy()
    plot_df["residual"] = plot_df["y_pred"] - plot_df["y_true"]

    fig, axes = plt.subplots(
        len(embeddings),
        len(datasets),
        figsize=figsize,
        sharex="col",
        sharey="col",
        constrained_layout=True,
        squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        dataset_df = plot_df[plot_df["dataset"] == dataset]
        if not dataset_df.empty:
            x_min, x_max = _padded_limits(dataset_df["y_true"])
            y_min, y_max = _padded_limits(dataset_df["residual"], include_zero=True)
        else:
            x_min, x_max, y_min, y_max = 0.0, 1.0, -1.0, 1.0

        for row, embedding in enumerate(embeddings):
            ax = axes[row, col]
            panel_df = dataset_df[dataset_df["embedding"] == embedding].copy()

            ax.axhline(0, color="0.25", linewidth=1.0, linestyle="--", zorder=1)
            if not panel_df.empty:
                if show_pred_std and "y_pred_std" in panel_df.columns:
                    ax.errorbar(
                        panel_df["y_true"],
                        panel_df["residual"],
                        yerr=panel_df["y_pred_std"],
                        fmt="none",
                        ecolor="0.55",
                        elinewidth=0.7,
                        alpha=0.22,
                        zorder=2,
                    )
                ax.scatter(
                    panel_df["y_true"],
                    panel_df["residual"],
                    s=point_size,
                    alpha=point_alpha,
                    color="#2f6f9f",
                    edgecolors="none",
                    zorder=3,
                )
                _plot_rolling_mean(ax, panel_df["y_true"], panel_df["residual"], smooth_frac)
                metrics = regression_metrics(panel_df["y_true"], panel_df["y_pred"])
                _annotate_metrics(ax, f"MAE = {metrics['MAE']:.3g}\nRMSE = {metrics['RMSE']:.3g}")
            else:
                _annotate_metrics(ax, "No data")

            ax.set_title(f"{_dataset_label(dataset)} | {embedding}", fontsize=POSTER_FONT_SIZE)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=POSTER_FONT_SIZE)
            ax.set_xlabel(_axis_label("True value", dataset), fontsize=POSTER_FONT_SIZE)
            ax.set_ylabel(_axis_label("Prediction residual", dataset), fontsize=POSTER_FONT_SIZE)

    return fig


def plot_parity_grid(
    df: pd.DataFrame,
    *,
    color_by: str | None = None,
    use_aggregated: bool = True,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    embeddings: Sequence[str] = DEFAULT_EMBEDDINGS,
    figsize: tuple[float, float] = (24, 16),
    show_pred_std: bool = False,
    point_size: float = 18,
    point_alpha: float = 0.45,
    cmap: str = "viridis",
):
    """Plot predicted versus true values with embeddings as rows and datasets as columns."""
    plot_df = aggregate_seed_predictions(df) if use_aggregated else df.copy()
    if color_by is not None and color_by not in plot_df.columns:
        raise ValueError(f"color_by={color_by!r} is not a column in the plotting frame.")

    fig, axes = plt.subplots(
        len(embeddings),
        len(datasets),
        figsize=figsize,
        sharex="col",
        sharey="col",
        constrained_layout=True,
        squeeze=False,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)

    color_mappable = None
    if color_by is not None:
        color_values = pd.to_numeric(plot_df[color_by], errors="coerce")
        finite_colors = color_values[np.isfinite(color_values)]
        vmin = float(finite_colors.min()) if len(finite_colors) else None
        vmax = float(finite_colors.max()) if len(finite_colors) else None
    else:
        vmin = vmax = None

    for col, dataset in enumerate(datasets):
        dataset_df = plot_df[plot_df["dataset"] == dataset]
        if not dataset_df.empty:
            axis_values = pd.concat([dataset_df["y_true"], dataset_df["y_pred"]])
            lim_min, lim_max = _padded_limits(axis_values)
        else:
            lim_min, lim_max = 0.0, 1.0

        for row, embedding in enumerate(embeddings):
            ax = axes[row, col]
            panel_df = dataset_df[dataset_df["embedding"] == embedding].copy()

            ax.plot([lim_min, lim_max], [lim_min, lim_max], color="0.25", linewidth=1.0, zorder=1)
            if not panel_df.empty:
                if show_pred_std and "y_pred_std" in panel_df.columns:
                    ax.errorbar(
                        panel_df["y_true"],
                        panel_df["y_pred"],
                        yerr=panel_df["y_pred_std"],
                        fmt="none",
                        ecolor="0.55",
                        elinewidth=0.7,
                        alpha=0.22,
                        zorder=2,
                    )
                if color_by is None:
                    ax.scatter(
                        panel_df["y_true"],
                        panel_df["y_pred"],
                        s=point_size,
                        alpha=point_alpha,
                        color="#2f6f9f",
                        edgecolors="none",
                        zorder=3,
                    )
                else:
                    color_mappable = ax.scatter(
                        panel_df["y_true"],
                        panel_df["y_pred"],
                        c=panel_df[color_by],
                        s=point_size,
                        alpha=point_alpha,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolors="none",
                        zorder=3,
                    )

                metrics = regression_metrics(panel_df["y_true"], panel_df["y_pred"])
                _annotate_metrics(
                    ax,
                    f"R2 = {metrics['R2']:.3g}\nMAE = {metrics['MAE']:.3g}\nRMSE = {metrics['RMSE']:.3g}",
                )
            else:
                _annotate_metrics(ax, "No data")

            ax.set_title(f"{_dataset_label(dataset)} | {embedding}", fontsize=POSTER_FONT_SIZE)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            _set_matching_ticks(ax, lim_min, lim_max)
            ax.set_box_aspect(1)
            ax.tick_params(labelsize=POSTER_FONT_SIZE)
            ax.set_xlabel(_axis_label("True value", dataset), fontsize=POSTER_FONT_SIZE)
            ax.set_ylabel(_axis_label("Predicted value", dataset), fontsize=POSTER_FONT_SIZE)

    if color_mappable is not None:
        colorbar = fig.colorbar(color_mappable, ax=axes.ravel().tolist(), shrink=0.82)
        colorbar.set_label(color_by, fontsize=POSTER_FONT_SIZE)
        colorbar.ax.tick_params(labelsize=POSTER_FONT_SIZE)

    return fig


def _dataset_label(label: str) -> str:
    """Return a display label that lets Matplotlib render rho as mathtext."""
    if label in {r"\rho", "rho", "ρ", r"$\rho$", r"$\rho_{cryst}$"}:
        return r"$\rho_{cryst}$"
    return str(label)


def _dataset_units(label: str) -> str | None:
    """Return the display units for a dataset label, if known."""
    return UNITS_MAP.get(_dataset_label(label))


def _axis_label(label: str, dataset: str) -> str:
    units = _dataset_units(dataset)
    return f"{label} ({units})" if units else label


def _padded_limits(values, *, include_zero: bool = False, pad_frac: float = 0.06) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if len(values) == 0:
        return 0.0, 1.0

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        pad = max(abs(vmin) * pad_frac, 1.0)
    else:
        pad = (vmax - vmin) * pad_frac
    return vmin - pad, vmax + pad


def _set_matching_ticks(ax, lim_min: float, lim_max: float) -> None:
    """Use the same visible major tick positions on both parity axes."""
    ticks = np.asarray(ax.get_xticks())
    ticks = ticks[(ticks >= lim_min) & (ticks <= lim_max)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def _plot_rolling_mean(ax, x, y, smooth_frac: float) -> None:
    smooth_df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(smooth_df) < 3:
        return

    smooth_df = smooth_df.sort_values("x")
    window = max(3, int(np.ceil(len(smooth_df) * smooth_frac)))
    if window % 2 == 0:
        window += 1
    smooth_df["trend"] = smooth_df["y"].rolling(window=window, center=True, min_periods=1).mean()

    ax.plot(
        smooth_df["x"],
        smooth_df["trend"],
        color="#c44e52",
        linewidth=2.0,
        zorder=4,
    )


def _annotate_metrics(ax, text: str) -> None:
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=POSTER_FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
