"""Plotting helpers for notebook and script result displays."""

from __future__ import annotations

from math import floor, log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import linregress


def round_to_n(x: float, n: int) -> float:
    """Round to n significant digits, keeping zero readable."""

    if x == 0 or not np.isfinite(x):
        return float(x)
    return round(float(x), -int(floor(log10(abs(x)))) + (n - 1))


def plot_tigge_vs_analogue_spread(
    tigge_spread: xr.DataArray,
    ana_spread: np.ndarray,
    horizons_days: list[int],
    output: "str | Path | None" = None,
) -> plt.Figure:
    """Scatter TIGGE spread vs analogue ensemble spread for each horizon.

    Parameters
    ----------
    tigge_spread:
        TIGGE spread DataArray with dims (time, step).
    ana_spread:
        Analogue spread array, shape (n_times, len(horizons_days)), aligned
        with tigge_spread.time.
    horizons_days:
        Lead times in days corresponding to the columns of ana_spread.
    """

    n_h = len(horizons_days)
    fig, axs = plt.subplots(1, n_h, figsize=(4 * n_h, 4), squeeze=False)

    for h_idx, h in enumerate(horizons_days):
        ax = axs[0, h_idx]
        step = np.timedelta64(h, "D")
        x = np.asarray(tigge_spread.sel(step=step)).squeeze()  # TIGGE spread
        y = ana_spread[:, h_idx]                                # analogue spread
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            ax.set_title(f"h={h}d — no data")
            continue
        reg = linregress(x[valid], y[valid])
        ax.scatter(x[valid], y[valid], alpha=0.25, s=8, rasterized=True)
        xline = np.array([x[valid].min(), x[valid].max()])
        ax.plot(xline, reg.slope * xline + reg.intercept, "r-", lw=1.5)
        ax.set_title(f"h={h}d   r={round_to_n(reg.rvalue, 2)}", fontsize=10)
        ax.set_xlabel("TIGGE spread")
        ax.set_ylabel("analogue spread")

    fig.suptitle("TIGGE spread vs analogue ensemble spread", size=13)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120)
    return fig


def plot_regression_coefficients(
    results: dict,
    estimator_label: str = "",
    output: "str | Path | None" = None,
) -> plt.Figure:
    """Plot regression coefficients a(h), b(h) and R²(h) vs lead time.

    Parameters
    ----------
    results:
        Output of ``regression.fit_horizon_regressions``.
    estimator_label:
        Name of the dimension estimator (for the title).
    """

    horizons = results["horizons"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].bar(horizons, results["coef_a"])
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_title("a(h) — dimension coefficient")
    axs[0].set_xlabel("lead time (days)")

    axs[1].bar(horizons, results["coef_b"])
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_title("b(h) — density proxy coefficient")
    axs[1].set_xlabel("lead time (days)")

    axs[2].plot(horizons, results["r2"], "o-")
    axs[2].set_ylim(0, max(0.05, float(np.nanmax(results["r2"])) * 1.1))
    axs[2].set_title("R²(h)")
    axs[2].set_xlabel("lead time (days)")

    title = "Regression: spread_growth = a·dim + b·density"
    if estimator_label:
        title += f"  [{estimator_label}]"
    fig.suptitle(title, size=12)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120)
    return fig


def align_spread_and_dimension(spread: xr.DataArray, dim: xr.Dataset) -> tuple[xr.DataArray, xr.Dataset]:
    """Align TIGGE spread times with dimension initial times."""

    intersection = np.sort(np.intersect1d(spread.time.data, dim.initial_time.data))
    return spread.sel(time=intersection), dim.sel(initial_time=intersection)


def plot_dimension_vs_spread(
    spread: xr.DataArray,
    dim: xr.Dataset,
    output: str | Path | None = None,
    title: str = "Dimension vs spread",
) -> plt.Figure:
    """Scatter every dimension estimate against TIGGE spread."""

    nrows = len(dim.data_vars)
    ncols = len(spread.step)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for row, name in enumerate(dim.data_vars):
        for col, step in enumerate(spread.step):
            ax = axs[row, col]
            x = spread.sel(step=step).data.squeeze()
            for K in dim.K:
                y = dim[name].sel(K=K).data.squeeze()
                reg = linregress(x, y)
                label = f"K={int(K)}, r={round_to_n(reg.rvalue, 2)}, p={reg.pvalue:.1e}"
                ax.scatter(x, y, label=label, alpha=0.3)
            ax.legend()
            ax.set_title(f"h={step.data // np.timedelta64(1, 'D')}")
            ax.set_ylabel(name)
            ax.set_xlabel("spread")

    fig.tight_layout()
    fig.suptitle(title, y=1.0, size=15)
    if output is not None:
        fig.savefig(output)
    return fig


def plot_dimension_vs_spread_growth(
    spread: xr.DataArray,
    dim: xr.Dataset,
    reference: str = "previous",
    steps: xr.DataArray | np.ndarray | None = None,
    output: str | Path | None = None,
) -> plt.Figure:
    """Scatter dimensions against spread growth ratios."""

    if steps is None:
        steps = spread.step[1:]

    nrows = len(dim.data_vars)
    ncols = len(steps)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for row, name in enumerate(dim.data_vars):
        for col, step in enumerate(steps):
            ax = axs[row, col]
            denominator_step = step - np.timedelta64(1, "D") if reference == "previous" else spread.step.isel(step=0)
            x = (spread.sel(step=step) / spread.sel(step=denominator_step)).data.squeeze()
            for K in dim.K:
                y = dim[name].sel(K=K).data.squeeze()
                reg = linregress(x, y)
                label = f"K={int(K)}, r={round_to_n(reg.rvalue, 2)}, p={reg.pvalue:.1e}"
                ax.scatter(x, y, label=label, alpha=0.3)
            ax.legend()
            ax.set_title(f"h={step.data // np.timedelta64(1, 'D')}")
            ax.set_ylabel(name)
            ax.set_xlabel("spread growth")

    subtitle = "spread(t)/spread(t-1)" if reference == "previous" else "spread(t)/spread(initial)"
    fig.tight_layout()
    fig.suptitle(f"Dimension vs {subtitle}", y=1.0, size=15)
    if output is not None:
        fig.savefig(output)
    return fig
