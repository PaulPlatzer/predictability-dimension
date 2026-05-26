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
