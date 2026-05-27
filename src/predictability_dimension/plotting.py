"""Plotting helpers for notebook and script result displays."""

from __future__ import annotations

from math import floor, log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
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


def plot_regression_sensitivity(
    results_by_k: dict,
    growth_type: str,
    estimator: str,
    horizons: list[int],
    x1_by_k: dict,
    x2: np.ndarray,
    y: np.ndarray,
    h_3d: int = 7,
    output: "str | Path | None" = None,
) -> plt.Figure:
    """4-panel figure: regression sensitivity to K + 3D scatter at one horizon.

    Panels:
    1. a(h) [dimension coefficient, full model] for each K
    2. b(h) [density coefficient, full model] for each K
    3. R²(h) — full model (solid) and density-only model (dashed) for each K
    4. 3D scatter (dim, density, spread_growth) at h=h_3d, K=50, with
       the two regression planes (full vs density-only).

    Parameters
    ----------
    results_by_k:
        {K: result_dict} where each result_dict is the output of
        ``regression.fit_horizon_regressions`` (contains 'full', 'reduced',
        'scaler', 'horizons').
    growth_type:
        'cumulative' or 'incremental' (for the figure title).
    estimator:
        Dimension estimator name (for the title).
    horizons:
        List of lead times in days.
    x1_by_k:
        {K: x1_array} with raw (unstandardised) dimension values.
    x2:
        Raw density proxy values, shape (n,).
    y:
        Target (spread growth) array, shape (n, len(horizons)).
    h_3d:
        Horizon used for the 3D panel.
    """

    K_list  = sorted(results_by_k)
    colors  = plt.cm.viridis(np.linspace(0, 0.9, len(K_list)))
    h_arr   = np.array(horizons)

    fig = plt.figure(figsize=(20, 4.5))
    gs  = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1.6], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3], projection="3d")

    # ── panels 1–3: sensitivity curves ───────────────────────────────────────
    for ki, K in enumerate(K_list):
        res = results_by_k[K]
        c   = colors[ki]
        lbl = f"K={K}"
        ax1.plot(h_arr, res["full"]["coef_a"],  "o-",  color=c, label=lbl)
        ax2.plot(h_arr, res["full"]["coef_b"],  "o-",  color=c, label=lbl)
        ax3.plot(h_arr, res["full"]["r2"],      "o-",  color=c, label=lbl + " full")
        ax3.plot(h_arr, res["reduced"]["r2"],   "o--", color=c, alpha=0.55)

    ax3.text(0.03, 0.04, "dashed = density only",
             transform=ax3.transAxes, fontsize=7, alpha=0.7)

    growth_label = ("spread(t+h)/spread(t)"   if growth_type == "cumulative"
                    else "spread(t+h)/spread(t+h-1)")

    for ax, ylabel in [
        (ax1, "a(h) — dimension coeff."),
        (ax2, "b(h) — density coeff."),
        (ax3, "R²(h)"),
    ]:
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.set_xticks(h_arr)

    # ── panel 4: 3D scatter at h=h_3d ────────────────────────────────────────
    K_3d   = 50 if 50 in results_by_k else K_list[len(K_list) // 2]
    h_idx  = list(horizons).index(h_3d)
    res_3d = results_by_k[K_3d]
    scaler = res_3d["scaler"]

    x1_raw = x1_by_k[K_3d]
    X_raw  = np.column_stack([x1_raw, x2])
    X_std  = scaler.transform(X_raw) if scaler is not None else X_raw
    x1_s, x2_s = X_std[:, 0], X_std[:, 1]
    y_h    = y[:, h_idx]

    valid  = np.isfinite(x1_s) & np.isfinite(x2_s) & np.isfinite(y_h)
    # Downsample for rendering speed
    step   = max(1, valid.sum() // 600)
    idx    = np.where(valid)[0][::step]
    ax4.scatter(x1_s[idx], x2_s[idx], y_h[idx],
                alpha=0.25, s=7, c="steelblue", depthshade=True)

    # Grid for regression planes
    x1_grid = np.linspace(np.nanpercentile(x1_s[valid], 1),
                           np.nanpercentile(x1_s[valid], 99), 14)
    x2_grid = np.linspace(np.nanpercentile(x2_s[valid], 1),
                           np.nanpercentile(x2_s[valid], 99), 14)
    XX, YY  = np.meshgrid(x1_grid, x2_grid)

    a  = res_3d["full"]["coef_a"][h_idx]
    bf = res_3d["full"]["coef_b"][h_idx]
    cf = res_3d["full"]["intercept"][h_idx]
    ax4.plot_surface(XX, YY, a * XX + bf * YY + cf,
                     alpha=0.30, color="tomato")

    br = res_3d["reduced"]["coef_b"][h_idx]
    cr = res_3d["reduced"]["intercept"][h_idx]
    ax4.plot_surface(XX, YY, br * YY + cr,
                     alpha=0.30, color="orange")

    ax4.set_xlabel("dim (std)", fontsize=8, labelpad=2)
    ax4.set_ylabel("density (std)", fontsize=8, labelpad=2)
    ax4.set_zlabel(f"growth h={h_3d}d", fontsize=8, labelpad=2)
    ax4.set_title(f"K={K_3d}, h={h_3d}d", fontsize=9)

    r2_full = res_3d["full"]["r2"][h_idx]
    r2_red  = res_3d["reduced"]["r2"][h_idx]
    ax4.legend(handles=[
        Patch(color="tomato",  alpha=0.7, label=f"dim+density  R²={r2_full:.3f}"),
        Patch(color="orange",  alpha=0.7, label=f"density only R²={r2_red:.3f}"),
    ], fontsize=7, loc="upper left")

    fig.suptitle(f"{estimator}  [{growth_label}]", size=11, y=1.01)

    if output is not None:
        fig.savefig(output, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ana_growth_vs_tigge_growth(
    tigge_spread: xr.DataArray,
    ana_spread: xr.DataArray,
    horizons_days: list[int],
    output: "str | Path | None" = None,
) -> plt.Figure:
    """Scatter analogue spread growth vs TIGGE spread growth for each horizon.

    Both growth ratios are cumulative: quantity(t+h) / quantity(t, h=0).

    Parameters
    ----------
    tigge_spread:
        TIGGE spread DataArray with dims (time, step).  Must include step=0.
    ana_spread:
        Analogue spread DataArray with dims (time, step).  Must include step=0.
    horizons_days:
        Lead times in days to plot.
    """

    step0 = np.timedelta64(0, "D")
    tigge0 = np.asarray(tigge_spread.sel(step=step0)).squeeze()
    ana0   = np.asarray(ana_spread.sel(step=step0)).squeeze()

    n_h = len(horizons_days)
    fig, axs = plt.subplots(1, n_h, figsize=(4 * n_h, 4), squeeze=False)

    for h_idx, h in enumerate(horizons_days):
        ax   = axs[0, h_idx]
        step = np.timedelta64(h, "D")
        x    = np.asarray(tigge_spread.sel(step=step)).squeeze() / tigge0   # TIGGE growth
        y    = np.asarray(ana_spread.sel(step=step)).squeeze()   / ana0      # ana growth
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            ax.set_title(f"h={h}d — no data")
            continue
        reg = linregress(x[valid], y[valid])
        ax.scatter(x[valid], y[valid], alpha=0.25, s=8, rasterized=True)
        xline = np.array([np.nanpercentile(x[valid], 1), np.nanpercentile(x[valid], 99)])
        ax.plot(xline, reg.slope * xline + reg.intercept, "r-", lw=1.5)
        # 1:1 reference
        lim = [min(xline[0], np.nanpercentile(y[valid], 1)),
               max(xline[1], np.nanpercentile(y[valid], 99))]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="1:1")
        ax.set_title(f"h={h}d   r={round_to_n(reg.rvalue, 2)}", fontsize=10)
        ax.set_xlabel("TIGGE spread growth")
        ax.set_ylabel("analogue spread growth")

    fig.suptitle("Cumulative spread growth: TIGGE vs analogues", size=13)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120)
    return fig


def plot_model_family_comparison(
    model_results: dict,          # ordered {label: fit_single_model_result}
    full_model_key: str | None = None,
    estimator_label: str = "",
    output: "str | Path | None" = None,
) -> plt.Figure:
    """R²(h) for all model families + coefficient panels for the richest model.

    Parameters
    ----------
    model_results:
        Ordered dict ``{model_name: result}`` where each result is the output
        of ``regression.fit_single_model``.
    full_model_key:
        Key in ``model_results`` whose coefficients are shown in the right
        panels.  Defaults to the last entry.
    estimator_label:
        Dimension estimator name (for the title).
    """

    if full_model_key is None:
        full_model_key = list(model_results)[-1]

    full_res  = model_results[full_model_key]
    pred_names = full_res["pred_names"]
    n_panels   = 1 + len(pred_names)   # R² + one per predictor

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_results)))
    h      = full_res["horizons"]

    fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

    # ── R² panel ─────────────────────────────────────────────────────────────
    ax_r2 = axs[0, 0]
    for ci, (name, res) in enumerate(model_results.items()):
        ls = "-" if name == full_model_key else "--"
        ax_r2.plot(h, res["r2"], "o" + ls, color=colors[ci],
                   lw=1.8, label=name)
    ax_r2.set_xlabel("lead time (days)")
    ax_r2.set_ylabel("R²")
    ax_r2.set_xticks(h)
    ax_r2.set_ylim(bottom=0)
    ax_r2.axhline(0, color="k", lw=0.5, ls="--")
    ax_r2.legend(fontsize=7)
    ax_r2.set_title("R² by model family", fontsize=9)

    # ── coefficient panels (full model) ──────────────────────────────────────
    for pi, pname in enumerate(pred_names):
        ax = axs[0, pi + 1]
        ax.bar(h, full_res["coefs"][pname], color="steelblue", alpha=0.8)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel(f"coef [{pname}]")
        ax.set_xticks(h)
        ax.set_title(f"{pname}\n({full_model_key})", fontsize=8)

    title = f"Model family comparison — {estimator_label}"
    fig.suptitle(title, size=11)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_multipredictor_regression(
    results: dict,
    r2_baselines: dict[str, np.ndarray] | None = None,
    title: str = "",
    output: "str | Path | None" = None,
) -> plt.Figure:
    """Coefficient and R² figure for a multi-predictor regression.

    One bar-per-horizon panel per predictor, plus a final R² comparison panel.

    Parameters
    ----------
    results:
        Output of ``regression.fit_multipredictor_regressions``.
    r2_baselines:
        Optional dict ``{label: r2_array}`` of reference R² curves to overlay
        in the R² panel (e.g. the 2-predictor model, density-only model).
    title:
        Figure title.
    """

    h          = results["horizons"]
    pred_names = results["pred_names"]
    n_preds    = len(pred_names)
    n_panels   = n_preds + 1          # one per predictor + R² panel

    fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

    # ── coefficient panels ───────────────────────────────────────────────────
    for pi, name in enumerate(pred_names):
        ax  = axs[0, pi]
        vals = results["full"]["coefs"][name]
        ax.bar(h, vals, color="steelblue", alpha=0.8)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("lead time (days)")
        ax.set_ylabel(f"coef [{name}]")
        ax.set_xticks(h)
        ax.set_title(name, fontsize=9)

    # ── R² panel ─────────────────────────────────────────────────────────────
    ax_r2 = axs[0, -1]
    ax_r2.plot(h, results["full"]["r2"], "o-", color="steelblue",
               lw=2, label="full (all predictors)")
    ax_r2.plot(h, results["reduced"]["r2"], "o--", color="steelblue",
               alpha=0.55, label=f"reduced (no {results.get('skip_key','dim')})")

    if r2_baselines:
        colors_base = ["tomato", "orange", "green", "purple"]
        for ci, (lbl, r2) in enumerate(r2_baselines.items()):
            ax_r2.plot(h, r2, "s:", color=colors_base[ci % len(colors_base)],
                       lw=1.5, label=lbl)

    ax_r2.set_ylim(bottom=0)
    ax_r2.set_xlabel("lead time (days)")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("R² comparison", fontsize=9)
    ax_r2.set_xticks(h)
    ax_r2.legend(fontsize=7)

    if title:
        fig.suptitle(title, size=11)
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_tigge_growth_vs_crps_skill(
    tigge_spread: xr.DataArray,
    crps_ana: np.ndarray,
    crps_clim: np.ndarray,
    horizons_days: list[int],
    output: "str | Path | None" = None,
) -> plt.Figure:
    """Scatter TIGGE spread growth vs CRPS skill (CRPS_clim - CRPS_ana).

    x-axis: TIGGE spread growth = spread(t+h) / spread(t, h=0)
    y-axis: CRPS_clim(t,h) - CRPS_ana(t,h)  (positive = analogues beat clim.)

    Parameters
    ----------
    tigge_spread:
        TIGGE spread DataArray with dims (time, step).
    crps_ana, crps_clim:
        Arrays of shape (n_times, len(horizons_days)).
    horizons_days:
        Lead times in days corresponding to the columns of crps_* arrays.
    """

    step0 = np.timedelta64(0, "D")
    spread0 = np.asarray(tigge_spread.sel(step=step0)).squeeze()

    n_h = len(horizons_days)
    fig, axs = plt.subplots(1, n_h, figsize=(4 * n_h, 4), squeeze=False)

    for h_idx, h in enumerate(horizons_days):
        ax = axs[0, h_idx]
        step = np.timedelta64(h, "D")
        x = np.asarray(tigge_spread.sel(step=step)).squeeze() / spread0  # spread growth
        y = crps_clim[:, h_idx] - crps_ana[:, h_idx]                     # CRPS skill
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            ax.set_title(f"h={h}d — no data")
            continue
        reg = linregress(x[valid], y[valid])
        ax.scatter(x[valid], y[valid], alpha=0.25, s=8, rasterized=True)
        xline = np.array([np.nanpercentile(x[valid], 1), np.nanpercentile(x[valid], 99)])
        ax.plot(xline, reg.slope * xline + reg.intercept, "r-", lw=1.5)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_title(f"h={h}d   r={round_to_n(reg.rvalue, 2)}", fontsize=10)
        ax.set_xlabel("TIGGE spread growth: spread(t+h)/spread(t)")
        ax.set_ylabel("CRPS_clim − CRPS_ana")

    fig.suptitle("TIGGE spread growth vs analogue CRPS skill", size=13)
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
