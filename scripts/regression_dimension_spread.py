"""Regression analysis: TIGGE spread growth ~ dimension + density proxy.

Two growth definitions are computed:

  cumulative:   y(t,h) = spread(t+h) / spread(t, h=0)
  incremental:  y(t,h) = spread(t+h) / spread(t+h-1)

For each, two nested models are fitted per lead time h and estimator:

  Full model    : y = a(h)*dim(t) + b(h)*density(t) + intercept
  Reduced model : y =               b(h)*density(t) + intercept

The comparison Δ R² = R²_full − R²_reduced quantifies the marginal
contribution of the dimension beyond what density alone explains.

Sensitivity to K (number of analogues for both dimension and density) is
shown within each figure.  Output: one PNG per (estimator, growth type).

Usage
-----
    python scripts/regression_dimension_spread.py
    python scripts/regression_dimension_spread.py --estimators d_mle d_pca
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.config import DATA_DIR, DEFAULT_DIMENSION_OUTPUT, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import plot_model_family_comparison, plot_regression_sensitivity
from predictability_dimension.regression import fit_horizon_regressions, fit_single_model

DEFAULT_ANA    = DATA_DIR / "analogue_spread_density.nc"
DEFAULT_OUTDIR = Path("outputs/figures")
HORIZONS_DAYS  = list(range(1, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim",        default=str(DEFAULT_DIMENSION_OUTPUT))
    p.add_argument("--spread",     default=str(DEFAULT_TIGGE_SPREAD))
    p.add_argument("--ana",        default=str(DEFAULT_ANA))
    p.add_argument("--estimators", nargs="+",
                   default=["d_mle", "d_mle_smooth", "d_pca", "d_pca_smooth", "d_ess", "d_ess_smooth"])
    p.add_argument("--k-density",  type=int, default=50,
                   help="K value for density proxy (must exist in analogue file)")
    p.add_argument("--outdir",     default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def _growth_targets(spread_aligned: xr.DataArray) -> dict[str, np.ndarray]:
    """Build cumulative and incremental spread-growth matrices (n, 7)."""
    step0    = np.timedelta64(0, "D")
    spread0  = spread_aligned.sel(step=step0).values.squeeze()

    cumulative = np.column_stack([
        spread_aligned.sel(step=np.timedelta64(h, "D")).values.squeeze() / spread0
        for h in HORIZONS_DAYS
    ])
    incremental = np.column_stack([
        spread_aligned.sel(step=np.timedelta64(h, "D")).values.squeeze()
        / spread_aligned.sel(step=np.timedelta64(h - 1, "D")).values.squeeze()
        for h in HORIZONS_DAYS
    ])
    return {"cumulative": cumulative, "incremental": incremental}


def main() -> None:
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print("Loading data …")
    dim   = xr.open_dataset(args.dim)
    tigge = xr.open_dataset(args.spread)
    ana   = xr.open_dataset(args.ana)

    spread_da = tigge["__xarray_dataarray_variable__"].sel(variable="gh")

    # Density: support both 1-D legacy (time) and 2-D (time, K_density) layouts
    dp = ana["density_proxy"]
    if "K_density" in dp.dims:
        density_da = dp.sel(K_density=args.k_density)
    else:
        density_da = dp

    # ── align ─────────────────────────────────────────────────────────────────
    common_times = np.intersect1d(
        np.intersect1d(spread_da.time.values, dim.initial_time.values),
        density_da.time.values,
    )
    if len(common_times) == 0:
        raise RuntimeError("No common dates across all three datasets.")
    print(f"Common dates: {len(common_times)}")

    spread_aligned = spread_da.sel(time=common_times)
    dim_aligned    = dim.sel(initial_time=common_times)
    x2             = density_da.sel(time=common_times).values  # (n,)

    # ── growth targets ────────────────────────────────────────────────────────
    targets    = _growth_targets(spread_aligned)
    k_dim_vals = [int(k) for k in dim.K.values]

    # ── regression + figures ──────────────────────────────────────────────────
    for growth_type, y in targets.items():
        print(f"\n=== {growth_type} ===")

        for estimator in args.estimators:
            if estimator not in dim.data_vars:
                print(f"  [skip] {estimator} not in dimension file")
                continue

            results_by_k: dict[int, dict] = {}
            x1_by_k:      dict[int, np.ndarray] = {}

            for K_dim in k_dim_vals:
                x1 = dim_aligned[estimator].sel(K=K_dim).values
                res = fit_horizon_regressions(x1, x2, y, horizons=HORIZONS_DAYS)
                results_by_k[K_dim] = res
                x1_by_k[K_dim]      = x1

                dr2 = res["delta_r2"]
                print(f"  {estimator} K={K_dim:>3d}  "
                      f"R²_full={res['full']['r2']}  ΔR²={dr2}")

            out_path = outdir / f"regression_{estimator}_{growth_type}_sensitivity.png"
            plot_regression_sensitivity(
                results_by_k  = results_by_k,
                growth_type   = growth_type,
                estimator     = estimator,
                horizons      = HORIZONS_DAYS,
                x1_by_k       = x1_by_k,
                x2            = x2,
                y             = y,
                h_3d          = 7,
                output        = out_path,
            )

    # ── Model family comparison (if analogue_spread available) ───────────────
    _run_model_families(ana, dim_aligned, spread_aligned, x2,
                        common_times, args.estimators, outdir,
                        results_by_k_last=results_by_k)

    print(f"\nAll figures saved in {outdir}/")


def _ana_spread_growth(
    ana: xr.Dataset, common_times: np.ndarray, horizons: list[int]
) -> np.ndarray | None:
    """
    Compute cumulative analogue spread growth: ana_spread(t,h) / ana_spread(t,0).
    Returns None if step=0 is not available.
    """
    step0 = np.timedelta64(0, "D")
    if step0 not in ana["analogue_spread"].step.values:
        print("  [model families] step=0 absent from analogue_spread.")
        print("  Re-run compute_analogue_spread.py to regenerate with h=0.")
        return None

    ana_sel  = ana["analogue_spread"].sel(time=common_times)
    spread0  = ana_sel.sel(step=step0).values                          # (n,)
    growth   = np.column_stack([
        ana_sel.sel(step=np.timedelta64(h, "D")).values / spread0
        for h in horizons
    ])                                                                  # (n, n_h)
    return growth


def _run_model_families(
    ana: xr.Dataset,
    dim_aligned: xr.Dataset,
    spread_aligned: xr.DataArray,
    x2: np.ndarray,
    common_times: np.ndarray,
    estimators: list[str],
    outdir: Path,
    results_by_k_last: dict | None = None,
) -> None:
    """Fit the 5 predictor families and produce comparison figures."""

    if "analogue_spread" not in ana.data_vars:
        print("\n[model families] Skipped — analogue_spread not in analogue file.")
        return

    print("\n=== Model family comparison (cumulative growth) ===")

    # ── TIGGE growth target ───────────────────────────────────────────────────
    step0   = np.timedelta64(0, "D")
    spread0 = spread_aligned.sel(step=step0).values.squeeze()
    y_cum   = np.column_stack([
        spread_aligned.sel(step=np.timedelta64(h, "D")).values.squeeze() / spread0
        for h in HORIZONS_DAYS
    ])                                                                  # (n, 7)

    # ── Analogue spread growth ────────────────────────────────────────────────
    x_ag = _ana_spread_growth(ana, common_times, HORIZONS_DAYS)
    if x_ag is None:
        return

    # ── Neighbourhood skill (optional) ───────────────────────────────────────
    has_neigh = "neighbourhood_skill" in ana.data_vars
    if has_neigh:
        step_crps = ana["neighbourhood_skill"].step_crps.values
        h_idx_map = {int(s / np.timedelta64(1, "D")): i for i, s in enumerate(step_crps)}
        x_ns = np.column_stack([
            ana["neighbourhood_skill"].sel(time=common_times).isel(
                step_crps=h_idx_map[h]).values
            for h in HORIZONS_DAYS
        ])                                                              # (n, 7)
    else:
        x_ns = None
        print("  neighbourhood_skill absent — running M1, M2, M4 only.")
        print("  Re-run compute_analogue_spread.py with --crps --crps-catalogue")

    K_REF = 50

    for estimator in estimators:
        if estimator not in dim_aligned.data_vars:
            continue

        x1 = dim_aligned[estimator].sel(K=K_REF).values               # (n,)

        # ── Define model families ─────────────────────────────────────────────
        families: dict[str, dict] = {}

        families["M1: ana_growth"] = fit_single_model(
            static_preds  = {},
            horizon_preds = {"ana_growth": x_ag},
            y=y_cum, horizons=HORIZONS_DAYS,
        )
        families["M2: + density"] = fit_single_model(
            static_preds  = {"density": x2},
            horizon_preds = {"ana_growth": x_ag},
            y=y_cum, horizons=HORIZONS_DAYS,
        )
        if has_neigh:
            families["M3: + neigh_skill"] = fit_single_model(
                static_preds  = {"density": x2},
                horizon_preds = {"ana_growth": x_ag, "neigh_skill": x_ns},
                y=y_cum, horizons=HORIZONS_DAYS,
            )
        families["M4: + dim"] = fit_single_model(
            static_preds  = {"density": x2, "dim": x1},
            horizon_preds = {"ana_growth": x_ag},
            y=y_cum, horizons=HORIZONS_DAYS,
        )
        if has_neigh:
            families["M5: full"] = fit_single_model(
                static_preds  = {"density": x2, "dim": x1},
                horizon_preds = {"ana_growth": x_ag, "neigh_skill": x_ns},
                y=y_cum, horizons=HORIZONS_DAYS,
            )

        full_key = list(families)[-1]   # most complete model

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"\n  {estimator} K={K_REF}")
        print(f"  {'Model':<25}  R²(h=1..7)")
        for name, res in families.items():
            print(f"  {name:<25}  {res['r2'].round(3)}")

        # ── Figure ───────────────────────────────────────────────────────────
        out_path = outdir / f"model_families_{estimator}_K{K_REF}.png"
        plot_model_family_comparison(
            model_results  = families,
            full_model_key = full_key,
            estimator_label= f"{estimator} K={K_REF}",
            output         = out_path,
        )
        print(f"  → {out_path.name}")


if __name__ == "__main__":
    main()
