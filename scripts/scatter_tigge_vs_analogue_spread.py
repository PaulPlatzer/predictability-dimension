"""Scatter TIGGE spread and growth against analogue spread and growth (h=1…7 days).

Produces two figures:
  1. TIGGE spread (level) vs analogue spread (level)
  2. TIGGE spread growth (cumulative) vs analogue spread growth (cumulative)

Run compute_analogue_spread.py first (with step=0 available in analogue_spread).

Usage
-----
    python scripts/scatter_tigge_vs_analogue_spread.py
    python scripts/scatter_tigge_vs_analogue_spread.py \\
        --spread data/TIGGEspread.nc \\
        --ana    data/analogue_spread_density.nc \\
        --outdir outputs/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from predictability_dimension.config import DATA_DIR, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import (
    plot_ana_growth_vs_tigge_growth,
    plot_tigge_vs_analogue_spread,
)

DEFAULT_ANA   = DATA_DIR / "analogue_spread_density.nc"
DEFAULT_OUTDIR = Path("outputs/figures")
HORIZONS_DAYS  = list(range(1, 8))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spread", default=str(DEFAULT_TIGGE_SPREAD))
    p.add_argument("--ana",    default=str(DEFAULT_ANA))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    tigge     = xr.open_dataset(args.spread)
    spread_da = tigge["__xarray_dataarray_variable__"].sel(variable="gh")
    ana       = xr.open_dataset(args.ana)
    ana_spread = ana["analogue_spread"]

    # ── Common times ──────────────────────────────────────────────────────────
    common_times = np.intersect1d(spread_da.time.values, ana_spread.time.values)
    if len(common_times) == 0:
        raise RuntimeError("No common dates between TIGGE and analogue spread files.")

    tigge_aligned = spread_da.sel(time=common_times)
    ana_aligned   = ana_spread.sel(time=common_times)

    # ── Figure 1: spread levels ───────────────────────────────────────────────
    ana_steps_days = [int(h / np.timedelta64(1, "D")) for h in ana_aligned.step.values]
    h_lev   = [h for h in HORIZONS_DAYS if h in ana_steps_days]
    h_idx_l = [ana_steps_days.index(h) for h in h_lev]
    ana_np  = ana_aligned.values[:, h_idx_l]

    out_lev = outdir / "tigge_vs_ana_spread_level.png"
    plot_tigge_vs_analogue_spread(
        tigge_aligned, ana_np, horizons_days=h_lev, output=out_lev
    )
    print(f"Saved → {out_lev}")

    # ── Figure 2: cumulative growth ───────────────────────────────────────────
    step0 = np.timedelta64(0, "D")
    if step0 not in ana_aligned.step.values:
        print("step=0 not in analogue_spread — skipping growth comparison.")
        print("Re-run compute_analogue_spread.py to regenerate with h=0.")
        return

    h_grow = [h for h in HORIZONS_DAYS
              if np.timedelta64(h, "D") in tigge_aligned.step.values
              and h in ana_steps_days]

    out_grow = outdir / "tigge_vs_ana_spread_growth.png"
    plot_ana_growth_vs_tigge_growth(
        tigge_aligned, ana_aligned, horizons_days=h_grow, output=out_grow
    )
    print(f"Saved → {out_grow}")


if __name__ == "__main__":
    main()
