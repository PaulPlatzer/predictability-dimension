"""Recreate the TIGGE-spread versus dimension figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from predictability_dimension.config import DEFAULT_DIMENSION_OUTPUT, DEFAULT_TIGGE_SPREAD
from predictability_dimension.plotting import (
    align_spread_and_dimension,
    plot_dimension_vs_spread,
    plot_dimension_vs_spread_growth,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spread", type=Path, default=DEFAULT_TIGGE_SPREAD)
    parser.add_argument("--dimension", type=Path, default=DEFAULT_DIMENSION_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/generated"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spread = xr.open_dataarray(args.spread)
    dim = xr.open_dataset(args.dimension)
    spread, dim = align_spread_and_dimension(spread, dim)
    display_steps = spread.step.isel(step=range(2, 8))
    display_spread = spread.sel(step=display_steps)

    plot_dimension_vs_spread(display_spread, dim, args.output_dir / "spread-dim.png")
    plot_dimension_vs_spread_growth(spread, dim, "previous", display_steps, args.output_dir / "error_growth_dim1.png")
    plot_dimension_vs_spread_growth(spread, dim, "initial", display_steps, args.output_dir / "error_growth_dim2.png")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
