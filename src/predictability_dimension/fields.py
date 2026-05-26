"""Read gridded geopotential fields and convert them to state vectors.

The scientific choice here is explicit: dimensions are now estimated from
the full gridded z500 field, not from a prior PCA representation.  We still
apply lightweight preprocessing so Euclidean distances are not dominated by
the spatial mean state or by grid points with larger raw variance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


GEOPOTENTIAL_NAMES = ("z", "geopotential")


def infer_field_variable(ds: xr.Dataset, variable: str | None = None) -> str:
    """Return the data variable that contains the geopotential field."""

    if variable is not None:
        if variable not in ds.data_vars:
            available = ", ".join(ds.data_vars)
            raise KeyError(f"Variable {variable!r} not found. Available variables: {available}")
        return variable

    for name in GEOPOTENTIAL_NAMES:
        if name in ds.data_vars:
            return name

    for name, da in ds.data_vars.items():
        attrs = {key.lower(): str(value).lower() for key, value in da.attrs.items()}
        if "geopotential" in attrs.get("standard_name", ""):
            return name
        if "geopotential" in attrs.get("long_name", ""):
            return name
        if attrs.get("grib_shortname") == "z":
            return name

    available = ", ".join(ds.data_vars)
    raise ValueError(f"Could not infer the geopotential variable. Available variables: {available}")


def _select_or_squeeze(da: xr.DataArray, coord: str, value) -> xr.DataArray:
    """Select a coordinate value when requested, otherwise drop singleton axes."""

    if coord not in da.dims:
        return da

    if value is None:
        if da.sizes[coord] == 1:
            return da.isel({coord: 0}, drop=True)
        return da

    try:
        return da.sel({coord: value}, method="nearest")
    except (KeyError, TypeError, ValueError):
        return da.sel({coord: value})


def open_geopotential_field(
    path: str | Path,
    variable: str | None = None,
    level_hpa: float | None = 500,
    step: np.timedelta64 | None = None,
    member: int | None = None,
    chunks: dict | None = None,
) -> xr.DataArray:
    """Open a z500 field and select the physical axes used for the analysis."""

    ds = xr.open_dataset(path, chunks=chunks)
    name = infer_field_variable(ds, variable)
    da = ds[name]

    da = _select_or_squeeze(da, "isobaricInhPa", level_hpa)
    da = _select_or_squeeze(da, "step", step)
    da = _select_or_squeeze(da, "number", member)

    unresolved = [dim for dim in ("isobaricInhPa", "step", "number") if dim in da.dims]
    if unresolved:
        details = ", ".join(f"{dim}={da.sizes[dim]}" for dim in unresolved)
        raise ValueError(
            "The field still has non-time physical axes after selection "
            f"({details}). Pass --level-hpa, --step-days, or --member explicitly."
        )

    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension, got dimensions {da.dims!r}.")

    da.name = name
    return da


def field_to_state_matrix(
    field: xr.DataArray,
    sample_dim: str = "time",
    feature_dims: Iterable[str] | None = None,
    anomaly: bool = True,
    standardize: bool = True,
    area_weight: bool = True,
) -> xr.DataArray:
    """Stack a gridded field into a 2D ``sample x feature`` matrix.

    Parameters are deliberately plain because this is research code: notebooks
    should make each preprocessing choice visible and easy to change.
    """

    if sample_dim not in field.dims:
        raise ValueError(f"Sample dimension {sample_dim!r} is absent from {field.dims!r}.")

    da = field.transpose(sample_dim, ...)

    if anomaly:
        da = da - da.mean(sample_dim)

    if standardize:
        scale = da.std(sample_dim)
        da = da / scale.where(scale > 0)
        da = da.fillna(0.0)

    if area_weight and "latitude" in da.dims:
        weights = np.sqrt(np.cos(np.deg2rad(da["latitude"].clip(-90.0, 90.0))))
        da = da * weights

    if feature_dims is None:
        feature_dims = [dim for dim in da.dims if dim != sample_dim]
    else:
        feature_dims = list(feature_dims)

    states = da.stack(feature=feature_dims).transpose(sample_dim, "feature")
    states.name = "state_vector"
    states.attrs.update(
        {
            "anomaly": anomaly,
            "standardize": standardize,
            "area_weight": area_weight,
            "feature_dims": ",".join(feature_dims),
        }
    )
    return states


def as_numpy_matrix(states: xr.DataArray) -> np.ndarray:
    """Return a finite 2D NumPy matrix suitable for scikit-learn/skdim."""

    matrix = np.asarray(states.to_numpy(), dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("The state matrix contains NaN or infinite values.")
    return matrix

