"""Local intrinsic-dimension estimators."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import xarray as xr
from tqdm.auto import tqdm


def _empty_result(n_samples: int, k_values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(k_values), n_samples)
    return np.full(shape, np.nan), np.full(shape, np.nan)


def estimate_local_dimensions(
    states: np.ndarray,
    k_values: Iterable[int],
    distances: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    estimators: Iterable[str] = ("mle", "pca", "ess"),
    smooth: bool = True,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Estimate local dimensions for several K values.

    ``distances`` and ``indices`` may come from one nearest-neighbour search at
    the largest K.  Reusing them keeps the notebook and script workflows short.
    """

    import skdim

    X = np.asarray(states, dtype=float)
    k_values = [int(k) for k in k_values]
    estimators = tuple(estimator.lower() for estimator in estimators)
    results: dict[str, np.ndarray] = {}

    if "mle" in estimators:
        dim, dim_smooth = _empty_result(len(X), k_values)
        for row, K in enumerate(tqdm(k_values, desc="dimension MLE")):
            knn = (distances[:, :K], indices[:, :K]) if distances is not None and indices is not None else None
            raw, smoothed = skdim.id.MLE().fit_transform_pw(
                X,
                precomputed_knn_arrays=knn,
                smooth=smooth,
                n_neighbors=K,
            )
            dim[row] = raw
            dim_smooth[row] = smoothed
        results["d_mle"] = dim
        results["d_mle_smooth"] = dim_smooth

    if "pca" in estimators:
        dim, dim_smooth = _empty_result(len(X), k_values)
        for row, K in enumerate(tqdm(k_values, desc="dimension lPCA")):
            knn = indices[:, :K] if indices is not None else None
            estimator = skdim.id.lPCA().fit_pw(
                X,
                precomputed_knn=knn,
                smooth=smooth,
                n_neighbors=K,
                n_jobs=n_jobs,
            )
            dim[row] = estimator.dimension_pw_
            dim_smooth[row] = estimator.dimension_pw_smooth_
        results["d_pca"] = dim
        results["d_pca_smooth"] = dim_smooth

    if "ess" in estimators:
        dim, dim_smooth = _empty_result(len(X), k_values)
        for row, K in enumerate(tqdm(k_values, desc="dimension ESS")):
            knn = (distances[:, :K], indices[:, :K]) if distances is not None and indices is not None else None
            raw, smoothed = skdim.id.ESS().fit_transform_pw(
                X,
                precomputed_knn_arrays=knn,
                smooth=smooth,
                n_neighbors=K,
                n_jobs=n_jobs,
            )
            dim[row] = raw
            dim_smooth[row] = smoothed
        results["d_ess"] = dim
        results["d_ess_smooth"] = dim_smooth

    return results


def dimensions_to_dataset(
    dimensions: dict[str, np.ndarray],
    k_values: Iterable[int],
    sample_coord: np.ndarray,
    sample_dim: str = "initial_time",
) -> xr.Dataset:
    """Pack dimension estimates into a self-describing xarray Dataset."""

    k_values = np.asarray(list(k_values), dtype=int)
    coords = {"K": k_values, sample_dim: sample_coord}
    data_vars = {
        name: xr.DataArray(values, dims=("K", sample_dim), coords=coords)
        for name, values in dimensions.items()
    }
    return xr.Dataset(data_vars=data_vars)


def comp_d_mle(states, Kvalues, dist=None, ind=None, smooth=True):
    """Compatibility wrapper for the original notebook function name."""

    result = estimate_local_dimensions(
        states,
        Kvalues,
        distances=dist,
        indices=ind,
        estimators=("mle",),
        smooth=smooth,
    )
    return result["d_mle"].T.squeeze(), result["d_mle_smooth"].T.squeeze()


def comp_d_pca(states, Kvalues, dist=None, ind=None, smooth=True, n_jobs: int = 1):
    """Compatibility wrapper for the original notebook function name."""

    result = estimate_local_dimensions(
        states,
        Kvalues,
        distances=dist,
        indices=ind,
        estimators=("pca",),
        smooth=smooth,
        n_jobs=n_jobs,
    )
    return result["d_pca"].T.squeeze(), result["d_pca_smooth"].T.squeeze()


def comp_d_ess(states, Kvalues, dist=None, ind=None, smooth=True, n_jobs: int = 1):
    """Compatibility wrapper for the original notebook function name."""

    result = estimate_local_dimensions(
        states,
        Kvalues,
        distances=dist,
        indices=ind,
        estimators=("ess",),
        smooth=smooth,
        n_jobs=n_jobs,
    )
    return result["d_ess"].T.squeeze(), result["d_ess_smooth"].T.squeeze()
