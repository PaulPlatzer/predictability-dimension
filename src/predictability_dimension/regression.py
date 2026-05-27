"""Linear regression utilities for predictability analysis.

Public API
----------
fit_horizon_regressions(x1, x2, y, horizons)
    Two-predictor (dim, density) model — full vs reduced (no dim).

fit_multipredictor_regressions(static_preds, horizon_preds, y, horizons)
    N-predictor model supporting both static and horizon-varying predictors.
    Compares the full model against a reduced model that drops one predictor.

Two nested models are fitted per lead time h:

  Full model    : y(t,h) = a(h)*x1(t) + b(h)*x2(t) + intercept(h)
  Reduced model : y(t,h) =              b(h)*x2(t) + intercept(h)

where y is a spread-growth target, x1 is a local dimension estimate, and
x2 is a local density proxy.  Comparing both models reveals the marginal
contribution of dimension (Δ R²).

Predictors are standardised with a single StandardScaler fitted once on the
joint data, so coefficients a and b are directly comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def fit_horizon_regressions(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    horizons: list[int],
    standardize: bool = True,
) -> dict:
    """Fit full and reduced regression models for each forecast horizon.

    Parameters
    ----------
    x1, x2:
        Predictor arrays, shape (n,).  x1 is the dimension estimate, x2 is
        the density proxy.
    y:
        Target array, shape (n, len(horizons)).
    horizons:
        Horizon values in days (used for labelling).
    standardize:
        Scale predictors to zero mean / unit variance before fitting.

    Returns
    -------
    dict with keys:
        'horizons'  : (n_h,) int array
        'full'      : dict — coef_a, coef_b, intercept, r2
        'reduced'   : dict — coef_b, intercept, r2   (x1 excluded)
        'delta_r2'  : (n_h,) improvement in R² from adding dimension
        'scaler'    : fitted StandardScaler (or None if standardize=False)
    """

    X_full = np.column_stack([x1, x2])

    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_full)
    else:
        scaler = None
        X_fit = X_full

    X_red_fit = X_fit[:, [1]]   # density column only, already on the same scale

    n_h = len(horizons)
    full    = {k: np.full(n_h, np.nan) for k in ("coef_a", "coef_b", "intercept", "r2")}
    reduced = {k: np.full(n_h, np.nan) for k in ("coef_b", "intercept", "r2")}

    for h_idx in range(n_h):
        y_h   = y[:, h_idx]
        valid = np.isfinite(y_h) & np.isfinite(X_fit).all(axis=1)
        if valid.sum() < 10:
            continue

        # Full model: dim + density
        reg = LinearRegression().fit(X_fit[valid], y_h[valid])
        full["coef_a"][h_idx]    = reg.coef_[0]
        full["coef_b"][h_idx]    = reg.coef_[1]
        full["intercept"][h_idx] = reg.intercept_
        full["r2"][h_idx]        = reg.score(X_fit[valid], y_h[valid])

        # Reduced model: density only
        reg_red = LinearRegression().fit(X_red_fit[valid], y_h[valid])
        reduced["coef_b"][h_idx]    = reg_red.coef_[0]
        reduced["intercept"][h_idx] = reg_red.intercept_
        reduced["r2"][h_idx]        = reg_red.score(X_red_fit[valid], y_h[valid])

    return {
        "horizons":  np.array(horizons),
        "full":      full,
        "reduced":   reduced,
        "delta_r2":  full["r2"] - reduced["r2"],
        "scaler":    scaler,
    }


def fit_single_model(
    static_preds: dict[str, np.ndarray],
    horizon_preds: dict[str, np.ndarray],
    y: np.ndarray,
    horizons: list[int],
    standardize: bool = True,
) -> dict:
    """Fit one independent linear regression per horizon.

    Parameters
    ----------
    static_preds:
        ``{name: array(n,)}`` — predictors constant across horizons.
    horizon_preds:
        ``{name: array(n, n_h)}`` — predictors that vary with horizon.
    y:
        Target array, shape ``(n, n_h)``.
    horizons:
        Horizon values in days (for labelling).
    standardize:
        Standardise all predictors to zero mean / unit variance using a
        single scaler fitted on data pooled across all horizons.

    Returns
    -------
    dict with keys ``'horizons'``, ``'pred_names'``, ``'coefs'`` (dict of arrays),
    ``'intercept'`` (array), ``'r2'`` (array).
    """

    all_names = list(static_preds) + list(horizon_preds)
    n_h = len(horizons)

    def _build_X(h_idx: int) -> np.ndarray:
        cols = [v for v in static_preds.values()]
        cols += [v[:, h_idx] for v in horizon_preds.values()]
        return np.column_stack(cols)

    if standardize and all_names:
        X_pooled = np.concatenate([_build_X(h) for h in range(n_h)], axis=0)
        scaler   = StandardScaler().fit(X_pooled)
        scale    = lambda X: scaler.transform(X)  # noqa: E731
    else:
        scaler = None
        scale  = lambda X: X  # noqa: E731

    coefs     = {n: np.full(n_h, np.nan) for n in all_names}
    intercept = np.full(n_h, np.nan)
    r2        = np.full(n_h, np.nan)

    for h_idx in range(n_h):
        y_h   = y[:, h_idx]
        X_h   = scale(_build_X(h_idx))
        valid = np.isfinite(y_h) & np.isfinite(X_h).all(axis=1)
        if valid.sum() < 10:
            continue
        reg = LinearRegression().fit(X_h[valid], y_h[valid])
        for i, name in enumerate(all_names):
            coefs[name][h_idx] = reg.coef_[i]
        intercept[h_idx] = reg.intercept_
        r2[h_idx]        = reg.score(X_h[valid], y_h[valid])

    return {
        "horizons":   np.array(horizons),
        "pred_names": all_names,
        "coefs":      coefs,
        "intercept":  intercept,
        "r2":         r2,
    }


def fit_multipredictor_regressions(
    static_preds: dict[str, np.ndarray],
    horizon_preds: dict[str, np.ndarray],
    y: np.ndarray,
    horizons: list[int],
    standardize: bool = True,
    skip_key: str = "dim",
) -> dict:
    """Fit full and reduced models with any number of predictors.

    Predictors are split into two groups:

    * ``static_preds``  — shape ``(n,)``, the same for every horizon
      (typically: dimension, density proxy).
    * ``horizon_preds`` — shape ``(n, n_h)``, one column per horizon
      (typically: analogue_spread, neighbourhood_skill).

    The *full* model uses all predictors.  The *reduced* model drops the
    predictor identified by ``skip_key`` (default: ``"dim"``).

    A single ``StandardScaler`` is fitted on the predictor matrix averaged
    across horizons, so that coefficients are comparable across h and
    across predictors.

    Parameters
    ----------
    static_preds:
        ``{name: array(n,)}``
    horizon_preds:
        ``{name: array(n, n_h)}``
    y:
        Target array, shape ``(n, n_h)``.
    horizons:
        Horizon values in days (for labelling).
    standardize:
        Standardise predictors before fitting.
    skip_key:
        Predictor name to drop in the reduced model.

    Returns
    -------
    dict with keys:

    * ``'horizons'``  : ``(n_h,)`` int array
    * ``'full'``      : ``{'coefs': {name: array(n_h)}, 'r2': array(n_h)}``
    * ``'reduced'``   : same structure, without ``skip_key``
    * ``'delta_r2'``  : ``(n_h,)`` — ``R²_full − R²_reduced``
    * ``'pred_names'``: list of predictor names (full model order)
    * ``'scaler'``    : fitted ``StandardScaler`` or ``None``
    """

    all_names    = list(static_preds) + list(horizon_preds)
    n_h          = len(horizons)
    n_preds      = len(all_names)

    # ── build scaler fitted on the average predictor matrix across horizons ──
    def _build_X(h_idx: int) -> np.ndarray:
        cols = [v for v in static_preds.values()]
        cols += [v[:, h_idx] for v in horizon_preds.values()]
        return np.column_stack(cols)

    if standardize:
        X_stack = np.concatenate([_build_X(h) for h in range(n_h)], axis=0)
        scaler  = StandardScaler().fit(X_stack)
    else:
        scaler = None

    def _scale(X: np.ndarray) -> np.ndarray:
        return scaler.transform(X) if scaler is not None else X

    # ── indices for the reduced model (drop skip_key) ──────────────────────
    skip_idx    = all_names.index(skip_key) if skip_key in all_names else None
    red_names   = [n for n in all_names if n != skip_key]
    red_cols    = [i for i, n in enumerate(all_names) if n != skip_key]

    # ── storage ──────────────────────────────────────────────────────────────
    full_coefs  = {n: np.full(n_h, np.nan) for n in all_names}
    full_r2     = np.full(n_h, np.nan)
    red_coefs   = {n: np.full(n_h, np.nan) for n in red_names}
    red_r2      = np.full(n_h, np.nan)

    for h_idx in range(n_h):
        X_h   = _scale(_build_X(h_idx))
        y_h   = y[:, h_idx]
        valid = np.isfinite(y_h) & np.isfinite(X_h).all(axis=1)
        if valid.sum() < 10:
            continue

        # Full
        reg = LinearRegression().fit(X_h[valid], y_h[valid])
        for i, name in enumerate(all_names):
            full_coefs[name][h_idx] = reg.coef_[i]
        full_r2[h_idx] = reg.score(X_h[valid], y_h[valid])

        # Reduced
        X_red = X_h[:, red_cols]
        reg_r = LinearRegression().fit(X_red[valid], y_h[valid])
        for i, name in enumerate(red_names):
            red_coefs[name][h_idx] = reg_r.coef_[i]
        red_r2[h_idx] = reg_r.score(X_red[valid], y_h[valid])

    return {
        "horizons":   np.array(horizons),
        "full":       {"coefs": full_coefs, "r2": full_r2},
        "reduced":    {"coefs": red_coefs,  "r2": red_r2},
        "delta_r2":   full_r2 - red_r2,
        "pred_names": all_names,
        "scaler":     scaler,
    }
