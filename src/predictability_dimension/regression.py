"""Linear regression utilities for predictability analysis.

The model is:
    y(t, h) = a(h) * x1(t) + b(h) * x2(t) + intercept(h)

where y is a predictability target (e.g. TIGGE spread growth), x1 is a
local dimension estimate, and x2 is a local density proxy.  A separate
regression is fit for each lead time h.
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
) -> dict[str, np.ndarray]:
    """Fit y[:, h] = a(h)*x1 + b(h)*x2 independently for each horizon.

    Parameters
    ----------
    x1, x2:
        Predictor arrays, shape (n,).  x1 is the dimension estimate, x2 is
        the density proxy.
    y:
        Target array, shape (n, len(horizons)).  Typically the TIGGE spread
        growth ratio spread(t+h) / spread(t, h=0).
    horizons:
        Horizon values in days, used only for labelling the output.
    standardize:
        Scale predictors to zero mean / unit variance before fitting so that
        coefficients a and b are directly comparable in magnitude.

    Returns
    -------
    dict with keys:
        'horizons'  : (n_h,) int array
        'coef_a'    : (n_h,) coefficient of x1 (dimension)
        'coef_b'    : (n_h,) coefficient of x2 (density proxy)
        'intercept' : (n_h,) regression intercept
        'r2'        : (n_h,) coefficient of determination
    """

    X = np.column_stack([x1, x2])
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    n_h = len(horizons)
    coef_a = np.full(n_h, np.nan)
    coef_b = np.full(n_h, np.nan)
    intercept = np.full(n_h, np.nan)
    r2 = np.full(n_h, np.nan)

    for h_idx in range(n_h):
        y_h = y[:, h_idx]
        valid = np.isfinite(y_h) & np.isfinite(X_fit).all(axis=1)
        if valid.sum() < 10:
            continue
        reg = LinearRegression().fit(X_fit[valid], y_h[valid])
        coef_a[h_idx] = reg.coef_[0]
        coef_b[h_idx] = reg.coef_[1]
        intercept[h_idx] = reg.intercept_
        r2[h_idx] = reg.score(X_fit[valid], y_h[valid])

    return {
        "horizons": np.array(horizons),
        "coef_a": coef_a,
        "coef_b": coef_b,
        "intercept": intercept,
        "r2": r2,
    }
