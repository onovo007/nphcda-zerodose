"""
Lightweight inferential statistics (numpy + scipy only, dependency-safe):
- OLS with HC3 heteroskedasticity-robust standard errors and p-values.
- Benjamini-Hochberg false-discovery-rate adjustment.
- Beta regression for a bounded (0,1) outcome: true Beta MLE via statsmodels BetaModel when it
  imports cleanly, otherwise a logit-link linear model with HC3 robust SEs (the Beta-regression
  mean model). Either way returns coefficients, standard errors and p-values.

These power the inferential follow-up to LASSO driver selection (robust SEs / p-values) without
depending on statsmodels.api (which is broken under some scipy versions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _ss


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def add_ci(df: pd.DataFrame, coef_col: str, se_col: str, z: float = 1.96) -> pd.DataFrame:
    """Add 95% confidence-interval bounds (coef +/- z*SE) to an inference table."""
    df = df.copy()
    df["CI_low"] = df[coef_col] - z * df[se_col]
    df["CI_high"] = df[coef_col] + z * df[se_col]
    return df


def ols_robust(y, X_df: pd.DataFrame) -> pd.DataFrame:
    """OLS with HC3 robust SEs. Returns term, coef, robust_SE, t, p_value (intercept first)."""
    y = np.asarray(y, dtype=float)
    Xd = X_df.values.astype(float)
    X = np.column_stack([np.ones(len(Xd)), Xd])
    terms = ["(intercept)"] + list(X_df.columns)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.clip(np.einsum("ij,jk,ik->i", X, XtX_inv, X), 0, 0.9999)  # leverage
    omega = (resid ** 2) / (1.0 - h) ** 2  # HC3
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    df = max(n - k, 1)
    p = 2 * _ss.t.sf(np.abs(t), df)
    return pd.DataFrame({"term": terms, "coef": beta, "robust_SE": se, "t": t, "p_value": p})


def beta_regression(p_outcome, X_df: pd.DataFrame):
    """Beta regression of a (0,1) outcome on X. Returns (table, method_label).

    Tries true Beta MLE (statsmodels BetaModel, logit link); falls back to a logit-link linear
    model with HC3 robust SEs (the Beta mean model) if BetaModel is unavailable.
    """
    eps = 1e-3
    y = np.clip(np.asarray(p_outcome, dtype=float), eps, 1 - eps)
    terms = ["(intercept)"] + list(X_df.columns)
    try:
        from statsmodels.othermod.betareg import BetaModel  # does not import statsmodels.api
        Xc = np.column_stack([np.ones(len(X_df)), X_df.values.astype(float)])
        m = BetaModel(y, Xc).fit(disp=0)
        k = len(terms)
        params = np.asarray(m.params)[:k]
        se = np.asarray(m.bse)[:k]
        pv = np.asarray(m.pvalues)[:k]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, params / se, 0.0)
        tbl = pd.DataFrame({"term": terms, "coef": params, "SE": se, "z": z, "p_value": pv})
        return tbl, "Beta regression (logit link, maximum likelihood)"
    except Exception:
        logit = np.log(y / (1 - y))
        r = ols_robust(logit, X_df).rename(columns={"robust_SE": "SE", "t": "z"})
        return r, "Logit-linear regression with HC3 robust SE (Beta mean model)"
