"""
Domain 2 - Dropout and completion dynamics.

Live Prophet forecasts of the three national dropout series (Penta1 to Penta3, Penta1 to
Measles1, Measles1 to Measles2), LASSO-selected drivers per dropout pair, and state-by-year
dropout matrices. Logic lifted from D1_D2 cells 27, 29 and 32.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as C
from models.d1_forecast import run_prophet, _horizon_to


@st.cache_data(show_spinner="Fitting dropout forecasts (Prophet)...")
def dropout_forecasts(_nat, key: str) -> dict:
    nat = _nat
    series = {}
    for col, label in C.DROPOUT_TARGETS.items():
        if col not in nat.columns:
            continue
        ts = nat[["ds", col]].rename(columns={col: "y"}).dropna()
        fc = run_prophet(ts)
        cutoff = ts["ds"].max()
        hist = fc[fc["ds"] <= cutoff]
        fore = fc[fc["ds"] > cutoff].reset_index(drop=True)
        lo80 = fore["yhat"] - (fore["yhat"] - fore["yhat_lower"]) * C.PI_80_FACTOR
        hi80 = fore["yhat"] + (fore["yhat_upper"] - fore["yhat"]) * C.PI_80_FACTOR
        series[col] = dict(
            label=label,
            obs_x=ts["ds"].dt.strftime("%Y-%m-%d").tolist(), obs_y=ts["y"].tolist(),
            hist_x=hist["ds"].dt.strftime("%Y-%m-%d").tolist(), hist_y=hist["yhat"].tolist(),
            fore_x=fore["ds"].dt.strftime("%Y-%m-%d").tolist(), fore_y=fore["yhat"].tolist(),
            lo95=fore["yhat_lower"].tolist(), hi95=fore["yhat_upper"].tolist(),
            lo80=lo80.tolist(), hi80=hi80.tolist(), cutoff=cutoff.strftime("%Y-%m-%d"),
        )
    return series


def lasso_drivers(agg: pd.DataFrame, model_dataset: pd.DataFrame) -> dict:
    """LassoCV coefficients per dropout target (D1_D2 cell 29)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV

    feats = [f for f in C.LASSO_FEATURES if f in model_dataset.columns]
    targets = [t for t in C.DROPOUT_TARGETS if t in agg.columns]

    sd = (agg[agg["year"] == 2024].groupby("state")[targets].mean().reset_index())
    sd["state"] = sd["state"].astype(str).str.strip().str.title()
    mod = model_dataset.copy()
    mod["state_key"] = mod["state_name"].astype(str).str.strip().str.title()
    merged = sd.merge(mod[["state_key"] + feats], left_on="state", right_on="state_key", how="inner")

    X = merged[feats].fillna(merged[feats].median())
    X_sc = StandardScaler().fit_transform(X)
    results = {}
    for target in targets:
        y = merged[target].fillna(merged[target].median()).values
        lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
        lasso.fit(X_sc, y)
        coefs = pd.Series(np.abs(lasso.coef_), index=feats).sort_values(ascending=False)
        results[target] = coefs[coefs > 0]
    return results


def _driver_design(agg: pd.DataFrame, model_dataset: pd.DataFrame):
    """Merge state 2024 dropout means with standardized equity covariates. Returns (merged, feats)."""
    feats = [f for f in C.LASSO_FEATURES if f in model_dataset.columns]
    targets = [t for t in C.DROPOUT_TARGETS if t in agg.columns]
    sd = agg[agg["year"] == 2024].groupby("state")[targets].mean().reset_index()
    sd["state"] = sd["state"].astype(str).str.strip().str.title()
    mod = model_dataset.copy()
    mod["state_key"] = mod["state_name"].astype(str).str.strip().str.title()
    merged = sd.merge(mod[["state_key"] + feats], left_on="state", right_on="state_key", how="inner")
    return merged, feats, targets


@st.cache_data(show_spinner="Bootstrapping LASSO drivers and fitting robust regressions...")
def lasso_drivers_inference(_agg, _model_dataset, key: str, n_boot: int = 200) -> dict:
    """For each dropout pair: LASSO selection-stability (bootstrap selection frequency) plus an
    inferential regression on the LASSO-selected drivers with HC3 robust SEs, p-values and a
    Benjamini-Hochberg FDR adjustment. Dropout can be negative, so the inferential model is OLS
    with robust SEs (the statistically correct family for an unbounded rate)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    import stats_infer as si

    merged, feats, targets = _driver_design(_agg, _model_dataset)
    if merged.empty or not feats:
        return {}
    Xraw = merged[feats].fillna(merged[feats].median())
    scaler = StandardScaler().fit(Xraw)
    X_sc = scaler.transform(Xraw)
    Xz = pd.DataFrame(X_sc, columns=feats)
    rng = np.random.default_rng(42)
    n = len(merged)
    out = {}
    for target in targets:
        y = merged[target].fillna(merged[target].median()).values.astype(float)
        base = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_sc, y)
        selected = [f for f, c in zip(feats, base.coef_) if abs(c) > 0]
        # Bootstrap selection frequency.
        freq = {f: 0 for f in feats}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                lb = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_sc[idx], y[idx])
            except Exception:
                continue
            for f, c in zip(feats, lb.coef_):
                if abs(c) > 0:
                    freq[f] += 1
        sel = selected or sorted(feats, key=lambda f: freq[f], reverse=True)[:5]
        reg = si.ols_robust(y, Xz[sel])
        reg = reg[reg["term"] != "(intercept)"].copy()
        reg["p_FDR"] = si.bh_fdr(reg["p_value"].values)
        lasso_abs = {f: abs(c) for f, c in zip(feats, base.coef_)}
        reg.insert(1, "selection_freq_pct", reg["term"].map(lambda f: round(100 * freq[f] / max(n_boot, 1), 0)))
        reg.insert(1, "lasso_abs_coef", reg["term"].map(lambda f: round(lasso_abs.get(f, 0), 3)))
        reg = reg.rename(columns={"term": "Driver", "lasso_abs_coef": "|LASSO coef|",
                                  "selection_freq_pct": "Selection freq (%)", "coef": "Std beta",
                                  "robust_SE": "Robust SE", "p_value": "p-value", "p_FDR": "p (FDR)"})
        for c in ["Std beta", "Robust SE", "t", "p-value", "p (FDR)"]:
            reg[c] = reg[c].round(3)
        reg["Driver"] = reg["Driver"].map(lambda s: s.replace("pct_", "").replace("_", " ").title())
        out[C.DROPOUT_TARGETS[target]] = reg.sort_values("p (FDR)").reset_index(drop=True)
    return out


@st.cache_data(show_spinner="Forecasting state dropout (Prophet, 2026-2027)...")
def state_dropout_forecasts(_dhis2, key: str) -> pd.DataFrame:
    """Per-state Prophet forecast of each dropout pair; monthly 2026-2027 (long) for microplanning."""
    from data_io import prep_dhis2, state_monthly
    agg = state_monthly(prep_dhis2(_dhis2))
    zmap = (agg.drop_duplicates("state").set_index("state")["zone"].to_dict()
            if "zone" in agg.columns else {})
    out = []
    for state, g in agg.groupby("state"):
        g = g.sort_values("ds")
        last = g["ds"].max()
        per = _horizon_to(last)
        for col, label in C.DROPOUT_TARGETS.items():
            if col not in g.columns:
                continue
            ts = g[["ds", col]].rename(columns={col: "y"}).dropna()
            if len(ts) < 18:
                continue
            try:
                fc = run_prophet(ts, periods=per)
            except Exception:
                continue
            fr = fc[(fc["ds"] > last) & (fc["ds"].dt.year.isin([2026, 2027]))]
            for _, r in fr.iterrows():
                out.append({
                    "zone": zmap.get(state, ""), "state": state, "dropout_pair": label,
                    "month": r["ds"].strftime("%Y-%m"), "year": int(r["ds"].year),
                    "forecast_dropout_pct": round(float(r["yhat"]), 1),
                    "lower95_pct": round(float(r["yhat_lower"]), 1),
                    "upper95_pct": round(float(r["yhat_upper"]), 1)})
    return pd.DataFrame(out)


def state_year_observed(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Observed state-by-year mean dropout matrix (fast path, D1_D2 cell 32 historical block)."""
    g = agg.groupby(["state", "year"])[metric].mean().reset_index()
    piv = g.pivot(index="state", columns="year", values=metric).round(1)
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv


@st.cache_data(show_spinner="Forecasting state dropout (Prophet, all states)...")
def state_year_with_forecast(_agg, metric: str, key: str, to_year: int = 2027) -> pd.DataFrame:
    """Observed + per-state Prophet calendar-year forecast (on-demand heavy path, cell 32)."""
    agg = _agg
    hist = state_year_observed(agg, metric)
    last_year = int(max(hist.columns))
    fc_rows = []
    for state in sorted(agg["state"].unique()):
        ts = agg[agg["state"] == state][["ds", metric]].rename(columns={metric: "y"}).dropna()
        if len(ts) < 18:
            continue
        months = (to_year - ts["ds"].max().year) * 12 + (12 - ts["ds"].max().month) + 1
        try:
            fc = run_prophet(ts, periods=max(months, C.FORECAST_MONTHS))
        except Exception:
            continue
        fut = fc[fc["ds"] > ts["ds"].max()].copy()
        fut["year"] = fut["ds"].dt.year
        yr = fut.groupby("year")["yhat"].mean()
        for y, v in yr.items():
            if y > last_year and y <= to_year:
                fc_rows.append({"state": state, "year": int(y), "value": round(v, 1)})
    fc_df = pd.DataFrame(fc_rows)
    out = hist.copy()
    if not fc_df.empty:
        fp = fc_df.pivot(index="state", columns="year", values="value")
        out = out.join(fp, how="left")
    return out.reindex(sorted(out.columns), axis=1)
