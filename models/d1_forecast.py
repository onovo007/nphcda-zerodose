"""
Domain 1 - Antigen coverage forecasting.

Live national Prophet forecasts for BCG, Penta1, Penta3 and Measles1, re-expressed as a
percent of the 2024 baseline so the 80 percent target is a meaningful line (D1_D2 cells
19-20). Plus a fast linear-trend LGA at-risk screen (the full per-LGA Prophet is the
on-demand heavy path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import config as C
from data_io import prep_dhis2


def run_prophet(ts_df: pd.DataFrame, periods: int = C.FORECAST_MONTHS):
    from prophet import Prophet
    df = ts_df[["ds", "y"]].dropna().copy()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                interval_width=0.95, changepoint_prior_scale=0.05, seasonality_prior_scale=10)
    m.add_seasonality(name="semi_annual", period=182.5, fourier_order=3)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq="MS")
    return m.predict(future)


@st.cache_data(show_spinner="Fitting national antigen forecasts (Prophet)...")
def national_forecasts(_nat, key: str) -> dict:
    """Return per-antigen series (% of 2024 baseline) + the at-risk summary table."""
    nat = _nat
    cutoff = nat["ds"].max()
    series = {}
    summary = []
    for antigen, col in C.ANTIGEN_TS.items():
        if col not in nat.columns:
            continue
        ts = nat[["ds", col]].rename(columns={col: "y"})
        fc = run_prophet(ts)
        base = ts[ts["ds"].dt.year == 2024]["y"].mean()
        fc_pct = fc.copy()
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            fc_pct[c] = fc[c] / base * 100
        obs_pct = ts["y"] / base * 100

        hist = fc_pct[fc_pct["ds"] <= cutoff]
        fore = fc_pct[fc_pct["ds"] > cutoff].reset_index(drop=True)
        fore["months_ahead"] = ((fore["ds"].dt.year - cutoff.year) * 12
                                + (fore["ds"].dt.month - cutoff.month))
        lo80 = fore["yhat"] - (fore["yhat"] - fore["yhat_lower"]) * C.PI_80_FACTOR
        hi80 = fore["yhat"] + (fore["yhat_upper"] - fore["yhat"]) * C.PI_80_FACTOR
        win = fore[(fore["months_ahead"] >= C.AT_RISK_WINDOW_MONTHS[0])
                   & (fore["months_ahead"] <= C.AT_RISK_WINDOW_MONTHS[1])]

        min_pct = float(fore["yhat"].min())
        min_when = fore.loc[fore["yhat"].idxmin(), "ds"]
        crosses = bool((win["yhat"] < C.THRESHOLD_PCT).any())
        below = fore.loc[fore["yhat"] < C.THRESHOLD_PCT, "ds"]
        first_below = below.min() if len(below) else None

        series[antigen] = dict(
            obs_x=ts["ds"].dt.strftime("%Y-%m-%d").tolist(), obs_y=obs_pct.tolist(),
            hist_x=hist["ds"].dt.strftime("%Y-%m-%d").tolist(), hist_y=hist["yhat"].tolist(),
            fore_x=fore["ds"].dt.strftime("%Y-%m-%d").tolist(), fore_y=fore["yhat"].tolist(),
            lo95=fore["yhat_lower"].tolist(), hi95=fore["yhat_upper"].tolist(),
            lo80=lo80.tolist(), hi80=hi80.tolist(),
            cutoff=cutoff.strftime("%Y-%m-%d"), at_risk=crosses,
        )
        summary.append({
            "Antigen": antigen,
            "Min forecast (% of 2024 baseline)": round(min_pct, 1),
            "Month of minimum": min_when.strftime("%b %Y"),
            "Crosses 80% in 6-12m": "Yes" if crosses else "No",
            "First month below 80%": first_below.strftime("%b %Y") if first_below is not None else "-",
        })
    return {"series": series, "summary": pd.DataFrame(summary)}


@st.cache_data(show_spinner="Screening LGAs for at-risk antigens...")
def lga_at_risk_screen(_dhis2, key: str) -> pd.DataFrame:
    """
    Fast linear-trend screen across all LGAs and antigens: project 12 months ahead and
    express as a percent of the LGA's 2024 baseline; flag projections below 80 percent.
    This is the always-on light path; the full per-LGA Prophet is the on-demand heavy run.
    """
    d = prep_dhis2(_dhis2)
    rows = []
    grp_cols = ["zone", "state", "lga"] if "zone" in d.columns else ["state", "lga"]
    for keys, g in d.groupby(grp_cols):
        g = g.sort_values("ds")
        rec = dict(zip(grp_cols, keys if isinstance(keys, tuple) else (keys,)))
        for antigen, col in C.ANTIGEN_TS.items():
            if col not in g:
                continue
            ts = g[["ds", col]].dropna()
            base = ts[ts["ds"].dt.year == 2024][col].mean()
            if not base or base <= 0 or len(ts) < 12:
                continue
            t = (ts["ds"] - ts["ds"].min()).dt.days.values.astype(float)
            y = ts[col].values.astype(float)
            try:
                slope, intercept = np.polyfit(t, y, 1)
            except Exception:
                continue
            t_future = t.max() + 365.0  # 12 months ahead
            proj = slope * t_future + intercept
            pct = proj / base * 100
            if pct < C.THRESHOLD_PCT:
                rows.append({**rec, "Antigen": antigen,
                             "2024 baseline (doses/month)": round(float(base), 1),
                             "Projected % of baseline (12m)": round(float(max(pct, 0)), 1)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Projected % of baseline (12m)").reset_index(drop=True)
    return out
