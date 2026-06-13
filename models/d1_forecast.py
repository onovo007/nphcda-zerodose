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
import names as N
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


def _horizon_to(last_ds, year: int = 2027, month: int = 12) -> int:
    """Prophet periods needed to reach the end of the target year."""
    return max((year - last_ds.year) * 12 + (month - last_ds.month), C.FORECAST_MONTHS)


@st.cache_data(show_spinner="Fitting national antigen forecasts (Prophet)...")
def national_forecasts(_nat, key: str, end_year: int = 2027,
                       metric: str = "baseline", cohort_annual: float | None = None) -> dict:
    """Return per-antigen series + the at-risk summary table.

    end_year sets how far ahead Prophet forecasts (to December of that year). The forecast
    always starts the month after the last observed data point.

    metric:
      - "baseline": each antigen is expressed as a percent of its own mean 2024 monthly doses
        (a denominator-free performance index; 80% = 80% of the 2024 level).
      - "coverage": WHO-style administrative coverage = monthly doses / (annual cohort / 12),
        using cohort_annual as the eligible denominator (here under-five / 5, a labelled proxy).
    """
    nat = _nat
    cutoff = nat["ds"].max()
    periods = _horizon_to(cutoff, year=end_year)
    coverage = (metric == "coverage" and cohort_annual and cohort_annual > 0)
    unit_label = "% coverage (admin)" if coverage else "% of 2024 baseline"
    value_col = f"Min forecast ({unit_label})"
    series = {}
    summary = []
    monthly_long = []
    for antigen, col in C.ANTIGEN_TS.items():
        if col not in nat.columns:
            continue
        ts = nat[["ds", col]].rename(columns={col: "y"})
        fc = run_prophet(ts, periods=periods)
        base = (cohort_annual / 12.0) if coverage else ts[ts["ds"].dt.year == 2024]["y"].mean()
        fr = fc[fc["ds"] > cutoff]
        monthly_long.append(pd.DataFrame({
            "Antigen": antigen, "Month": fr["ds"].dt.strftime("%Y-%m"), "Year": fr["ds"].dt.year,
            "Forecast doses": fr["yhat"].round(0), "Lower 95% doses": fr["yhat_lower"].round(0),
            "Upper 95% doses": fr["yhat_upper"].round(0),
            unit_label: (fr["yhat"] / base * 100).round(1) if base else None,
        }))
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
            value_col: round(min_pct, 1),
            "Month of minimum": min_when.strftime("%b %Y"),
            "Crosses 80% in 6-12m": "Yes" if crosses else "No",
            "First month below 80%": first_below.strftime("%b %Y") if first_below is not None else "-",
        })
    monthly = pd.concat(monthly_long, ignore_index=True) if monthly_long else pd.DataFrame()
    return {"series": series, "summary": pd.DataFrame(summary), "monthly": monthly,
            "unit_label": unit_label, "value_col": value_col}


@st.cache_data(show_spinner="Back-testing the national forecasts (hold-out)...")
def backtest_national(_nat, key: str, holdout: int = 6) -> pd.DataFrame:
    """Out-of-sample back-test: refit Prophet on all but the last `holdout` months and compare the
    forecast to the held-out actuals. Reports MAPE and empirical 95% prediction-interval coverage."""
    nat = _nat
    rows = []
    for antigen, col in C.ANTIGEN_TS.items():
        if col not in nat.columns:
            continue
        ts = nat[["ds", col]].rename(columns={col: "y"}).dropna().sort_values("ds")
        if len(ts) < holdout + 18:
            continue
        train, test = ts.iloc[:-holdout], ts.iloc[-holdout:]
        try:
            fc = run_prophet(train, periods=holdout)
        except Exception:
            continue
        m = test.merge(fc[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left").dropna(subset=["yhat"])
        if m.empty:
            continue
        ape = (np.abs(m["y"] - m["yhat"]) / m["y"].replace(0, np.nan)).dropna()
        mape = float(ape.mean() * 100) if len(ape) else float("nan")
        cov = float(((m["y"] >= m["yhat_lower"]) & (m["y"] <= m["yhat_upper"])).mean() * 100)
        rows.append({"Antigen": antigen, "Hold-out months": int(len(m)),
                     "MAPE (%)": round(mape, 1), "95% PI coverage (%)": round(cov, 0)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Forecasting antigen coverage by state (Prophet, 2026-2027)...")
def state_antigen_forecasts(_dhis2, key: str) -> pd.DataFrame:
    """Prophet forecast per state and antigen; returns monthly 2026-2027 projections (long)."""
    d = prep_dhis2(_dhis2)
    cc = [c for c in C.ANTIGEN_TS.values() if c in d.columns]
    zmap = (d.drop_duplicates("state").set_index("state")["zone"].to_dict()
            if "zone" in d.columns else {})
    out = []
    for state, g in d.groupby("state"):
        gm = g.groupby("ds")[cc].sum().reset_index().sort_values("ds")
        last = gm["ds"].max()
        per = _horizon_to(last)
        for antigen, col in C.ANTIGEN_TS.items():
            if col not in gm.columns:
                continue
            ts = gm[["ds", col]].rename(columns={col: "y"})
            base = ts[ts["ds"].dt.year == 2024]["y"].mean()
            try:
                fc = run_prophet(ts, periods=per)
            except Exception:
                continue
            fr = fc[(fc["ds"] > last) & (fc["ds"].dt.year.isin([2026, 2027]))]
            for _, r in fr.iterrows():
                out.append({
                    "zone": zmap.get(state, ""), "state": state, "antigen": antigen,
                    "month": r["ds"].strftime("%Y-%m"), "year": int(r["ds"].year),
                    "forecast_doses": round(float(r["yhat"])),
                    "lower95_doses": round(float(r["yhat_lower"])),
                    "upper95_doses": round(float(r["yhat_upper"])),
                    "pct_of_2024_baseline": round(float(r["yhat"]) / base * 100, 1) if base else None,
                })
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Projecting antigen coverage by LGA (trend, 2026-2027)...")
def lga_antigen_projections(_dhis2, key: str) -> pd.DataFrame:
    """
    Fast linear-trend monthly projections per LGA and antigen for 2026-2027 (microplanning).
    Trend method (not Prophet) so all 774 LGAs x 4 antigens return in seconds rather than the
    tens of minutes a full per-LGA Prophet run would take.
    """
    d = prep_dhis2(_dhis2)
    grp = ["zone", "state", "lga"] if "zone" in d.columns else ["state", "lga"]
    months = pd.date_range("2026-01-01", "2027-12-01", freq="MS")
    out = []
    for keys, g in d.groupby(grp):
        rec = dict(zip(grp, keys if isinstance(keys, tuple) else (keys,)))
        if "lga" in rec:
            rec["lga"] = N.clean_lga_name(rec["lga"])
        g = g.sort_values("ds")
        for antigen, col in C.ANTIGEN_TS.items():
            if col not in g.columns:
                continue
            ts = g[["ds", col]].dropna()
            base = ts[ts["ds"].dt.year == 2024][col].mean()
            if len(ts) < 12 or not base or base <= 0:
                continue
            t0 = ts["ds"].min()
            t = (ts["ds"] - t0).dt.days.values.astype(float)
            y = ts[col].values.astype(float)
            try:
                slope, intercept = np.polyfit(t, y, 1)
            except Exception:
                continue
            for m in months:
                proj = max(slope * (m - t0).days + intercept, 0.0)
                out.append({**rec, "antigen": antigen, "month": m.strftime("%Y-%m"),
                            "year": int(m.year), "projected_doses": round(float(proj)),
                            "pct_of_2024_baseline": round(float(proj) / base * 100, 1)})
    return pd.DataFrame(out)


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
        if "lga" in rec:
            rec["lga"] = N.clean_lga_name(rec["lga"])
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
                target_month = (ts["ds"].max() + pd.DateOffset(months=12)).strftime("%b %Y")
                rows.append({**rec, "Antigen": antigen,
                             "2024 baseline (doses/month)": round(float(base), 1),
                             "Projection month": target_month,
                             "Projected % of baseline (12m)": round(float(max(pct, 0)), 1)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Projected % of baseline (12m)").reset_index(drop=True)
    return out
