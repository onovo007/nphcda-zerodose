"""
Data-quality assessment for the uploaded DHIS2 export and NDHS file:
completeness, DHIS2 state-month reporting rate, missingness, freshness, outliers, and the
730-of-774 LGA reporting-coverage headline. Returns numbers + Plotly figures for the view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from theme import style_fig

TOTAL_LGAS = 774


def dhis2_quality(dhis2_prepped: pd.DataFrame) -> dict:
    d = dhis2_prepped
    n_states = d["state"].nunique() if "state" in d else 0
    n_lgas = d["lga"].nunique() if "lga" in d else 0
    months = sorted(d["ds"].dropna().unique())
    span = (pd.Timestamp(months[0]), pd.Timestamp(months[-1])) if months else (None, None)

    # Completeness of the count columns
    cc = [c for c in C.COUNT_COLS if c in d.columns]
    completeness = float((1 - d[cc].isna().to_numpy().mean()) * 100) if cc else 0.0

    # LGAs reporting non-zero Penta1 in the most recent year (the 730-of-774 headline)
    reporting = 0
    if "year" in d and "penta_1_count" in d:
        recent = d[d["year"] == d["year"].max()]
        rep = recent.groupby("lga")["penta_1_count"].sum()
        reporting = int((rep > 0).sum())

    # DHIS2 reporting rate: filled state-month cells vs the complete grid
    rate = np.nan
    if "state" in d and months:
        grid = d["state"].nunique() * len(months)
        filled = d.dropna(subset=["penta_1_count"]).groupby(["state", "ds"]).ngroups
        rate = round(filled / grid * 100, 1) if grid else np.nan

    return {
        "n_states": n_states, "n_lgas": n_lgas, "n_months": len(months),
        "span": span, "completeness": round(completeness, 1),
        "reporting_lgas": reporting, "total_lgas": TOTAL_LGAS,
        "reporting_rate": rate,
    }


def missingness_by_state(dhis2_prepped: pd.DataFrame) -> go.Figure:
    """Heatmap of state-month Penta1 reporting (present vs missing)."""
    d = dhis2_prepped.dropna(subset=["ds"])
    piv = (d.groupby(["state", "ds"])["penta_1_count"].sum().reset_index()
           .pivot(index="state", columns="ds", values="penta_1_count"))
    present = (~piv.isna() & (piv.fillna(0) > 0)).astype(int)
    fig = go.Figure(go.Heatmap(
        z=present.values,
        x=[pd.Timestamp(c).strftime("%b-%y") for c in present.columns],
        y=present.index.tolist(),
        colorscale=[[0, "#F4C7C3"], [1, C.STEEL]],
        showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y}<br>%{x}<br>%{z}<extra></extra>",
    ))
    fig.update_layout(title="DHIS2 reporting by state and month (filled vs missing)",
                      height=760, yaxis=dict(autorange="reversed"))
    return style_fig(fig)


def outliers(national_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score outliers in the national antigen series (EpidPredict pattern)."""
    nat = national_monthly_df
    rows = []
    for label, col in C.ANTIGEN_TS.items():
        if col not in nat:
            continue
        y = nat[col].astype(float)
        mu, sd = y.mean(), (y.std() + 1e-9)
        z = (y - mu) / sd
        for i in nat.index:
            zi = float(z.loc[i])
            if abs(zi) <= 2:
                continue
            doses = float(nat.loc[i, col])
            rows.append({
                "Antigen": label,
                "Month": pd.Timestamp(nat.loc[i, "ds"]).strftime("%b %Y"),
                "Doses": int(doses),
                "Series mean": int(mu),
                "Z-score": round(zi, 2),
                "Deviation from mean (%)": round((doses - mu) / mu * 100, 1) if mu else 0.0,
                "Direction": "Spike" if zi > 0 else "Drop",
                "Severity": "High" if abs(zi) >= 3 else "Moderate",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(df["Z-score"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df
