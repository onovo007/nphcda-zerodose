"""
Data ingestion: load the bundled sample inputs, read user uploads, validate against the
declared schemas, and provide light shared transforms (DHIS2 parsing, dropout columns).

Everything is cached on content so re-runs are instant but genuinely computed from the data.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st

import config as C

DATASET_KEYS = ["dhis2", "ndhs_long", "model_dataset", "under5", "lga_population"]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_sample() -> dict[str, pd.DataFrame]:
    """Load every bundled sample input as a raw DataFrame."""
    out = {}
    for key, path in C.SAMPLE_FILES.items():
        try:
            out[key] = pd.read_csv(path)
        except Exception:
            out[key] = None
    return out


def read_uploaded(file) -> pd.DataFrame | None:
    if file is None:
        return None
    name = file.name.lower()
    raw = file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw))


def df_hash(df: pd.DataFrame | None) -> str:
    """Stable content key for cache invalidation (pandas-3.0 safe)."""
    if df is None:
        return "none"
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]
    except Exception:
        return hashlib.md5(f"{df.shape}|{list(df.columns)}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate(key: str, df: pd.DataFrame | None) -> dict:
    """Return {ok, status, missing, present, n_rows, note} for a dataset."""
    schema = C.SCHEMAS[key]
    if df is None:
        return {"ok": False, "status": "missing", "missing": schema["required"],
                "present": [], "n_rows": 0, "note": schema["note"], "label": schema["label"]}
    cols = {c.strip() for c in df.columns}
    missing = [c for c in schema["required"] if c not in cols]
    rec_missing = [c for c in schema.get("recommended", []) if c not in cols]
    ok = len(missing) == 0
    status = "ok" if ok and not rec_missing else ("partial" if ok else "invalid")
    return {
        "ok": ok, "status": status, "missing": missing, "rec_missing": rec_missing,
        "present": sorted(cols & set(schema["required"])), "n_rows": len(df),
        "note": schema["note"], "label": schema["label"],
    }


# --------------------------------------------------------------------------------------
# Shared light transforms (lifted from the notebooks)
# --------------------------------------------------------------------------------------
def prep_dhis2(df: pd.DataFrame) -> pd.DataFrame:
    """Parse period, coerce counts, add year/month and dropout columns (D1_D2 cell 9)."""
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    for c in C.COUNT_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["ds"] = pd.to_datetime(d["period"], format="%b-%y", errors="coerce")
    d["year"] = d["ds"].dt.year
    d["month"] = d["ds"].dt.month
    for c in ("zone", "state", "lga"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    return d


def national_monthly(dhis2_prepped: pd.DataFrame) -> pd.DataFrame:
    """National monthly antigen counts + dropout series (nat frame in D1_D2 cell 9)."""
    cc = [c for c in ["penta_1_count", "penta_3_count", "measles_1_count",
                      "measles_2_count", "bcg_count"] if c in dhis2_prepped.columns]
    nat = dhis2_prepped.groupby("ds")[cc].sum().reset_index()
    nat["dropout_p1p3"] = ((nat["penta_1_count"] - nat["penta_3_count"]) / nat["penta_1_count"] * 100).clip(-50, 100)
    if "measles_1_count" in nat:
        nat["dropout_p1m1"] = ((nat["penta_1_count"] - nat["measles_1_count"]) / nat["penta_1_count"] * 100).clip(-50, 100)
        if "measles_2_count" in nat:
            nat["dropout_m1m2"] = ((nat["measles_1_count"] - nat["measles_2_count"]) / nat["measles_1_count"] * 100).clip(-50, 100)
    return nat


def state_monthly(dhis2_prepped: pd.DataFrame) -> pd.DataFrame:
    """State-month aggregate with dropout columns (agg frame in D1_D2 cell 9)."""
    cc = [c for c in C.COUNT_COLS if c in dhis2_prepped.columns]
    agg = (dhis2_prepped.groupby(["state", "zone", "year", "month", "ds"])[cc]
           .sum().reset_index().sort_values(["state", "ds"]).reset_index(drop=True))
    agg["dropout_p1p3"] = ((agg["penta_1_count"] - agg["penta_3_count"]) / agg["penta_1_count"] * 100).clip(-50, 100)
    if "measles_1_count" in agg:
        agg["dropout_p1m1"] = ((agg["penta_1_count"] - agg["measles_1_count"]) / agg["penta_1_count"] * 100).clip(-50, 100)
        if "measles_2_count" in agg:
            agg["dropout_m1m2"] = ((agg["measles_1_count"] - agg["measles_2_count"]) / agg["measles_1_count"] * 100).clip(-50, 100)
    return agg


def prep_under5(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the under-five population file (D5 cell 15) -> state cohort 12-23m."""
    pop = df.copy().iloc[1:].copy()
    pop.columns = ["zone_abbr", "state_raw", "under5"][: pop.shape[1]]
    pop["under5_n"] = (pop["under5"].astype(str).str.replace(",", "", regex=False)
                       .str.strip().replace({"": np.nan}).astype(float))
    pop["cohort_12_23m"] = (pop["under5_n"] / 5).round(0)
    pop["jk"] = (pop["state_raw"].astype(str).str.strip().str.upper()
                 .str.replace(" ", "", regex=False).str.replace(",ABUJA", "", regex=False)
                 .str.replace(",", "", regex=False))
    return pop.dropna(subset=["under5_n"])


def prep_live_births(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the DHIS2 live-births file (period like '21-Jan' = %y-%b) -> ds/year + numeric count."""
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    if "live_births_count" in d.columns:
        d["live_births_count"] = pd.to_numeric(
            d["live_births_count"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    d["ds"] = pd.to_datetime(d["period"], format="%y-%b", errors="coerce")
    if d["ds"].isna().mean() > 0.5:  # fall back to the doses-file format if needed
        d["ds"] = pd.to_datetime(d["period"], format="%b-%y", errors="coerce")
    d["year"] = d["ds"].dt.year
    for c in ("zone", "state", "lga"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    return d


def national_live_births(df: pd.DataFrame, year: int = 2024) -> float | None:
    """National total DHIS2 live births for a year (the eligible-infant denominator candidate)."""
    d = prep_live_births(df)
    s = float(d[d["year"] == year]["live_births_count"].sum())
    return s if s > 0 else None


def _state_key(s) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalpha())


def _survey_weights(nd: pd.DataFrame, under5: pd.DataFrame | None):
    if under5 is None:
        return None
    try:
        u = prep_under5(under5)
        wmap = {_state_key(r["state_raw"]): r["under5_n"] for _, r in u.iterrows()}
        w = nd["State"].map(lambda s: wmap.get(_state_key(s)))
        return pd.to_numeric(w, errors="coerce") if w.notna().sum() >= 30 else None
    except Exception:
        return None


def national_survey_value(ndhs_antigens: pd.DataFrame, column: str,
                          under5: pd.DataFrame | None = None) -> float | None:
    """National value for any ndhs_antigens column (population-weighted by under-five, else mean)."""
    nd = ndhs_antigens.copy()
    nd.columns = [c.strip() for c in nd.columns]
    if column not in nd.columns:
        return None
    vals = pd.to_numeric(nd[column], errors="coerce")
    weights = _survey_weights(nd, under5)
    if weights is not None:
        m = vals.notna() & weights.notna()
        return round(float((vals[m] * weights[m]).sum() / weights[m].sum()), 1)
    return round(float(vals.mean()), 1)


def survey_national_coverage(ndhs_antigens: pd.DataFrame, under5: pd.DataFrame | None = None) -> dict:
    """National survey coverage per tracer antigen from ndhs_antigens2024 (population-weighted)."""
    cov = {}
    for antigen, col in C.SURVEY_ANTIGEN_COLS.items():
        v = national_survey_value(ndhs_antigens, col, under5)
        if v is not None:
            cov[antigen] = v
    return cov


def get_active_data() -> dict[str, pd.DataFrame] | None:
    """Return the dataset dict currently in session (uploaded overrides sample)."""
    return st.session_state.get("data")


def set_sample_data() -> None:
    st.session_state["data"] = load_sample()
    st.session_state["data_source"] = "Bundled project sample data"
