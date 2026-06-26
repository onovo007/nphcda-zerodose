"""
Spatial layer: load the shipped GRID3 GeoJSON (state + LGA), join model outputs onto the
geometry, and compute Getis-Ord Gi* hotspot classes with a k=5 nearest-neighbour weight
(libpysal + esda), exactly as in D5 cell 27. Falls back gracefully if a join is sparse.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import names as N


@st.cache_data(show_spinner=False)
def load_geojson(level: str) -> dict:
    path = C.GEO_STATES if level == "state" else C.GEO_LGAS
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def load_gdf(level: str):
    import geopandas as gpd
    path = C.GEO_STATES if level == "state" else C.GEO_LGAS
    return gpd.read_file(path)


@st.cache_data(show_spinner="Computing Gi* hotspots...")
def lga_gi_star(_lga_df, key: str, value_col: str = "zd_proxy_pct") -> pd.DataFrame:
    """
    Return a DataFrame keyed by (state_key, lga_key) with Gi* z, p and hotspot class.
    _lga_df must carry 'state', 'lga' and the value column.
    """
    from libpysal.weights import KNN
    from esda.getisord import G_Local

    gdf = load_gdf("lga").copy()
    work = _lga_df.copy()
    work["state_key"] = work["state"].map(N.nstate)
    work["lga_key"] = work["lga"].map(N.nlga)
    merged = gdf.merge(
        work[["state_key", "lga_key", value_col]].drop_duplicates(["state_key", "lga_key"]),
        on=["state_key", "lga_key"], how="left",
    )
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
    # Fill unmatched geometry with the national median so the weight matrix stays complete
    merged[value_col] = merged[value_col].fillna(merged[value_col].median())

    try:
        w = KNN.from_dataframe(merged, k=C.LGA_KNN)
        w.transform = "r"
        gi = G_Local(merged[value_col].values, w, star=True, seed=42)
        z = gi.Zs
        p = gi.p_sim
        merged["gi_z"] = z
        merged["gi_p"] = p
        merged["gi_class"] = [N.hotspot_class(zz, pp) for zz, pp in zip(z, p)]
    except Exception as exc:  # pragma: no cover - defensive
        merged["gi_z"] = np.nan
        merged["gi_p"] = np.nan
        merged["gi_class"] = "Not Significant"
        st.info(f"Gi* computation unavailable, showing rates only ({exc}).")
    return merged[["state_key", "lga_key", "state", "lga", value_col, "gi_z", "gi_p", "gi_class"]]


@st.cache_data(show_spinner=False)
def state_gi_star(_state_df, key: str, value_col: str = "value") -> pd.DataFrame:
    """Getis-Ord Gi* on the state surface (GRID3 admin-1, Queen contiguity), used for the
    forecast zero-dose hotspot maps (Figure 11). _state_df must carry 'state_key' and value_col.
    Returns a DataFrame keyed by state_key with the value, Gi* z, p and hotspot class.
    """
    from libpysal.weights import Queen, KNN
    from esda.getisord import G_Local

    gdf = load_gdf("state").copy()
    work = _state_df[["state_key", value_col]].drop_duplicates("state_key")
    merged = gdf.merge(work, on="state_key", how="left")
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce")
    merged[value_col] = merged[value_col].fillna(merged[value_col].median())

    try:
        try:
            w = Queen.from_dataframe(merged, use_index=False)
        except Exception:
            w = KNN.from_dataframe(merged, k=C.LGA_KNN)
        w.transform = "r"
        gi = G_Local(merged[value_col].values, w, star=True, seed=42)
        merged["gi_z"] = gi.Zs
        merged["gi_p"] = gi.p_sim
        merged["gi_class"] = [N.hotspot_class(zz, pp) for zz, pp in zip(gi.Zs, gi.p_sim)]
    except Exception as exc:  # pragma: no cover - defensive
        merged["gi_z"] = np.nan
        merged["gi_p"] = np.nan
        merged["gi_class"] = "Not Significant"
        st.info(f"State Gi* computation unavailable, showing rates only ({exc}).")
    return pd.DataFrame(merged.drop(columns="geometry"))[
        ["state_key", "state", value_col, "gi_z", "gi_p", "gi_class"]]
