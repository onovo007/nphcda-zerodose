"""
Domain 5 - zero-dose estimates from two methods, served for the bundled project data.

Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation (773 LGAs).
    State zero-dose rates from the hierarchical Beta regression on the NDHS 2008-2024 panel,
    forecast to 2026-2028; each state's 2026 burden is allocated to its LGAs using the LGA's
    DHIS2 Penta1 share (calibrated to the state posterior) and its NPC 2022 population share.
Method 2: Bayesian small-area estimation (SAE) (774 LGAs).
    An LGA-level model fitted directly to the NDHS state observations through an aggregation
    likelihood (design effect 2), with six LGA covariates and a BYM2 spatial effect on the GRID3
    LGA adjacency; every LGA receives its own posterior rate and credible interval.

Both use the same denominator: the 12-23-month cohort = state under-five (2024 projection) / 5,
shared to LGAs by NPC 2022 population. The full fits are heavy (PyMC, thousands of parameters), so
the outputs of the executed notebooks are bundled in data/sample/two_methods and loaded here when
the active data is the bundled project data. Uploaded data runs Method 1 live (models/d5_zerodose).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

import config as C
import names as N

TM_DIR = C.DATA_DIR / "two_methods"

M1 = "Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation"
M2 = "Method 2: Bayesian small-area estimation (SAE)"
M1_SHORT, M2_SHORT = "Method 1", "Method 2 (SAE)"
SHARES = (50, 60, 80)


def _fp_ndhs(ndhs_long: pd.DataFrame) -> str:
    z = pd.to_numeric(ndhs_long.get("zero_dose_pct"), errors="coerce")
    return f"{len(ndhs_long)}|{round(float(z.sum()), 2)}"


def is_bundled(data: dict) -> bool:
    """True when the active NDHS and DHIS2 files are the bundled project data."""
    try:
        from models.d1_forecast import _fp_d1
        meta = json.loads((TM_DIR / "meta.json").read_text())
        return (meta.get("ndhs_fp") == _fp_ndhs(data["ndhs_long"])
                and meta.get("dhis2_fp") == _fp_d1(data["dhis2"]))
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load() -> dict:
    """All bundled two-method tables."""
    r = lambda f: pd.read_csv(TM_DIR / f)  # noqa: E731
    out = {
        "lga": r("lga_both_methods_774.csv"),
        "state": r("state_both_methods.csv"),
        "national_zone": r("national_zone_both_methods.csv"),
        "pareto_scenarios": r("pareto_scenarios.csv"),
        "pareto_curves": r("pareto_curves.csv"),
        "topk": r("topk_shares.csv"),
        "validation": r("validation_both_methods.csv"),
        "zone_validation": r("zone_validation_nmdhs.csv"),
        "archetype": r("archetype_both_methods.csv"),
        "m1_state_fc": r("method1_state_forecasts_2026_2028.csv"),
        "m2_zone": r("method2_sae_zone_estimates_2026.csv"),
        "m2_ppc": r("method2_sae_posterior_predictive_check.csv"),
        "keys": r("lga_map_keys_774.csv"),
        "nmdhs": r("nmdhs_2025_26_penta1_by_state_zone.csv"),
        "diag_m1": r("method1_mcmc_diagnostics.csv"),
        "diag_m2": r("method2_sae_mcmc_diagnostics.csv"),
        "summary": json.loads((TM_DIR / "results.json").read_text()),
    }
    out["lga"] = out["lga"].loc[:, ~out["lga"].columns.duplicated()]
    out["lga"] = out["lga"].merge(out["keys"][["lga_uid", "state_key", "geo_lga_key"]],
                                  on="lga_uid", how="left")
    return out


def _p(m: str) -> str:
    return "m1" if m == M1 else "m2"


# --------------------------------------------------------------------------------------
# State results in the schema used by the existing figure builders
# --------------------------------------------------------------------------------------
def state_res(tm: dict, method: str, ndhs_long: pd.DataFrame) -> pd.DataFrame:
    """State table with zd_obs_*, zd_pred_<yr>_mean/lo95/hi95, zd_count_<yr>, risk index, tier."""
    st_ = tm["state"].copy()
    nd = ndhs_long.copy()
    nd["state"] = nd["state"].astype(str).str.strip()
    obs = nd.pivot_table(index="state", columns="year", values="zero_dose_pct", aggfunc="first")
    p = _p(method)
    rows = []
    fc = tm["m1_state_fc"].set_index("state")
    for _, s in st_.iterrows():
        r = {"state": s["state"], "zone": s["zone"], "cohort_12_23m": s["cohort_12_23m"],
             "lgas": s["lgas"], "nmdhs_2025_26": s["nmdhs_2025_26"]}
        for y in C.NDHS_YEARS:
            r[f"zd_obs_{y}"] = float(obs.loc[s["state"], y]) if (s["state"] in obs.index and y in obs.columns) else np.nan
        r["zd_pred_2026_mean"] = s[f"{p}_rate"]
        r["zd_pred_2026_lo95"] = s[f"{p}_rate_lo95"]
        r["zd_pred_2026_hi95"] = s[f"{p}_rate_hi95"]
        r["zd_count_2026"] = s[f"{p}_children"]
        r["zd_count_2026_lo95"] = s[f"{p}_children_lo95"]
        r["zd_count_2026_hi95"] = s[f"{p}_children_hi95"]
        r["lga_rate_range"] = s[f"{p}_lga_rate_range"]
        if p == "m1" and s["state"] in fc.index:
            f = fc.loc[s["state"]]
            for y in (2027, 2028):
                r[f"zd_pred_{y}_mean"] = f[f"rate_{y}"]
                r[f"zd_pred_{y}_lo95"] = f[f"rate_{y}_lo95"]
                r[f"zd_pred_{y}_hi95"] = f[f"rate_{y}_hi95"]
                r[f"zd_count_{y}"] = f[f"rate_{y}"] / 100 * s["cohort_12_23m"]
        rows.append(r)
    res = pd.DataFrame(rows)
    res["score_rate"] = N.minmax_scale(res["zd_pred_2026_mean"])
    res["score_count"] = N.minmax_scale(res["zd_count_2026"])
    res["score_trend"] = N.minmax_scale((res["zd_obs_2024"] - res["zd_obs_2018"]).fillna(0))
    res["score_uncert"] = N.minmax_scale(res["zd_pred_2026_hi95"] - res["zd_pred_2026_lo95"])
    res["risk_index"] = (0.45 * res["score_rate"] + 0.30 * res["score_count"]
                         + 0.15 * res["score_trend"] + 0.10 * res["score_uncert"])
    res["state_rank"] = res["risk_index"].rank(ascending=False).astype(int)
    res["priority_tier"] = pd.cut(res["risk_index"], bins=[-np.inf, 25, 50, 75, np.inf],
                                  labels=["Tier 4: Lower", "Tier 3: Moderate",
                                          "Tier 2: High", "Tier 1: Critical"])
    return res.sort_values("state_rank").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# LGA tables
# --------------------------------------------------------------------------------------
def band_for(cum: pd.Series) -> pd.Series:
    """Priority band from the cumulative share of the national burden (50/60/80 cut-offs)."""
    return pd.Series(np.select(
        [cum <= 50, cum <= 60, cum <= 80],
        ["A: first 50%", "B: 50-60%", "C: 60-80%"], "D: 80-100%"), index=cum.index)


def lga_ranked(tm: dict, method: str) -> pd.DataFrame:
    """LGAs ranked by estimated zero-dose children under the chosen method, with the cumulative
    share of the national burden and the 50/60/80 priority band."""
    lg = tm["lga"].copy()
    p = _p(method)
    lg = lg[lg[f"{p}_children"].notna()].sort_values(f"{p}_children", ascending=False).reset_index(drop=True)
    total = float(lg[f"{p}_children"].sum())
    cum = lg[f"{p}_children"].cumsum() / total * 100
    out = pd.DataFrame({
        "Burden rank": np.arange(1, len(lg) + 1),
        "State": lg["state"], "LGA": lg["lga_clean"], "Zone": lg["zone"],
        "Zero-dose children (est)": lg[f"{p}_children"].round(0).astype(int),
        "Children low (95%)": lg[f"{p}_children_lo95"].round(0).astype(int),
        "Children high (95%)": lg[f"{p}_children_hi95"].round(0).astype(int),
        "Zero-dose rate (%)": lg[f"{p}_rate"].round(1),
        "Rate low (95%)": lg[f"{p}_rate_lo95"].round(1),
        "Rate high (95%)": lg[f"{p}_rate_hi95"].round(1),
        "Cohort 12-23 months": lg["cohort_12_23m"].round(0).astype(int),
        "Cumulative % of burden": cum.round(1),
        "Priority band": band_for(cum).values,
        "Probability in top 155 (%)": (lg[f"{p}_p_top155"] * 100).round(0),
        "Archetype": lg["archetype_type"],
        "Other method rank": lg["m2_rank" if p == "m1" else "m1_rank"],
    })
    out["State rank"] = out.groupby("State")["Zero-dose children (est)"].rank(
        ascending=False, method="first").astype(int)
    out["Other method rank"] = pd.to_numeric(out["Other method rank"], errors="coerce").round(0).astype("Int64")
    out = out.rename(columns={"Other method rank": ("Method 2 (SAE) rank" if p == "m1" else "Method 1 rank")})
    return out


def n_for_share(ranked: pd.DataFrame, share: float) -> int:
    """Smallest number of top-ranked LGAs whose cumulative burden reaches the share."""
    c = ranked["Zero-dose children (est)"].astype(float).values
    cum = np.cumsum(c) / c.sum() * 100
    return int(np.searchsorted(cum, share) + 1)


def comparison_table(tm: dict) -> pd.DataFrame:
    """One row per LGA with both methods side by side."""
    lg = tm["lga"]
    out = pd.DataFrame({
        "State": lg["state"], "LGA": lg["lga_clean"], "Zone": lg["zone"],
        "Method 1 rank": pd.to_numeric(lg["m1_rank"], errors="coerce").round(0).astype("Int64"),
        "Method 2 (SAE) rank": lg["m2_rank"].round(0).astype(int),
        "Method 1 children": lg["m1_children"].round(0).astype("Int64"),
        "Method 2 (SAE) children": lg["m2_children"].round(0).astype(int),
        "Method 1 rate (%)": lg["m1_rate"].round(1),
        "Method 2 (SAE) rate (%)": lg["m2_rate"].round(1),
        "Method 2 (SAE) rate 95% CrI": [f"{a:.0f}-{b:.0f}" for a, b in zip(lg["m2_rate_lo95"], lg["m2_rate_hi95"])],
        "In top 155, Method 1": np.where(pd.to_numeric(lg["m1_rank"], errors="coerce") <= 155, "Yes", "No"),
        "In top 155, Method 2 (SAE)": np.where(lg["m2_rank"] <= 155, "Yes", "No"),
        "Archetype": lg["archetype_type"],
        "Method 1 status": lg["m1_status"],
    })
    return out.sort_values("Method 2 (SAE) rank").reset_index(drop=True)


def map_frame(tm: dict, method: str) -> pd.DataFrame:
    """state / lga / state_key / lga_key / rate for the LGA maps and Gi*."""
    lg = tm["lga"]
    p = _p(method)
    return pd.DataFrame({"state": lg["state"], "lga": lg["lga_clean"], "state_key": lg["state_key"],
                         "lga_key": lg["geo_lga_key"], "zd_rate": lg[f"{p}_rate"],
                         "zd_children": lg[f"{p}_children"]})


def gi_lga(method: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(TM_DIR / f"gi_lga_{_p(method)}.parquet")
    except Exception:
        return None


def gi_state(method: str, year: int) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(TM_DIR / f"gi_state_{_p(method)}_{year}.parquet")
    except Exception:
        return None


def diagnostics(tm: dict, method: str) -> pd.DataFrame:
    """Convergence table for the hyperparameters (no per-LGA or per-state vectors)."""
    d = tm["diag_m1" if method == M1 else "diag_m2"].rename(columns={"Unnamed: 0": "Parameter"})
    if "Parameter" not in d.columns:
        d = d.rename(columns={d.columns[0]: "Parameter"})
    d = d[~d["Parameter"].astype(str).str.contains(r"\[")]
    keep = ["Parameter", "mean", "sd", "ess_bulk", "ess_tail", "r_hat"]
    d = d[[c for c in keep if c in d.columns]].copy()
    for c in ("mean", "sd"):
        d[c] = d[c].round(3)
    for c in ("ess_bulk", "ess_tail"):
        d[c] = d[c].round(0).astype(int)
    d["r_hat"] = d["r_hat"].round(3)
    return d.rename(columns={"mean": "Posterior mean", "sd": "SD", "ess_bulk": "ESS (bulk)",
                             "ess_tail": "ESS (tail)", "r_hat": "R-hat"})


# --------------------------------------------------------------------------------------
# LGA priority list: burden rank (chosen method) + equity-deprivation tier + archetype
# --------------------------------------------------------------------------------------
TOP_N = 155  # top 20 percent of Nigeria's 774 LGAs


@st.cache_data(show_spinner=False)
def equity_index() -> pd.DataFrame:
    """LGA equity-deprivation index: equal-weight mean of four min-max (0-100) components -
    remoteness (travel time to a facility), low women's education, poverty and low relative wealth;
    tiers are quartiles across the 774 LGAs (Critical = most deprived quarter)."""
    cov = pd.read_csv(TM_DIR / "lga_archetype_covariates_774.csv")

    def mm(s):
        s = pd.to_numeric(s, errors="coerce")
        return (s - s.min()) / (s.max() - s.min()) * 100
    comp = pd.DataFrame({"remoteness": mm(cov["travel_time_hc"]),
                         "low_education": mm(-cov["edu_mean_years_women_15_49"]),
                         "poverty": mm(cov["poverty_rate"]),
                         "low_wealth": mm(-cov["relative_wealth_index"])})
    out = cov[["lga_uid"]].copy()
    out["Equity index"] = comp.mean(axis=1).round(1)
    out["Equity tier"] = pd.qcut(out["Equity index"].rank(method="first"), 4,
                                 labels=["Low", "Moderate", "High", "Critical"]).astype(str)
    return out


def priority_table(tm: dict, method: str, bundles: dict, evidence: dict) -> pd.DataFrame:
    """Every LGA ranked by estimated zero-dose children under the chosen method, with equity tier,
    archetype, matched intervention bundle and the TOP PRIORITY flag (top 155 by burden and a
    Critical or High equity tier)."""
    lg = tm["lga"].merge(equity_index(), on="lga_uid", how="left")
    p = _p(method)
    lg = lg.sort_values(f"{p}_children", ascending=False, na_position="last").reset_index(drop=True)
    rank = pd.Series(np.arange(1, len(lg) + 1), dtype="Int64")
    rank[lg[f"{p}_children"].isna()] = pd.NA
    out = pd.DataFrame({
        "Burden rank": rank, "State": lg["state"], "Zone": lg["zone"], "LGA": lg["lga_clean"],
        "Zero-dose children": lg[f"{p}_children"].round(0).astype("Int64"),
        "Children 95% CrI": [f"{a:,.0f}-{b:,.0f}" if pd.notna(a) else "" for a, b in
                             zip(lg[f"{p}_children_lo95"], lg[f"{p}_children_hi95"])],
        "Zero-dose rate (%)": lg[f"{p}_rate"].round(1),
        "Equity index": lg["Equity index"], "Equity tier": lg["Equity tier"],
        "Archetype": lg["archetype_type"],
        "Intervention bundle": lg["archetype_type"].map(bundles),
        "Evidence base (method and citation)": lg["archetype_type"].map(evidence),
    })
    import evidence_views as EV
    b = EV.lga_barriers()
    if b is not None:
        b = lg[["lga_uid"]].merge(b, on="lga_uid", how="left")
        pos = out.columns.get_loc("Intervention bundle")
        out.insert(pos, "Dominant barrier", b["dominant_barrier"].values)
        out.insert(pos + 1, "Flagged barriers", b["barriers_flagged"].values)
        out.insert(pos + 2, "Candidate components (LGA barriers)", b["candidate_components"].values)
        out.insert(pos + 3, "Profile membership", b["membership"].values)
        out = out.rename(columns={"Intervention bundle": "Profile package"})
    out.insert(out.columns.get_loc("Equity index"), "P(top 155), Method 2 (%)", (lg["m2_p_top155"] * 100).round(0).values)
    top = (out["Burden rank"].fillna(10 ** 6) <= TOP_N) & out["Equity tier"].isin(["Critical", "High"])
    out["Priority flag"] = np.where(top, "TOP PRIORITY", "Standard")
    if p == "m1":
        out.loc[out["Burden rank"].isna(), "Priority flag"] = "Not estimated (no Penta1 data 2021-2024)"
    return out
