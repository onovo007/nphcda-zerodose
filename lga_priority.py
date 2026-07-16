"""
LGA Priority List and Archetypes page - NPHCDA filters, views and downloads the ranked local
governments (worst zero-dose burden first), each tagged with its archetype, equity-deprivation tier
and matched intervention bundle. Archetypes come from the LGA covariate/archetype analysis (IHME, DHS,
Meta, Weiss, ACLED); zero-dose figures come from the platform model.
"""
import pandas as pd
import streamlit as st

import config as C
from theme import clean, domain_banner

RANK_FILE = C.DATA_DIR / "lga_priority_ranking.csv"
ARCH_FILE = C.DATA_DIR / "lga_archetype_summary.csv"

TIER_ORDER = ["Critical", "High", "Moderate", "Low"]
TIER_COLOR = {"Critical": "#B2182B", "High": "#EF8A62", "Moderate": "#F0C24B", "Low": "#9ECAE1"}
TIER_DEF = {"Critical": "worst quarter on the equity index (most deprived)", "High": "second quarter",
            "Moderate": "third quarter", "Low": "least-deprived quarter"}
ARCH_COLOR = {"Remote Rural / Hard-to-Reach": "#7A1616", "Conflict-Affected / Nomadic": "#D6604D",
              "Riverine / Geographically Isolated": "#2C7FB8",
              "Peri-urban / Migrant Dense (transitional)": "#C8902A",
              "Urban Slums (better-off urban core)": "#1C7A3D"}
ARCH_DEF = {
    "Remote Rural / Hard-to-Reach": "Deep-north deprivation: lowest education, highest undernutrition, lowest facility delivery",
    "Conflict-Affected / Nomadic": "Insecurity belt: highest political-violence fatalities, high dropout, weak antenatal and delivery",
    "Riverine / Geographically Isolated": "Creek and hard-to-reach terrain: extreme travel time, low improved water",
    "Peri-urban / Migrant Dense (transitional)": "Transitional: moderate on all fronts, fair care-seeking",
    "Urban Slums (better-off urban core)": "Better-off urban core with residual gaps in informal settlements",
}


@st.cache_data(show_spinner=False)
def _load(path):
    return pd.read_csv(path)


def _chips(mapping, defs):
    rows = "".join(
        f"<div style='display:flex;align-items:center;margin:2px 16px 2px 0'>"
        f"<span style='width:14px;height:14px;border-radius:3px;background:{c};display:inline-block;"
        f"margin-right:7px'></span><b>{clean(k)}</b>&nbsp;-&nbsp;<span style='color:#555'>{clean(defs[k])}</span></div>"
        for k, c in mapping.items())
    st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{rows}</div>", unsafe_allow_html=True)


def render():
    domain_banner("_banner_d5.jpg", "LGA Priority and Archetypes",
                  "Every reporting local government ranked by its estimated zero-dose children, tagged "
                  "with its archetype, equity-deprivation tier and matched intervention bundle.")
    if not RANK_FILE.exists():
        st.warning("The LGA priority file is not bundled. Run the archetype pipeline to generate it.")
        return
    df = _load(str(RANK_FILE))

    with st.expander("What the classifications mean (equity tier and archetype)", expanded=True):
        st.markdown("**Equity-deprivation tier** (from the LGA composite index of remoteness, low "
                    "education, poverty and low wealth; quartiles across the 774 local governments):")
        _chips(TIER_COLOR, TIER_DEF)
        st.markdown("**Archetype** (five data-driven groups from agglomerative clustering of 15 "
                    "local-government covariates):")
        _chips(ARCH_COLOR, ARCH_DEF)

    c1, c2, c3 = st.columns(3)
    states = c1.multiselect("State", sorted(df["State"].dropna().unique()))
    archs = c2.multiselect("Archetype", [a for a in ARCH_COLOR if a in set(df["Archetype"])])
    tiers = c3.multiselect("Equity tier", [t for t in TIER_ORDER if t in set(df["Equity tier"])])
    only_priority = st.checkbox("Show only TOP PRIORITY local governments "
                                "(high burden and high deprivation)", value=False)

    f = df.copy()
    if states:
        f = f[f["State"].isin(states)]
    if archs:
        f = f[f["Archetype"].isin(archs)]
    if tiers:
        f = f[f["Equity tier"].isin(tiers)]
    if only_priority:
        f = f[f["Priority flag"] == "TOP PRIORITY"]

    tot = int(pd.to_numeric(f["Zero-dose children"], errors="coerce").fillna(0).sum())
    nat = int(pd.to_numeric(df["Zero-dose children"], errors="coerce").fillna(0).sum())
    st.write(clean(f"**{len(f)} local governments** shown, holding **{tot:,} zero-dose children** "
                   f"(of {nat:,} across all reporting local governments)."))

    # Plain, robust table (no heavy per-row Styler that can choke Streamlit on 700 rows); the equity
    # tier and archetype colours are explained in the legend above.
    st.dataframe(f, use_container_width=True, height=460, hide_index=True, column_config={
        "Equity index": st.column_config.NumberColumn(format="%.2f"),
        "Zero-dose children": st.column_config.NumberColumn(format="%d"),
        "Zero-dose rate (%)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("Download this list (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "NPHCDA_LGA_priority_list.csv", "text/csv")

    if ARCH_FILE.exists():
        with st.expander("The five LGA archetypes - determinants, intervention levers and evidence"):
            st.dataframe(_load(str(ARCH_FILE)), use_container_width=True)
            st.caption(clean("Interventions matched to each archetype's binding constraint using an "
                             "evidence-to-barrier method (WHO Reaching Every District/Community; Gavi "
                             "zero-dose IRMMA; WHO/UNICEF Big Catch-Up; PIRI; BeSD)."))
    st.caption(clean("Zero-dose figures are model estimates. Archetype and equity tier use modelled "
                     "covariate surfaces (2014-2021). Local governments without DTP1 reporting are not "
                     "ranked here."))
