"""
LGA Priority List and Archetypes page - lets NPHCDA filter, view and download the ranked local
governments (worst zero-dose burden first), enriched with the archetype and equity tier. The archetype
classification comes from the LGA covariate/archetype analysis (IHME, DHS, Meta, Weiss, ACLED); the
zero-dose figures come from the platform model.
"""
import pandas as pd
import streamlit as st

import config as C
from theme import clean

RANK_FILE = C.DATA_DIR / "lga_priority_ranking.csv"
ARCH_FILE = C.DATA_DIR / "lga_archetype_summary.csv"


@st.cache_data(show_spinner=False)
def _load(path):
    return pd.read_csv(path)


def render():
    st.header("LGA priority list and archetypes")
    st.write(clean(
        "All reporting local governments ranked by their estimated number of zero-dose children "
        "(worst first), each tagged with its archetype and equity-deprivation tier. Use the filters to "
        "pull the local governments you want and download the list for microplanning."))

    if not RANK_FILE.exists():
        st.warning("The LGA priority file is not bundled. Run the archetype pipeline to generate it.")
        return
    df = _load(str(RANK_FILE))

    # filters
    c1, c2, c3 = st.columns(3)
    states = c1.multiselect("State", sorted(df["State"].dropna().unique()))
    archs = c2.multiselect("Archetype", sorted(df["Archetype"].dropna().unique()))
    tiers = c3.multiselect("Equity tier (1 = most deprived)", sorted(df["Equity tier"].dropna().unique()))
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
        f = f[f["Priority flag"].astype(str).str.contains("PRIORITY", na=False)]

    tot = int(pd.to_numeric(f["Zero-dose children"], errors="coerce").fillna(0).sum())
    st.write(clean(f"**{len(f)} local governments** shown, holding **{tot:,} zero-dose children** "
                   f"(of {int(pd.to_numeric(df['Zero-dose children'], errors='coerce').fillna(0).sum()):,} "
                   "nationally)."))

    def _hi(row):
        base = "background-color:#F7D9D4" if "PRIORITY" in str(row["Priority flag"]) else ""
        return [base] * len(row)

    st.dataframe(f.style.apply(_hi, axis=1), use_container_width=True, height=460)
    st.download_button("Download this list (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "NPHCDA_LGA_priority_list.csv", "text/csv")

    if ARCH_FILE.exists():
        with st.expander("The five LGA archetypes - determinants, intervention levers and evidence"):
            a = _load(str(ARCH_FILE))
            st.dataframe(a, use_container_width=True)
            st.caption(clean("Archetypes from agglomerative (Ward) clustering of 15 local-government "
                             "covariates. Interventions matched to each archetype's binding constraint "
                             "using an evidence-to-barrier method (WHO Reaching Every District/Community; "
                             "Gavi zero-dose IRMMA; WHO/UNICEF Big Catch-Up; PIRI; BeSD)."))
    st.caption(clean("Zero-dose figures are model estimates. Archetype and equity tier use covariate "
                     "surfaces (2014-2021). Local governments without DTP1 reporting are not ranked here."))
