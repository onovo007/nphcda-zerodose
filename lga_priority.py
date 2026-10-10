"""
LGA Priority List and Archetypes page - NPHCDA filters, views and downloads the ranked local
governments (worst zero-dose burden first), each tagged with its contextual profile (archetype), dominant
barrier, candidate components, equity-deprivation tier and priority probability. Profiles come from Ward
clustering of 15 LGA indicators (IHME, DHS, Meta, Weiss, ACLED); zero-dose figures come from both Domain 5
methods. Contextual-barrier evidence is drawn by evidence_views.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
import ai
import evidence_views as EV
import viz
from theme import clean, domain_banner, kpi_row, section, style_fig
from models import d5_two_methods as TM

ARCH_FILE = C.DATA_DIR / "lga_archetype_summary.csv"

BUNDLE = {
    "Deprived northern rural": "Reaching-Every-Community microplanning + BHCPF-funded outreach + community health workers; co-deliver with nutrition and antenatal contacts; defaulter tracing",
    "Low maternal care, high conflict exposure": "Security-integrated microplanning + negotiated access + mobile outreach teams; integrate with antenatal care; defaulter tracing",
    "Geographically isolated": "Outreach and mobile sessions (boat-based where waterways apply) + community health worker networks + multi-antigen bundling per visit",
    "Near-average": "Core fixed-site and outreach services + ward-level enumeration + Periodic Intensification of Routine Immunization",
    "Relatively advantaged": "Targeted social mobilisation + fixed-site plus outreach hybrid + private-sector last-mile reach",
}
EVIDENCE = {
    "Deprived northern rural": "Measured constraints: highest undernutrition, lowest maternal education and facility delivery, high DTP1-3 dropout. Evidence-based response: fixed-plus-outreach integrated with nutrition and antenatal contacts, with defaulter tracing. Frameworks: WHO Reaching Every District/Community; Gavi zero-dose IRMMA; WHO/UNICEF Big Catch-Up (2023).",
    "Low maternal care, high conflict exposure": "Measured constraints: highest conflict events and fatalities, weak antenatal and delivery care, high dropout. Evidence-based response: negotiated access, security-sensitive scheduling, mobile outreach and maternal-care integration. Frameworks: WHO/UNICEF immunization in humanitarian and conflict settings; polio negotiated-access experience. Population mobility is not measured and must be confirmed locally.",
    "Geographically isolated": "Measured constraints: very long travel time to care, low improved water. Evidence-based response: transport-adapted outreach with multi-antigen bundling per visit. Frameworks: WHO Reaching Every District/Community integrated outreach.",
    "Near-average": "Measured constraints: near the national average on every indicator; fair care-seeking. Evidence-based response: core services, enumeration and periodic intensification; use each LGA's flagged barriers to add components. Frameworks: WHO PIRI; RED/REC microplanning.",
    "Relatively advantaged": "Measured constraints: highest wealth, education and facility delivery; moderate conflict events. Evidence-based response: demand generation, fixed-plus-outreach hybrid and private-sector reach. Frameworks: WHO Behavioural and Social Drivers of vaccination (BeSD).",
}

TIER_ORDER = ["Critical", "High", "Moderate", "Low"]
TIER_COLOR = {"Critical": "#B2182B", "High": "#EF8A62", "Moderate": "#F0C24B", "Low": "#9ECAE1"}
TIER_DEF = {"Critical": "worst quarter on the equity index (most deprived)", "High": "second quarter",
            "Moderate": "third quarter", "Low": "least-deprived quarter"}
ARCH_COLOR = {"Deprived northern rural": "#8B1A1A", "Low maternal care, high conflict exposure": "#D6604D",
              "Geographically isolated": "#2C7FB8", "Near-average": "#F2C14E", "Relatively advantaged": "#1C7A3D"}
ARCH_DEF = {
    "Deprived northern rural": "highest undernutrition, lowest maternal education and facility delivery, high dropout",
    "Low maternal care, high conflict exposure": "highest conflict events and fatalities, weak antenatal and delivery care",
    "Geographically isolated": "very long travel time to care, low improved water",
    "Near-average": "near the national average on every indicator",
    "Relatively advantaged": "highest wealth, education and facility delivery; moderate conflict events",
}


METHOD_MD = """
**Evidence-to-barrier method - how the intervention bundles were developed (not guesswork or AI-only):**

1. Each archetype's covariate signature identifies its **binding constraint(s)** - a demand-side,
physical-access, security, or health-system barrier.
2. For each binding constraint we selected intervention packages that the **normative and peer-reviewed
immunization-implementation literature** shows to be effective against that specific barrier.
3. The approach follows the **Gavi zero-dose IRMMA framework** (Identify, Reach, Monitor and Measure,
Advocate) and the **WHO Reaching Every District / Reaching Every Community** strategy, and is deliberately
**multi-programme** (immunization, nutrition, reproductive/maternal and newborn health, family planning,
water and sanitation, service delivery), because the determinants are multi-programme.

**Key frameworks cited**
- WHO Reaching Every District / Reaching Every Community (RED/REC)
- Gavi zero-dose IRMMA framework (Identify, Reach, Monitor and Measure, Advocate)
- WHO, UNICEF and Gavi 'The Big Catch-Up' (2023)
- WHO Periodic Intensification of Routine Immunization (PIRI)
- WHO Behavioural and Social Drivers of vaccination (BeSD)
- Polio-programme negotiated access / 'Days of Tranquility'; permanent transit-point vaccination

These are framework references; align to the current NPHCDA National Immunization Policy and routine-
immunization microplanning guidelines before implementation.
"""


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


def _archetype_map(tm: dict):
    import spatial
    lg = tm["lga"]
    g = spatial.load_gdf("lga").merge(
        lg[["state_key", "geo_lga_key", "archetype_type"]].rename(columns={"geo_lga_key": "lga_key"}),
        on=["state_key", "lga_key"], how="left")
    g["archetype_type"] = g["archetype_type"].fillna("Not classified")
    return viz.choropleth(g, "archetype_type", categorical=True, color_map=ARCH_COLOR,
                          title="Contextual profiles (Ward clustering of 15 LGA indicators, k=5)",
                          legend_title="Profile", height=640)


def _archetype_burden_fig(tm: dict) -> go.Figure:
    a = tm["archetype"].sort_values("archetype")
    names = [clean(t) for t in a["archetype_type"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=a["cohort_share"], name="Share of children 12-23 months", orientation="h",
                marker_color="#CBD5E1")
    fig.add_bar(y=names, x=a["m1_share"], name=TM.M1_SHORT, orientation="h", marker_color=C.NAVY)
    fig.add_bar(y=names, x=a["m2_share"], name=TM.M2_SHORT, orientation="h", marker_color=C.NPHCDA_GREEN)
    fig.update_layout(barmode="group", title="Share of zero-dose children by contextual profile, 2026",
                      xaxis_title="Percent of the national total", yaxis=dict(autorange="reversed", automargin=True),
                      legend=dict(orientation="h", y=-0.2))
    return style_fig(fig, height=460)


def render():
    domain_banner("_banner_d5.jpg", "LGA Priority and Archetypes",
                  "Every local government ranked by its estimated zero-dose children, tagged with its "
                  "contextual profile, dominant barrier, candidate components and equity-deprivation tier.")
    tm = TM.load()
    method = st.radio("Rank LGAs by", [TM.M1, TM.M2], horizontal=True, key="lp_method",
                      help="The burden rank and zero-dose figures follow the chosen method. Archetype and "
                           "equity tier do not depend on the method.")
    df = TM.priority_table(tm, method, BUNDLE, EVIDENCE)
    a = tm["archetype"].set_index("archetype")
    p = TM._p(method)
    top2 = float(a.loc[[1, 2], f"{p}_share"].sum())
    n_top = int((df["Priority flag"] == "TOP PRIORITY").sum())
    ranked = df["Burden rank"].notna()
    kpi_row([
        {"label": "LGAs ranked", "value": f"{int(ranked.sum())} of 774", "sub": clean(method.split(":")[0]),
         "color": C.NAVY},
        {"label": "Zero-dose children, 2026", "value": f"{df['Zero-dose children'].sum() / 1e6:.2f}M",
         "sub": "sum over ranked LGAs", "color": C.ACCENT},
        {"label": "Profiles 1 and 2", "value": f"{top2:.0f}%",
         "sub": clean(f"of zero-dose children; {a.loc[[1, 2], 'cohort_share'].sum():.0f}% of the cohort"),
         "color": C.GOLD},
        {"label": "TOP PRIORITY LGAs", "value": str(n_top),
         "sub": "top 155 by burden and Critical/High deprivation", "color": C.STEEL,
         "help": "LGAs in the 155 highest-burden LGAs (the top 20 percent of 774) whose equity tier is "
                 "Critical or High."},
    ])

    with st.expander("What the classifications mean (equity tier and contextual profile)", expanded=False):
        st.markdown("**Equity-deprivation tier** (equal-weight index of remoteness, low women's education, "
                    "poverty and low relative wealth; quartiles across the 774 local governments):")
        _chips(TIER_COLOR, TIER_DEF)
        st.markdown("**Contextual profile** (archetype; five groups from Ward agglomerative clustering of 15 "
                    "local-government indicators; labels describe only what the indicators measure):")
        _chips(ARCH_COLOR, ARCH_DEF)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(_archetype_map(tm), use_container_width=True)
    with c2:
        st.plotly_chart(_archetype_burden_fig(tm), use_container_width=True)
        st.caption(clean(f"Profiles 1 and 2 (deprived northern rural; low maternal care, high conflict exposure) hold "
                         f"{a.loc[[1, 2], 'm1_share'].sum():.0f}% of zero-dose children under Method 1 and "
                         f"{a.loc[[1, 2], 'm2_share'].sum():.0f}% under Method 2 (SAE), against "
                         f"{a.loc[[1, 2], 'cohort_share'].sum():.0f}% of the 12-23-month cohort."))

    EV.render_profile_evidence(tm)

    section("LGA priority list")
    c1, c2, c3, c4 = st.columns(4)
    states = c1.multiselect("State", sorted(df["State"].dropna().unique()))
    archs = c2.multiselect("Contextual profile", [x for x in ARCH_COLOR if x in set(df["Archetype"])])
    bars = (c3.multiselect("Dominant barrier", sorted(df["Dominant barrier"].dropna().unique()))
            if "Dominant barrier" in df else [])
    tiers = c4.multiselect("Equity tier", [t for t in TIER_ORDER if t in set(df["Equity tier"])])
    only_priority = st.checkbox("Show only TOP PRIORITY local governments (high burden and high "
                                "deprivation)", value=False)
    f = df.copy()
    if states:
        f = f[f["State"].isin(states)]
    if archs:
        f = f[f["Archetype"].isin(archs)]
    if bars:
        f = f[f["Dominant barrier"].isin(bars)]
    if tiers:
        f = f[f["Equity tier"].isin(tiers)]
    if only_priority:
        f = f[f["Priority flag"] == "TOP PRIORITY"]
    tot = int(pd.to_numeric(f["Zero-dose children"], errors="coerce").fillna(0).sum())
    nat = int(pd.to_numeric(df["Zero-dose children"], errors="coerce").fillna(0).sum())
    st.write(clean(f"**{len(f)} local governments** shown, holding **{tot:,} zero-dose children** "
                   f"({tot / nat * 100:.1f}% of the national {nat:,})."))
    tier_bg = {"Critical": "#EDA9A2", "High": "#F6C9AE", "Moderate": "#FBE7B0", "Low": "#BBD5EA"}
    sty = (f.style.hide(axis="index")
           .map(lambda v: f"background-color:{tier_bg.get(v, '')}", subset=["Equity tier"])
           .format({"Equity index": "{:.1f}", "Zero-dose rate (%)": "{:.1f}", "Zero-dose children": "{:,.0f}"}, na_rep=""))
    st.dataframe(sty, use_container_width=True, height=460)
    st.download_button("Download this list (CSV)", f.to_csv(index=False).encode("utf-8"),
                       f"NPHCDA_LGA_priority_list_{p}.csv", "text/csv")
    ai.ai_block(f"lp_{p}", f"LGA Priority and Archetypes ({method})",
                "The highest-burden LGAs with archetype, equity tier and priority flag, and the share of "
                "zero-dose children per archetype. Name the archetypes and states that dominate the TOP "
                "PRIORITY list and the intervention bundle that fits them.",
                {"top_30": df.head(30)[["Burden rank", "State", "LGA", "Zero-dose children", "Zero-dose rate (%)",
                                        "Equity tier", "Archetype", "Priority flag"] + [c for c in ["Dominant barrier",
                                        "Candidate components (LGA barriers)", "P(top 155), Method 2 (%)"] if c in df]]
                           .to_dict(orient="records"),
                 "archetype_shares": tm["archetype"][["archetype_type", "lgas", "cohort_share", "m1_share",
                                                      "m2_share"]].round(1).to_dict(orient="records")})

    if ARCH_FILE.exists():
        with st.expander("The five contextual profiles - determinants, intervention levers and evidence"):
            arch = _load(str(ARCH_FILE)).copy()
            am = tm["archetype"].set_index("archetype")
            arch["Mean zero-dose rate (%)"] = arch["Cluster"].map(am[f"{p}_mean_rate"]).round(1)
            arch["Zero-dose children (sum)"] = arch["Cluster"].map(am[f"{p}_children"]).round(0).astype(int)
            st.dataframe(arch, use_container_width=True)
            st.caption(clean("Zero-dose columns follow the method selected above. Interventions are matched "
                             "to each archetype's binding constraint using an evidence-to-barrier method "
                             "(WHO Reaching Every District/Community; Gavi zero-dose IRMMA; WHO/UNICEF Big "
                             "Catch-Up; PIRI; BeSD)."))
    with st.expander("How the intervention bundles were developed - method and evidence"):
        st.markdown(METHOD_MD)
    st.caption(clean("Zero-dose figures are model estimates for 2026. Candidate components follow each LGA's own "
                     "flagged barriers (national top quartile of a domain score) and need local validation. "
                     "Profile and equity tier use modelled "
                     "covariate surfaces (2014-2021). Under Method 1, Guzamala (Borno) is not estimated (no "
                     "Penta1 data for 2021-2024)."))
