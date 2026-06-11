"""
NPHCDA Zero-Dose Predictive Modelling Platform (Domains 1, 2, 5).

A real-time, no-code tool: upload the raw immunization data and the app runs the Domain 1, 2
and 5 models live and renders the same tables and visualizations as the analytical notebooks.
Entry point for Streamlit / Hugging Face Spaces.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")  # no system C++ compiler needed (nutpie sampler)

import pandas as pd
import streamlit as st

import config as C
from theme import (inject_theme, hero, section, kpi_row, clean, sidebar_brand,
                   domain_banner, highlight_severity)

st.set_page_config(page_title="NPHCDA Zero-Dose Modelling Platform",
                   page_icon="💉", layout="wide", initial_sidebar_state="expanded")
inject_theme()

import data_io as io  # noqa: E402  (after page config)
import data_quality as dq  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402
import reports  # noqa: E402
import ai  # noqa: E402
import impsci  # noqa: E402
import domain1, domain2, domain5  # noqa: E402

STATUS_ICON = {"ok": "🟢", "partial": "🟡", "invalid": "🔴", "missing": "⚪"}


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def sidebar() -> str:
    with st.sidebar:
        sidebar_brand()
        st.caption(clean("CIDRE and Quantium Insights LLC, in technical support of NPHCDA. "
                         "Funders and reviewers: GAVI and UNICEF."))
        nav_options = ["Home", "Data and Quality", "Coverage Forecasting", "Dropout & Completion",
                       "Zero-Dose & Hotspots", "Implementation Science", "Reports & Briefs",
                       "Program Q&A (RAG)"]
        # Allow other pages to request navigation (e.g. the Home "Upload your own data" button).
        if st.session_state.get("_goto") in nav_options:
            st.session_state["navradio"] = st.session_state.pop("_goto")
        page = st.radio("Navigate", nav_options, label_visibility="collapsed", key="navradio")
        st.divider()
        src = st.session_state.get("data_source")
        if src:
            st.success(clean(f"Active data: {src}"))
        else:
            st.warning("No data loaded. Go to Home or Data and Quality.")

        st.divider()
        st.markdown("#### AI interpretation")
        key = st.text_input("OpenAI API key", type="password", key="openai_key",
                            placeholder="sk-...", help="Stored only for this session, never saved.")
        model = st.selectbox("Model", llm.MODELS, index=llm.MODELS.index(llm.DEFAULT_MODEL),
                             key="openai_model")
        st.session_state["llm"] = {"key": key.strip() if key else "", "model": model}
        st.toggle("Auto-interpret figures on load", value=True, key="ai_auto",
                  help="Generate an interpretation under each figure and table automatically. "
                       "Turn off to interpret only on demand and save API calls.")
        st.caption("AI enabled" if (key and key.strip()) else
                   clean("Optional. Add a key for AI interpretation, chat, and document Q&A."))
        st.caption(clean("Responses are grounded in the on-screen outputs only; off-topic or "
                         "harmful requests are declined."))
        return page


# --------------------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------------------
def page_home():
    hero("Zero-Dose Predictive Modelling Platform",
         "Upload the routine immunization data and run the coverage, dropout and zero-dose models in "
         "real time.",
         ["Coverage Forecasting", "Dropout & Completion", "Zero-Dose & Hotspots",
          "Bayesian + Prophet + spatial"])
    st.caption(clean("These are the GAVI-priority workstreams of NPHCDA's wider eight-domain "
                     "zero-dose analytical framework, delivered as a working platform."))

    c1, c2 = st.columns([2, 1])
    with c1:
        section("What this tool does")
        st.markdown(clean(
            "- **Coverage Forecasting.** National Prophet forecasts of BCG, Penta1, Penta3 and "
            "Measles1 as a percent of the 2024 baseline, against the 80 percent target, plus an LGA "
            "at-risk screen and state/LGA microplanning projections.\n"
            "- **Dropout & Completion.** Prophet forecasts of Penta1-to-Penta3, Penta1-to-Measles1 "
            "and Measles1-to-Measles2 dropout, with LASSO-selected drivers and state-by-year heatmaps.\n"
            "- **Zero-Dose & Hotspots.** A Bayesian hierarchical Beta regression of state zero-dose "
            "rates with credible intervals, population-weighted LGA burden, Pareto prioritization and "
            "Getis-Ord Gi* hotspot maps.\n"
            "- **Implementation Science.** Exploratory analysis of the state zero-dose dataset - "
            "correlation matrix, distributions, scatter with Pearson r and p, violin by zone with a "
            "Kruskal-Wallis test, Sankey, mosaic and a Bland-Altman agreement plot.\n"
            "- **Data and Quality.** Schema validation, completeness, reporting rates, and anomaly "
            "detection on the uploaded data.\n"
            "- **Reports and Briefs.** One-click generation of a premium factsheet and an editable "
            "policy brief from the live results.\n"
            "- **Program Q&A (RAG).** Ask questions of an uploaded programme report (PDF or Word) with "
            "page-cited, document-grounded answers.\n\n"
            "Every output carries a grounded AI interpretation and a chat. Models run live on the "
            "uploaded data; heavy steps are scoped so each click returns quickly."))
    with c2:
        section("Start here")
        st.write(clean("Option A - explore now with the canonical project inputs:"))
        if st.button("Use bundled sample data", type="primary", use_container_width=True):
            io.set_sample_data()
            st.rerun()
        st.write(clean("Option B - run the models on your own files:"))
        if st.button("Upload your own data", use_container_width=True):
            st.session_state["_goto"] = "Data and Quality"
            st.rerun()
        st.caption(clean("Uploads (DHIS2 export, NDHS file, LGA population, under-five cohort, model "
                         "dataset) live on the Data and Quality page in the sidebar."))
        if st.session_state.get("data_source"):
            st.success(clean(f"Data loaded: {st.session_state['data_source']}. "
                             "Open a domain from the sidebar."))

    section("Workstreams and research questions")
    st.dataframe(pd.DataFrame([
        {"Workstream": "Coverage Forecasting",
         "Research question": "Which antigens fall below the 80% target in 6-12 months?"},
        {"Workstream": "Dropout & Completion",
         "Research question": "What are predicted dropout rates and what drives incomplete vaccination?"},
        {"Workstream": "Zero-Dose & Hotspots",
         "Research question": "Where are zero-dose children most concentrated and why?"},
    ]), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Data and Quality
# --------------------------------------------------------------------------------------
UPLOAD_ORDER = ["dhis2", "ndhs_long", "model_dataset", "under5", "lga_population"]


def page_data():
    domain_banner("_banner_dq.jpg", "Data and Quality",
                  "Upload the raw sources or use the bundled sample. Each file is validated against "
                  "its expected schema before modelling.")

    if st.button("Use bundled sample data", type="primary"):
        io.set_sample_data()
        st.rerun()

    section("Upload data sources")
    data = dict(st.session_state.get("data") or {})
    for key in UPLOAD_ORDER:
        schema = C.SCHEMAS[key]
        up = st.file_uploader(clean(schema["label"]), type=["csv", "xlsx"], key=f"up_{key}")
        if up is not None:
            try:
                data[key] = io.read_uploaded(up)
            except Exception as exc:
                st.error(clean(f"Could not read {schema['label']}: {exc}"))
    if any(v is not None for v in data.values()):
        st.session_state["data"] = data
        if not st.session_state.get("data_source", "").startswith("Bundled"):
            st.session_state["data_source"] = "Uploaded data"

    data = io.get_active_data()
    if not data:
        st.info("No data loaded yet. Click the button above or upload files.")
        return

    section("Schema validation")
    rows = []
    for key in UPLOAD_ORDER:
        v = io.validate(key, data.get(key))
        miss = clean(", ".join(v["missing"])) if v["missing"] else "-"
        rows.append({"Dataset": clean(C.SCHEMAS[key]["label"]),
                     "Status": f"{STATUS_ICON[v['status']]} {v['status']}",
                     "Rows": v["n_rows"], "Missing required": miss})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if data.get("dhis2") is None:
        return

    d = io.prep_dhis2(data["dhis2"])
    q = dq.dhis2_quality(d)
    nat = io.national_monthly(d)
    span = q["span"]
    span_txt = f"{span[0]:%b %Y} to {span[1]:%b %Y}" if span[0] is not None else "-"

    qtab, atab = st.tabs(["Data quality", "Anomaly detection"])

    with qtab:
        section("DHIS2 data quality")
        kpi_row([
            {"label": "States", "value": str(q["n_states"]), "color": C.NAVY},
            {"label": "LGAs in file", "value": str(q["n_lgas"]), "color": C.STEEL},
            {"label": "Reporting LGAs", "value": f"{q['reporting_lgas']} / {q['total_lgas']}",
             "sub": "non-zero Penta1, latest year", "color": C.GOLD},
            {"label": "Count completeness", "value": f"{q['completeness']:.0f}%", "color": C.STEEL},
            {"label": "Reporting rate",
             "value": f"{q['reporting_rate']:.0f}%" if pd.notna(q["reporting_rate"]) else "-",
             "sub": clean(span_txt), "color": C.STEEL},
        ])
        ai.ai_block("dq_overview", "DHIS2 data quality overview",
                    "Headline data-quality metrics: number of states and LGAs in the file, LGAs "
                    "reporting non-zero Penta1 in the latest year (of 774), count completeness, the "
                    "state-month reporting rate, and the reporting period.", q)

        st.plotly_chart(dq.missingness_by_state(d), use_container_width=True)
        miss = (d.groupby("state")["penta_1_count"]
                .apply(lambda s: int((s.fillna(0) <= 0).sum())).sort_values(ascending=False))
        miss_ctx = {"reporting_rate_pct": q["reporting_rate"], "period": span_txt,
                    "states_with_most_missing_months": miss.head(8).to_dict()}
        ai.ai_block("dq_missing", "DHIS2 reporting completeness by state and month",
                    "Which states have the most missing or zero Penta1 state-months (reporting gaps), "
                    "and the overall state-month reporting rate.", miss_ctx)

    with atab:
        section("Anomaly detection (national antigen series)",
                "Months where a national dose count deviates sharply from its own history (z-score).")
        st.caption(clean("How to read this: Severity 'Moderate' = |z| between 2 and 3 (amber); 'High' "
                         "= |z| at or above 3 (red). Direction 'Spike' = unusually high, 'Drop' = "
                         "unusually low. Investigate High anomalies first - they often signal "
                         "data-entry errors, campaigns, or reporting changes."))
        out = dq.outliers(nat)
        if out.empty:
            st.success("No anomalies flagged - every month is within 2 standard deviations of its series mean.")
        else:
            kpi_row([
                {"label": "Anomalies flagged", "value": str(len(out)), "color": C.ACCENT},
                {"label": "High severity", "value": str(int((out["Severity"] == "High").sum())),
                 "sub": "|z| >= 3", "color": C.ACCENT},
                {"label": "Spikes", "value": str(int((out["Direction"] == "Spike").sum())), "color": C.GOLD},
                {"label": "Drops", "value": str(int((out["Direction"] == "Drop").sum())), "color": C.STEEL},
            ])
            st.dataframe(highlight_severity(out), use_container_width=True, hide_index=True)
            ai.ai_block("dq_outliers", "National-series anomaly flags",
                        "Months where a national antigen dose count is more than 2 standard deviations "
                        "from its series mean, with z-score, percent deviation, direction and severity. "
                        "Flag which anomalies most warrant a data check.", out)


# --------------------------------------------------------------------------------------
# Program document Q&A (RAG)
# --------------------------------------------------------------------------------------
def page_rag():
    domain_banner("_banner_rag.jpg", "Program Document Q&A (RAG)",
                  "Upload a programme report (PDF or Word). The app indexes it and answers your "
                  "questions in real time, grounded in the document, with page citations and a "
                  "retrieval-confidence score. Answers come only from the uploaded document.")
    cfg = st.session_state.get("llm") or {}
    if not cfg.get("key"):
        st.warning("Add your OpenAI API key in the sidebar to enable document Q&A.")
        return

    up = st.file_uploader("Upload a report (PDF or .docx)", type=["pdf", "docx"], key="rag_up")
    if up is not None:
        current = (st.session_state.get("rag_doc") or {}).get("name")
        if current != up.name or st.button("Re-index document"):
            with st.spinner("Reading and indexing the document..."):
                try:
                    pages = rag.extract(up.name, up.getvalue())
                    chunks = rag.chunk(pages)
                except Exception as exc:
                    st.error(clean(f"Could not read the document: {exc}"))
                    return
                if not chunks:
                    st.error("No readable text found in the document.")
                    return
                matrix, err = rag.build_index(cfg["key"], chunks)
                if err:
                    st.error(clean(err))
                    return
                st.session_state["rag_doc"] = {"name": up.name, "chunks": chunks, "matrix": matrix,
                                               "pages": pages[-1][0] if pages else 0}
                st.session_state["rag_chat"] = []
                st.session_state["rag_last_snips"] = None

    doc = st.session_state.get("rag_doc")
    if not doc:
        st.info("Upload a document to begin.")
        return
    st.success(clean(f"Indexed: {doc['name']} - {len(doc['chunks'])} passages across about "
                     f"{doc['pages']} pages."))

    st.session_state.setdefault("rag_chat", [])
    for m in st.session_state["rag_chat"]:
        with st.chat_message(m["role"]):
            st.markdown(clean(m["content"]))

    with st.form("rag_form", clear_on_submit=True):
        q = st.text_input("Ask the document",
                          placeholder="e.g. What is the national zero-dose reduction target?")
        c1, c2 = st.columns([1, 1])
        send = c1.form_submit_button("Ask")
        clear = c2.form_submit_button("Clear chat")
    if clear:
        st.session_state["rag_chat"] = []
        st.session_state["rag_last_snips"] = None
        st.rerun()
    if send and q and q.strip():
        st.session_state["rag_chat"].append({"role": "user", "content": q.strip()})
        with st.spinner("Searching the document..."):
            res = rag.answer(cfg["key"], cfg.get("model", llm.DEFAULT_MODEL), q.strip(),
                             doc["matrix"], doc["chunks"])
        if res.get("error"):
            ans = clean(res["error"])
            st.session_state["rag_last_snips"] = None
        else:
            cites = ", ".join(f"p. {p}" for p in res["citations"])
            ans = (clean(res["text"]) + f"\n\n---\nSources: {cites}  |  Retrieval confidence: "
                   f"{res['confidence']:.0f}%")
            st.session_state["rag_last_snips"] = res["snippets"]
        st.session_state["rag_chat"].append({"role": "assistant", "content": ans})
        st.rerun()

    snips = st.session_state.get("rag_last_snips")
    if snips:
        with st.expander("Retrieved passages used for the last answer"):
            for s in snips:
                st.markdown(clean(f"**p. {s['page']}** (similarity {s['score']:.2f}) - {s['text'][:400]}..."))


# --------------------------------------------------------------------------------------
# Reports & Briefs
# --------------------------------------------------------------------------------------
def page_reports():
    import streamlit.components.v1 as components
    domain_banner("_banner_reports.jpg", "Reports and Briefs",
                  "Generate a premium factsheet or an editable policy brief from the live model "
                  "outputs. Every figure traces to the on-screen results.")
    data = io.get_active_data()
    if not data:
        st.warning("Load data first on Home or the Data and Quality page.")
        return
    cfg = st.session_state.get("llm") or {}

    c1, c2 = st.columns([2, 1])
    doc_type = c1.radio("Document", ["Factsheet (1 page, HTML / print to PDF)",
                                     "Policy brief (editable Word .docx)"], horizontal=False)
    if cfg.get("key"):
        use_ai = c2.toggle("AI-drafted narrative", value=True,
                           help="Draft the narrative with the model, grounded in the findings. "
                                "Off uses a templated narrative built from the numbers.")
    else:
        use_ai = False
        c2.caption(clean("Add an OpenAI key in the sidebar for an AI-drafted narrative; a templated "
                         "narrative is used otherwise."))

    if st.button("Generate document", type="primary"):
        with st.spinner("Assembling findings from the live models (first run may fit the zero-dose model)..."):
            findings = reports.build_findings(data)
        kind = "factsheet" if doc_type.startswith("Factsheet") else "policy"
        with st.spinner("Drafting the narrative..."):
            nar = ""
            if use_ai and cfg.get("key"):
                nar = llm.compose_brief(cfg["key"], cfg.get("model", llm.DEFAULT_MODEL), kind, findings)
            if (not nar) or nar.strip() == llm.REFUSAL or len(nar.strip()) < 60:
                nar = reports.template_narrative(findings, kind)
        st.session_state["rep"] = {"findings": findings, "narrative": nar, "kind": kind}

    rep = st.session_state.get("rep")
    if not rep:
        return
    if rep["kind"] == "factsheet":
        html = reports.factsheet_html(rep["findings"], rep["narrative"])
        section("Factsheet preview")
        components.html(html, height=920, scrolling=True)
        st.download_button("Download factsheet (HTML - open and print to PDF)",
                           html.encode("utf-8"), "NPHCDA_zero_dose_factsheet.html", "text/html")
    else:
        section("Policy brief preview")
        st.markdown(clean(rep["narrative"]))
        docx_bytes = reports.policy_docx(rep["findings"], rep["narrative"])
        st.download_button(
            "Download policy brief (Word .docx)", docx_bytes,
            "NPHCDA_zero_dose_policy_brief.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# --------------------------------------------------------------------------------------
# Implementation Science - EDA
# --------------------------------------------------------------------------------------
def page_impsci():
    domain_banner("_banner_impsci.jpg", "Implementation Science - Exploratory Data Analysis",
                  "Descriptive statistics and univariate / bivariate analysis of the state zero-dose "
                  "dataset to surface the drivers that should shape implementation.")
    data = io.get_active_data()
    if not data or data.get("model_dataset") is None:
        st.warning("This domain needs the zero-dose model dataset (equity covariates). Load the "
                   "bundled sample data or upload it on the Data and Quality page.")
        return
    df = impsci.prep(data["model_dataset"])
    cols = impsci.numeric_cols(df)
    o = impsci.OUTCOME
    kpi_row([
        {"label": "States", "value": str(len(df)), "color": C.NAVY},
        {"label": "Variables analysed", "value": str(len(cols)), "color": C.STEEL},
        {"label": "Mean zero-dose 2024", "value": f"{df[o].mean():.0f}%", "color": C.ACCENT},
        {"label": "Range across states", "value": f"{df[o].min():.0f}-{df[o].max():.0f}%",
         "color": C.GOLD},
    ])
    tabs = st.tabs(["Descriptive stats", "Univariate", "Bivariate", "Validation (Bland-Altman)",
                    "Hypothesis tests"])

    with tabs[0]:
        section("Descriptive statistics", "Summary of the zero-dose outcome and its candidate drivers.")
        desc = impsci.describe_table(df)
        st.dataframe(desc, use_container_width=True, hide_index=True)
        ai.ai_block("is_desc", "Implementation Science - descriptive statistics",
                    "Summary statistics (mean, spread, min, max) of the state zero-dose rate and its "
                    "candidate equity/socioeconomic drivers.", desc.to_dict(orient="records"))

    with tabs[1]:
        section("Correlation matrix", "Pearson correlations among zero-dose and its drivers.")
        cfig, csum = impsci.corr_fig(df)
        st.plotly_chart(cfig, use_container_width=True)
        ai.ai_block("is_corr", "Implementation Science - correlation and multicollinearity",
                    "Two things: (1) the drivers most strongly correlated with the 2024 zero-dose rate "
                    "(signed Pearson r; positive means higher driver goes with higher zero-dose); and "
                    "(2) MULTICOLLINEARITY - the 'multicollinear_pairs' list gives predictor pairs with "
                    "|r| >= 0.8. For each such pair, flag it and recommend dropping the suggested "
                    "variable (keep the one more associated with the outcome) before fitting a "
                    "regression, to avoid unstable, hard-to-interpret coefficients.", csum)
        section("Distribution of a variable")
        var = st.selectbox("Variable", cols, format_func=impsci.pretty, key="is_hist_var")
        hfig, hstats = impsci.hist_box_fig(df, var)
        st.plotly_chart(hfig, use_container_width=True)
        ai.ai_block("is_hist", f"Distribution of {impsci.pretty(var)}",
                    "Histogram and boxplot statistics (mean, median, spread, skew) for the selected "
                    "variable across the 37 states.", {"variable": impsci.pretty(var), **hstats})

    with tabs[2]:
        section("Driver vs zero-dose (scatter, Pearson r and p)")
        x = st.selectbox("Driver (x-axis)", [c for c in cols if c != o],
                         format_func=impsci.pretty, key="is_scatter_x")
        sfig, ssum = impsci.scatter_fig(df, x)
        st.plotly_chart(sfig, use_container_width=True)
        ai.ai_block("is_scatter", f"{impsci.pretty(x)} vs zero-dose rate",
                    "Association between the selected driver and the 2024 zero-dose rate across states, "
                    "with the Pearson correlation and its p-value.", {"driver": impsci.pretty(x), **ssum})

        section("Zero-dose by zone (violin, Kruskal-Wallis)")
        vfig, vsum = impsci.violin_fig(df)
        st.plotly_chart(vfig, use_container_width=True)
        ai.ai_block("is_violin", "Zero-dose rate by geopolitical zone",
                    "Distribution of state zero-dose rates within each zone and whether zones differ "
                    "significantly (Kruskal-Wallis p).", vsum)

        section("Burden band composition by zone",
                "How many states in each zone fall in the Low (<20%), Moderate (20-40%) and High "
                "(>40%) zero-dose bands.")
        kfig, ksum = impsci.band_bar_fig(df)
        st.plotly_chart(kfig, use_container_width=True)
        ai.ai_block("is_band", "States per zero-dose burden band, by zone",
                    "The distribution of states across Low, Moderate and High zero-dose bands within "
                    "each zone. Name the zones dominated by the High band and what that implies for "
                    "where to concentrate effort.", ksum)

    with tabs[3]:
        section("Bland-Altman agreement",
                "Agreement between two coverage measures across states: bias (mean difference) and "
                "95 percent limits of agreement (LoA).")
        with st.expander("How to read a Bland-Altman plot (and when to use it)", expanded=False):
            st.markdown(clean(
                "- **Purpose.** Bland-Altman assesses whether two ways of measuring the same thing "
                "*agree*, which a correlation cannot tell you (two methods can correlate strongly yet "
                "disagree by a constant offset).\n"
                "- **Axes.** Each point is one state: x = the average of the two measures, y = their "
                "difference (Measure B minus Measure A).\n"
                "- **Bias** (solid navy line) = the mean difference. Near 0 means no systematic over- or "
                "under-statement; a large bias means one measure is consistently higher.\n"
                "- **Limits of agreement** (dashed red, bias +/- 1.96 SD) = the band within which about "
                "95 percent of differences fall. Narrow limits = good agreement; wide limits = the two "
                "measures can diverge a lot for an individual state.\n"
                "- **Look for:** points outside the limits (outlier states), and any funnel shape "
                "(disagreement growing with the level).\n"
                "- **Use cases here.** Compare survey rounds (e.g. DTP1 2018 vs 2024) to see if the "
                "shift is uniform, or compare a survey measure against an administrative/model estimate "
                "to check whether the two data sources can be used interchangeably for planning."))
        cmp_cols = [c for c in ["dtp1_2008", "dtp1_2013", "dtp1_2018", "dtp1_2024"] if c in df.columns]
        c1, c2 = st.columns(2)
        a = c1.selectbox("Measure A", cmp_cols, index=max(len(cmp_cols) - 2, 0), key="is_ba_a")
        b = c2.selectbox("Measure B", cmp_cols, index=len(cmp_cols) - 1, key="is_ba_b")
        bfig, bsum = impsci.bland_altman_fig(df, a, b)
        st.plotly_chart(bfig, use_container_width=True)
        ai.ai_block("is_ba", f"Bland-Altman: {impsci.pretty(a)} vs {impsci.pretty(b)}",
                    "Agreement between the two selected DTP1 coverage measures: the bias (mean "
                    "difference) and the 95 percent limits of agreement.",
                    {"measure_a": impsci.pretty(a), "measure_b": impsci.pretty(b), **bsum})

    with tabs[4]:
        section("Hypothesis testing",
                "Pose a question, pick the variables and a test, then run it. Tests use a 0.05 "
                "significance level.")
        num = impsci.all_numeric(df)
        test = st.selectbox("Statistical test",
                            ["Independent t-test", "Paired t-test", "One-way ANOVA", "Chi-square test"],
                            key="is_test")
        params: dict = {}
        if test == "Independent t-test":
            c1, c2 = st.columns(2)
            params["outcome"] = c1.selectbox("Outcome (numeric)", num,
                                             index=num.index(impsci.OUTCOME) if impsci.OUTCOME in num else 0,
                                             format_func=impsci.pretty, key="is_t_out")
            params["group_var"] = c2.selectbox("Grouping variable (split at its median)",
                                               [c for c in num if c != params["outcome"]],
                                               format_func=impsci.pretty, key="is_t_grp")
        elif test == "Paired t-test":
            pair = [c for c in ["dtp1_2008", "dtp1_2013", "dtp1_2018", "dtp1_2024"] if c in num] or num
            c1, c2 = st.columns(2)
            params["col_a"] = c1.selectbox("Measure A", pair, index=max(len(pair) - 2, 0),
                                           format_func=impsci.pretty, key="is_p_a")
            params["col_b"] = c2.selectbox("Measure B", pair, index=len(pair) - 1,
                                           format_func=impsci.pretty, key="is_p_b")
        elif test == "One-way ANOVA":
            params["outcome"] = st.selectbox("Outcome (numeric)", num,
                                             index=num.index(impsci.OUTCOME) if impsci.OUTCOME in num else 0,
                                             format_func=impsci.pretty, key="is_a_out")
            params["group_var"] = "zone_name"
            st.caption("Grouping: geopolitical zone.")
        else:  # Chi-square
            opts = ["zone_name"] + num
            fmt = lambda c: "Zone" if c == "zone_name" else impsci.pretty(c)
            c1, c2 = st.columns(2)
            params["var1"] = c1.selectbox("Variable 1", opts, index=0, format_func=fmt, key="is_c_v1")
            params["var2"] = c2.selectbox("Variable 2", opts,
                                          index=1 if len(opts) > 1 else 0, format_func=fmt, key="is_c_v2")
            st.caption("Numeric variables are split into Low/Mid/High thirds for the contingency table.")

        if st.button("Run test", type="primary", key="is_run_test"):
            st.session_state["is_test_res"] = impsci.hypothesis_test(df, test, **params)
        res = st.session_state.get("is_test_res")
        if res and not res.get("error"):
            st.markdown(f"**Question:** {clean(res['question'])}")
            m1, m2, m3 = st.columns(3)
            m1.metric(res["statistic_name"], res["statistic"])
            m2.metric("p-value", res["p_value"])
            m3.metric("Significant (a=0.05)", "Yes" if res["significant"] else "No")
            (st.success if res["significant"] else st.info)(clean(res["conclusion"]))
            det = res["detail"]
            if isinstance(det, dict) and det and isinstance(next(iter(det.values())), dict):
                st.dataframe(pd.DataFrame(det), use_container_width=True)
            else:
                st.dataframe(pd.DataFrame([det]) if isinstance(det, dict) else pd.DataFrame(det),
                             use_container_width=True, hide_index=True)
            ai.ai_block("is_hyptest", f"Hypothesis test - {res['test']}",
                        "A statistical test result (statistic, p-value, group/contingency detail) "
                        "answering the stated question. Explain in plain language what the result "
                        "means for programming, whether the difference/association is significant, and "
                        "one action it supports. Note the small sample (37 states) as a caveat.", res)

    st.divider()
    ai.chat_panel("impsci", "Implementation Science - state zero-dose EDA",
                  "State-level zero-dose rate and equity/socioeconomic drivers; correlations, "
                  "distributions and zone differences.",
                  {"top_correlations": impsci.corr_fig(df)[1],
                   "mean_zero_dose_2024": round(float(df[o].mean()), 1)},
                  suggestions=["Which driver correlates most with zero-dose?",
                               "Do zones differ significantly?"])


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
def main():
    page = sidebar()
    data = io.get_active_data()
    if page == "Home":
        page_home()
    elif page == "Data and Quality":
        page_data()
    elif page.startswith("Coverage"):
        domain1.render(data)
    elif page.startswith("Dropout"):
        domain2.render(data)
    elif page.startswith("Zero-Dose"):
        domain5.render(data)
    elif page.startswith("Implementation"):
        page_impsci()
    elif page.startswith("Reports"):
        page_reports()
    elif page.startswith("Program Q&A"):
        page_rag()


if __name__ == "__main__":
    main()
