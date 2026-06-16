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
import auth  # noqa: E402
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
                       "Zero-Dose & Hotspots", "Exploratory Data Analysis", "Ask the Analyst",
                       "Reports & Briefs", "Program Q&A (RAG)", "User Guide (SOP)",
                       "Methods & Validation"]
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

        user = auth.current_user()
        if user and user.get("email"):
            st.divider()
            st.caption(clean(f"Signed in: {user['name']} ({user['email']})"))
            if st.button("Sign out", use_container_width=True):
                auth.logout()
                st.rerun()
        auth.admin_panel()
        return page


# --------------------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------------------
def page_home():
    hero("Zero-Dose Predictive Modelling Platform",
         "One analytical framework that integrates immunization data from many sources into real-time "
         "predictive intelligence and programme oversight - pinpointing where zero-dose children are, "
         "why they are missed, and where to act first, so NPHCDA can optimize Nigeria's immunization "
         "programme and ensure that no child is left behind or exposed to vaccine-preventable disease.",
         ["Coverage Forecasting", "Dropout & Completion", "Zero-Dose & Hotspots",
          "Bayesian + Prophet + spatial"])
    st.caption(clean("A decision-support platform for the NPHCDA Digital Innovation Hub. Upload your "
                     "data or use the bundled sample to run every model live."))

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
            "- **Exploratory Data Analysis.** Exploratory analysis of the state zero-dose dataset - a "
            "correlation matrix with multicollinearity flags, distributions, scatter with Pearson r "
            "and p, violin by zone with a Kruskal-Wallis test, burden-band bars, a Bland-Altman "
            "agreement plot, and a build-your-own hypothesis test (t-test, paired t-test, ANOVA, "
            "chi-square).\n"
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
        st.write(clean("Option A - explore now with the bundled project sample data:"))
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
UPLOAD_ORDER = ["dhis2", "ndhs_long", "model_dataset", "under5", "lga_population",
                "ndhs_antigens", "live_births"]


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

    with st.expander("Data dictionary and provenance (source, vintage)"):
        st.dataframe(pd.DataFrame(
            [{"Dataset": clean(a), "Source": clean(b), "Vintage": clean(c), "Holds": clean(d_)}
             for a, b, c, d_ in C.PROVENANCE], ), use_container_width=True, hide_index=True)
        st.caption(clean("Figures are model estimates from these sources; cite the vintage when "
                         "sharing. Denominators: under-five from NPC 2022 projections (City "
                         "Population); LGA population NPC 2022."))

    if data.get("dhis2") is None:
        return

    d = io.prep_dhis2(data["dhis2"])
    bad = float(d["ds"].isna().mean()) if "ds" in d else 1.0
    if bad > 0.02:
        st.warning(clean(f"{bad*100:.0f}% of DHIS2 'period' values could not be parsed into a date - "
                         "expected formats like Jan-21, 21-Jan or 2021-01. Forecasts and reporting "
                         "rates use only the parsed rows; please check the period column."))
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

        section("DHIS2 reporting completeness")
        st.markdown(clean(
            "<div style='display:flex;gap:18px;align-items:center;font-size:.85rem'>"
            "<span><span style='display:inline-block;width:14px;height:14px;background:#2E6E8E;"
            "border-radius:3px;vertical-align:middle'></span> Reporting (non-zero Penta1)</span>"
            "<span><span style='display:inline-block;width:14px;height:14px;background:#E8746C;"
            "border-radius:3px;vertical-align:middle'></span> Missing or zero (no report)</span></div>"),
            unsafe_allow_html=True)
        with st.expander("What 'completeness' and 'reporting rate' mean (and how they are calculated)"):
            st.markdown(clean(
                "- **Count completeness** = of all expected antigen-count cells in the file (the count "
                "columns across every LGA-month row), the percentage that are present (not blank). "
                "Formula: 1 - (blank cells / total cells). It measures whether values were entered.\n"
                "- **Reporting rate (state-month)** = of the full grid of states x months, the percentage "
                "of state-months with a reported, non-zero Penta1. Formula: filled state-months / "
                "(states x months). It measures whether the unit reported at all that month.\n"
                "- **On the heatmap:** blue = reported non-zero Penta1 that month; red = missing or zero.\n"
                "- **Why state vs LGA differ:** a state can always report at state level (all blue) yet "
                "have many missing LGA-months underneath - the LGA drill-down reveals that local gap."))

        months = dq.month_list(d)
        start = end = None
        if len(months) > 1:
            labels = [m.strftime("%b %Y") for m in months]
            sl, el = st.select_slider("Month range to display", options=labels,
                                      value=(labels[0], labels[-1]), key="dq_month_range")
            start, end = months[labels.index(sl)], months[labels.index(el)]

        rtab_s, rtab_l = st.tabs(["By state", "By LGA (drill-down)"])
        with rtab_s:
            st.plotly_chart(dq.missingness_by_state(d, start, end), use_container_width=True)
            pres_s = dq.present_matrix(d, "state", start, end)
            miss_sm = (pres_s == 0).sum(axis=1).sort_values(ascending=False)
            miss_sm = miss_sm[miss_sm > 0]
            sm_ctx = {"reporting_rate_pct": q["reporting_rate"], "period": span_txt,
                      "states_with_missing_state_months": miss_sm.head(10).to_dict(),
                      "note": "counts are missing STATE-months (a state reported nothing that month)"}
            ai.ai_block("dq_missing", "DHIS2 state-month reporting (filled vs missing)",
                        "The state-month reporting rate and which states have missing state-months "
                        "(months where the whole state reported no or zero Penta1). Counts are out of "
                        "the number of months in view.", sm_ctx)
        with rtab_l:
            state_sel = st.selectbox("State to drill into", sorted(d["state"].dropna().unique()),
                                     key="dq_lga_state")
            lfig, pres_l = dq.missingness_by_lga(d, state_sel, start, end)
            st.plotly_chart(lfig, use_container_width=True)
            miss_lga = (pres_l == 0).sum(axis=1).sort_values(ascending=False)
            miss_lga = miss_lga[miss_lga > 0]
            n_cols = pres_l.shape[1]
            lga_ctx = {"state": state_sel, "months_in_view": int(n_cols),
                       "n_lgas": int(pres_l.shape[0]),
                       "total_missing_lga_months": int((pres_l == 0).sum().sum()),
                       "lgas_with_most_missing_months": miss_lga.head(12).to_dict()}
            ai.ai_block("dq_lga_missing", f"DHIS2 LGA reporting gaps - {state_sel}",
                        "Within the selected state, which LGAs have the most missing or zero Penta1 "
                        "months (out of the months in view), and the total missing LGA-months. Name the "
                        "worst LGAs and give one action to close the reporting gap.", lga_ctx)

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
        d1, d2 = st.columns(2)
        docx_bytes = reports.policy_docx(rep["findings"], rep["narrative"])
        d1.download_button(
            "Download policy brief (Word .docx)", docx_bytes,
            "NPHCDA_zero_dose_policy_brief.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        pptx_bytes = reports.policy_pptx(rep["findings"], rep["narrative"])
        d2.download_button(
            "Download policy deck (PowerPoint .pptx)", pptx_bytes,
            "NPHCDA_zero_dose_policy_deck.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation")


# --------------------------------------------------------------------------------------
# Implementation Science - EDA
# --------------------------------------------------------------------------------------
def page_impsci():
    domain_banner("_banner_impsci.jpg", "Exploratory Data Analysis",
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
        {"label": "States", "value": str(len(df)), "sub": "in the dataset", "color": C.NAVY},
        {"label": "Variables analysed", "value": str(len(cols)), "sub": "outcome + numeric drivers",
         "color": C.STEEL},
        {"label": "Avg state zero-dose 2024", "value": f"{df[o].mean():.1f}%",
         "sub": "mean across the 37 states", "color": C.ACCENT},
        {"label": "Zero-dose range (states)", "value": f"{df[o].min():.0f}% - {df[o].max():.0f}%",
         "sub": "lowest to highest state", "color": C.GOLD},
    ])

    # Reference metrics teams can quote directly, split into two clearly-titled tabs.
    survey = (io.survey_national_coverage(data["ndhs_antigens"], data.get("under5"))
              if data.get("ndhs_antigens") is not None else {})
    pop_cards = []
    if data.get("under5") is not None:
        cohort = float(io.prep_under5(data["under5"])["cohort_12_23m"].sum())
        pop_cards.append({"label": "Birth cohort (demographic)", "value": f"{cohort/1e6:.2f}M",
                          "sub": "under-five 2024 / 5 (eligible infants)", "color": C.NAVY})
    if data.get("live_births") is not None:
        lb = io.national_live_births(data["live_births"], 2024)
        if lb:
            pop_cards.append({"label": "DHIS2 live births 2024", "value": f"{lb/1e6:.2f}M",
                              "sub": "facility-reported (under-counts true births)", "color": C.STEEL})
    if data.get("ndhs_antigens") is not None:
        fv = io.national_survey_value(data["ndhs_antigens"], "Fully vaccinated (8 basic antigens)",
                                      data.get("under5"))
        nv = io.national_survey_value(data["ndhs_antigens"], "Received no vaccinations",
                                      data.get("under5"))
        if fv is not None:
            pop_cards.append({"label": "Fully vaccinated", "value": f"{fv:.1f}%",
                              "sub": "NDHS 2024, 8 basic antigens", "color": C.NPHCDA_GREEN})
        if nv is not None:
            pop_cards.append({"label": "No vaccinations (survey)", "value": f"{nv:.1f}%",
                              "sub": "NDHS 2024 zero-dose proxy", "color": C.ACCENT})

    ref_tabs = st.tabs(["NDHS 2024 survey coverage (national)", "Population and denominators"])
    with ref_tabs[0]:
        if survey:
            st.caption(clean("Population-weighted national survey coverage per antigen (NDHS 2024) - "
                             "reference values programme teams can quote directly."))
            kpi_row([{"label": f"{a} coverage", "value": f"{survey[a]:.1f}%",
                      "sub": "NDHS 2024, national", "color": C.ANTIGEN_PAL.get(a, C.STEEL)}
                     for a in ["BCG", "Penta1", "Penta3", "Measles1"] if a in survey])
        else:
            st.info(clean("Load the NDHS antigens 2024 file to show national survey coverage."))
    with ref_tabs[1]:
        if pop_cards:
            st.caption(clean("Eligible-infant denominators (birth cohort vs DHIS2 live births) and "
                             "headline survey coverage."))
            kpi_row(pop_cards)
        else:
            st.info(clean("Load the under-five and/or NDHS antigens files to show population metrics."))

    st.divider()
    tabs = st.tabs(["Descriptive stats", "Univariate", "Bivariate", "Validation (Bland-Altman)",
                    "Hypothesis tests", "Multiple regression"])

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

        st.divider()
        section("Drivers of zero-dose (parsimonious Beta regression, top LASSO-selected)",
                "LASSO selects drivers on the logit scale; the top 4 are fit with a Beta (or "
                "fractional-logit) model reporting the coefficient, its direction and a 95% CI.")
        st.caption(clean("Uncertainty quantification, not a significance verdict. A positive logit "
                         "coefficient means the driver is associated with higher zero-dose. These are "
                         "cross-sectional, ecological associations across 37 states (not causal); read "
                         "the direction and the CI width given the small sample."))
        if st.button("Run zero-dose driver model", key="is_beta_btn"):
            tbl, method, n = impsci.beta_drivers(df)
            st.session_state["is_beta"] = {"tbl": tbl, "method": method, "n": n}
        bres = st.session_state.get("is_beta")
        if bres and bres["tbl"] is not None:
            st.caption(clean(f"Model: {bres['method']} - {bres['n']} states, top 4 LASSO-selected drivers."))
            st.dataframe(bres["tbl"], use_container_width=True, hide_index=True)
            ai.ai_block("is_beta_block", "Zero-dose drivers - parsimonious Beta regression",
                        "A parsimonious Beta regression of the state zero-dose rate on the top "
                        "LASSO-selected drivers: coefficient (logit scale), direction and 95% CI. Frame "
                        "as directional associations with uncertainty (not causal); name the strongest "
                        "driver and its direction, note any CI that crosses zero, and give one "
                        "programming implication. These are ecological state-level associations.",
                        bres["tbl"].to_dict(orient="records"))

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

    with tabs[5]:
        section("Multiple linear regression (build your own)",
                "Fit a multivariable model of an outcome on the predictors you choose. Reports "
                "coefficients, HC3 robust standard errors, p-values, 95% CIs and model fit "
                "(R-squared, adjusted R-squared, overall F-test).")
        allnum = impsci.all_numeric(df)
        c1, c2 = st.columns([1, 2])
        out_var = c1.selectbox("Outcome variable", allnum,
                               index=allnum.index(impsci.OUTCOME) if impsci.OUTCOME in allnum else 0,
                               format_func=impsci.pretty, key="is_mreg_y")
        default_x = [p for p in impsci.DRIVERS if p in allnum and p != out_var][:3]
        preds = c2.multiselect("Predictor variables", [c for c in allnum if c != out_var],
                               default=default_x, format_func=impsci.pretty, key="is_mreg_x")
        if st.button("Fit regression", type="primary", key="is_mreg_btn"):
            if not preds:
                st.warning("Select at least one predictor.")
            else:
                r = impsci.multiple_regression(df, out_var, preds)
                st.session_state["is_mreg"] = r if r else "fail"
        res = st.session_state.get("is_mreg")
        if res == "fail":
            st.warning(clean("Not enough complete rows for the chosen predictors - pick fewer "
                             "predictors (need at least predictors + 2 states)."))
        elif isinstance(res, dict):
            s = res["stats"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Observations", s["n"])
            m2.metric("Predictors", s["predictors"])
            m3.metric("R-squared", s["R2"])
            m4.metric("Adj. R-squared", s["adj_R2"])
            st.caption(clean(f"Overall model F-test p = {s['F_p']}. HC3 robust standard errors. "
                             "Cross-sectional, ecological associations (state-level) - not causal."))
            st.dataframe(res["table"], use_container_width=True, hide_index=True)
            ai.ai_block("is_mreg_ai", f"Multiple regression - {impsci.pretty(out_var)}",
                        "A multivariable linear regression of the outcome on the selected predictors "
                        "(robust SEs). Explain which predictors are significantly associated (p<0.05), "
                        "the direction and magnitude of each, how much variation the model explains "
                        "(R-squared), and the programme implication. Caution: ecological state-level "
                        "associations (small n), not causal effects.",
                        {"outcome": impsci.pretty(out_var), "model_fit": s,
                         "coefficients": res["table"].to_dict(orient="records")})

    st.divider()
    ai.chat_panel("impsci", "Implementation Science - state zero-dose EDA",
                  "State-level zero-dose rate and equity/socioeconomic drivers; correlations, "
                  "distributions and zone differences.",
                  {"top_correlations": impsci.corr_fig(df)[1],
                   "mean_zero_dose_2024": round(float(df[o].mean()), 1)},
                  suggestions=["Which driver correlates most with zero-dose?",
                               "Do zones differ significantly?"])


# --------------------------------------------------------------------------------------
# User Guide / SOP
# --------------------------------------------------------------------------------------
def page_sop():
    domain_banner("_banner_sop.jpg", "User Guide - Standard Operating Procedure",
                  "A premium SOP for the NPHCDA Digital Innovation Hub: how any new user accesses and "
                  "uses the platform to model the zero-dose dataset, end to end.")

    # Workflow at a glance (illustrated step strip).
    steps = ["Sign in", "Load data", "Check quality", "Run models", "Explore & test",
             "Ask the Analyst", "Generate reports"]
    chips = ""
    for i, s in enumerate(steps):
        chips += (f"<div class='sopchip'><span class='sopn'>{i+1}</span>{s}</div>")
        if i < len(steps) - 1:
            chips += "<div class='soparr'>&rarr;</div>"
    st.markdown(
        "<style>"
        ".sopflow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:6px 0 4px}"
        f".sopchip{{display:flex;align-items:center;gap:8px;background:linear-gradient(180deg,#fff,#f3f8f4);"
        f"border:1px solid rgba(28,122,61,.25);border-radius:999px;padding:7px 14px;font-weight:600;"
        f"color:{C.NAVY};font-size:.86rem}}"
        f".sopn{{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;"
        f"border-radius:50%;background:{C.NPHCDA_GREEN};color:#fff;font-size:.78rem}}"
        f".soparr{{color:{C.GOLD};font-weight:800}}"
        "</style>"
        f"<div class='sopflow'>{chips}</div>", unsafe_allow_html=True)
    st.caption(clean("Estimated time end to end: about 5-10 minutes on the bundled sample data; longer "
                     "if you run the full Bayesian posterior or per-state forecasts."))
    st.download_button("Download this SOP (Word .docx)", reports.sop_docx(),
                       "NPHCDA_Zero_Dose_Platform_SOP.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    section("Before you start - what you need")
    st.markdown(clean(
        "- **Fastest path:** nothing - use the bundled project sample data to explore immediately.\n"
        "- **To model your own data**, prepare these five CSVs (same columns as the sample):"))
    st.dataframe(pd.DataFrame([
        {"File": "DHIS2 export", "Holds": "Monthly antigen doses by state and LGA", "Used by": "Coverage, Dropout, Zero-Dose"},
        {"File": "NDHS zero-dose (long)", "Holds": "Survey zero-dose rate by state and year", "Used by": "Zero-Dose"},
        {"File": "Under-five population", "Holds": "Under-5 cohort by state", "Used by": "Zero-Dose burden"},
        {"File": "LGA population", "Holds": "Population by LGA (NPC 2022)", "Used by": "LGA burden, hotspots"},
        {"File": "Zero-dose model dataset", "Holds": "State equity / socioeconomic covariates", "Used by": "Implementation Science, drivers"},
        {"File": "NDHS antigen coverage 2024", "Holds": "State survey coverage per antigen", "Used by": "Coverage (admin-vs-survey lines), reference metrics"},
        {"File": "DHIS2 live births (monthly)", "Holds": "Facility-reported live births by LGA", "Used by": "Coverage (optional eligible denominator)"},
    ]), use_container_width=True, hide_index=True)

    section("Step-by-step")
    st.markdown("#### 1. Sign in")
    st.markdown(clean("Open the platform link. Enter your name and email (and the access code if your "
                      "administrator set one). Your sign-in is recorded for usage tracking."))
    st.markdown("#### 2. Load the data")
    st.markdown(clean("On **Home**, click **Use bundled project sample data** to explore now, or click "
                      "**Upload your own data** to go to **Data and Quality** and add your five CSVs. A "
                      "green confirmation shows the data is loaded."))
    st.info(clean("Tip: start with the sample data on your first run so you can see every output before "
                  "using your own files."))
    st.markdown("#### 3. Check data quality and anomalies")
    st.markdown(clean("Open **Data and Quality**. Review completeness, reporting rates and the missing-"
                      "value heatmap (with a **By-LGA drill-down** and a month-range filter), then the "
                      "**Anomaly detection** tab for spikes or drops (colour-coded by severity). Fix "
                      "obvious data issues before modelling."))
    st.markdown("#### 4. Run the models")
    st.markdown(clean(
        "- **Coverage Forecasting** - which antigens fall below the 80% target; choose the forecast "
        "end-year (to 2032) and the 3/6/12-month scorecard horizon; switch to **WHO administrative "
        "coverage** to pick the eligible-infant denominator (under-five proxy or DHIS2 live births) "
        "and show the **NDHS survey reference lines**; review the LGA at-risk screen.\n"
        "- **Dropout & Completion** - dropout forecasts, LASSO drivers and the state-by-year heatmap.\n"
        "- **Zero-Dose & Hotspots** - the Bayesian state model, LGA burden, Pareto priorities and the "
        "Getis-Ord Gi* hotspot maps (run automatically)."))
    st.markdown("#### 5. Explore and test (Implementation Science)")
    st.markdown(clean("Use the correlation matrix (with multicollinearity flags), distributions, scatter "
                      "and zone violins, and the **Hypothesis tests** tab to run a t-test, ANOVA or "
                      "chi-square on variables you choose."))
    st.markdown("#### 6. Ask the Analyst")
    st.markdown(clean("Add your OpenAI API key in the sidebar, then open **Ask the Analyst** for grounded, "
                      "cross-domain answers and recommendations across all results."))
    st.markdown("#### 7. Generate reports")
    st.markdown(clean("On **Reports & Briefs**, generate the premium factsheet, the editable Word policy "
                      "brief, and the PowerPoint policy deck - all built from the live results."))
    st.markdown("#### 8. (Optional) Program Q&A")
    st.markdown(clean("On **Program Q&A (RAG)**, upload a programme report (PDF or Word) and ask questions "
                      "with page-cited, document-grounded answers."))

    section("How to read the outputs")
    st.dataframe(pd.DataFrame([
        {"Signal": "Coverage / at-risk", "Meaning": "Red = below the 80% target; green = on target"},
        {"Signal": "Anomaly severity", "Meaning": "High (|z| >= 3) vs Moderate; spike or drop flagged"},
        {"Signal": "Priority tier (LGA)", "Meaning": "Tier 1 Critical (red) to Tier 4 (blue), by burden"},
        {"Signal": "Pareto severity", "Meaning": "Critical/High/Moderate/Lower within state; band A/B/C of burden"},
        {"Signal": "Gi* hotspot", "Meaning": "Hot Spot (red) = significant high-burden cluster; p<0.01 most confident"},
        {"Signal": "Survey line (Coverage)", "Meaning": "Dotted purple = NDHS survey coverage; a large gap vs admin flags denominator/reporting issues"},
        {"Signal": "Credible / prediction interval", "Meaning": "The plausible range; wider = more uncertainty"},
    ]), use_container_width=True, hide_index=True)

    with st.expander("Tips and troubleshooting"):
        st.markdown(clean(
            "- **No data loaded** message: go to Home and load the sample or upload your files.\n"
            "- **AI says add a key:** paste your OpenAI key in the sidebar (kept for the session only).\n"
            "- **A heavy step is slow:** the full Bayesian posterior and per-state forecasts run on "
            "demand; use the defaults for a fast live run.\n"
            "- **Access denied at login:** your email may not be on the access list - contact the DIH "
            "administrator.\n"
            "- **Save a report as PDF:** open the factsheet and use your browser's Print to PDF."))

    with st.expander("Data governance and good practice"):
        st.markdown(clean(
            "- All figures are model estimates - review before use in official decisions.\n"
            "- Uploaded data stays in the session and is not written to disk; the OpenAI key is never "
            "stored.\n"
            "- Cite the data vintage (DHIS2 month, NDHS round, NPC 2022 denominator) when sharing.\n"
            "- For zero-dose definitions and targets, align with IA2030 and the national RI guidelines."))
    st.caption(clean("Print this page (browser Print to PDF) to share the SOP offline. For capacity "
                     "building, walk a new user through steps 1-7 once on the sample data."))


# --------------------------------------------------------------------------------------
# Methods & Validation (reviewer-facing)
# --------------------------------------------------------------------------------------
def page_methods():
    domain_banner("_banner_methods.jpg", "Methods and Validation",
                  "A reviewer-facing summary of the models, parameters, validation and honest caveats "
                  "- for GAVI, UNICEF and NPHCDA technical review.")
    st.download_button("Download Methods & Validation (Word .docx)", reports.methods_docx(),
                       "NPHCDA_Methods_and_Validation.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    for heading, kind, payload in reports.methods_sections():
        section(heading)
        if kind == "para":
            for para in payload:
                st.markdown(clean(para))
        elif kind == "bullets":
            st.markdown("\n".join(f"- {clean(b)}" for b in payload))
        elif kind == "table":
            headers, trows = payload
            st.dataframe(pd.DataFrame(trows, columns=headers), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Ask the Analyst - cross-domain agent
# --------------------------------------------------------------------------------------
def page_agent():
    domain_banner("_banner_agent.jpg", "Ask the Analyst",
                  "A grounded assistant across all results - coverage forecasting, dropout, zero-dose "
                  "modelling and hotspots, at national, state and LGA level.")
    data = io.get_active_data()
    if not data:
        st.warning("Load data first on Home or the Data and Quality page.")
        return
    # ZARA - the named, branded assistant (NPHCDA Zero-dose Analytics & Risk Assistant).
    avatar_svg = (
        "<svg width='52' height='52' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='1.6' "
        "stroke-linecap='round' stroke-linejoin='round'><rect x='4' y='8' width='16' height='11' rx='3'/>"
        "<circle cx='9' cy='13' r='1.4' fill='white' stroke='none'/><circle cx='15' cy='13' r='1.4' "
        "fill='white' stroke='none'/><line x1='12' y1='4' x2='12' y2='8'/><circle cx='12' cy='3' r='1.2'/>"
        "<line x1='8' y1='16.5' x2='16' y2='16.5'/></svg>")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:14px;background:linear-gradient(120deg,"
        f"{C.NAVY},{C.NPHCDA_GREEN});border-radius:14px;padding:14px 18px;margin-bottom:10px'>"
        f"<div style='background:rgba(255,255,255,.14);border-radius:50%;padding:6px;display:flex'>{avatar_svg}</div>"
        f"<div><div style='color:#fff;font-size:1.25rem;font-weight:700'>ZARA</div>"
        f"<div style='color:rgba(255,255,255,.9);font-size:.82rem'>Zero-dose Analytics &amp; Risk "
        f"Assistant - your NPHCDA biostatistics, epidemiology and immunization-programme advisor</div>"
        f"</div></div>", unsafe_allow_html=True)
    cfg = st.session_state.get("llm") or {}
    if not cfg.get("key"):
        st.info(clean("Add your OpenAI API key in the sidebar to chat with ZARA."))
    with st.spinner("ZARA is assembling results from all workstreams (first run may fit the zero-dose model)..."):
        ctx = reports.build_findings(data)
    ai.chat_panel("analyst", "All workstream results (coverage, dropout, zero-dose, hotspots)",
                  "Combined live outputs: national antigen coverage forecasts and at-risk antigens; "
                  "dropout forecasts and drivers; state zero-dose forecasts, tiers and burden; "
                  "population-weighted LGA burden, Pareto concentration and the top LGAs.", ctx,
                  suggestions=["Which states and LGAs should we prioritize first, and why?",
                               "What is the single biggest risk across all workstreams?",
                               "Draft three recommendations for the next quarter."],
                  system=llm.ANALYST_SYSTEM, assistant_avatar="🤖",
                  heading="##### Ask ZARA")


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
def main():
    if not auth.require_login():
        return
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
    elif page.startswith("Exploratory"):
        page_impsci()
    elif page.startswith("Ask the Analyst"):
        page_agent()
    elif page.startswith("User Guide"):
        page_sop()
    elif page.startswith("Methods"):
        page_methods()
    elif page.startswith("Reports"):
        page_reports()
    elif page.startswith("Program Q&A"):
        page_rag()


if __name__ == "__main__":
    main()
