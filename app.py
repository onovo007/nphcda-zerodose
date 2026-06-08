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
from theme import inject_theme, hero, section, kpi_row, clean

st.set_page_config(page_title="NPHCDA Zero-Dose Modelling Platform",
                   page_icon="💉", layout="wide", initial_sidebar_state="expanded")
inject_theme()

import data_io as io  # noqa: E402  (after page config)
import data_quality as dq  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402
import domain1, domain2, domain5  # noqa: E402

STATUS_ICON = {"ok": "🟢", "partial": "🟡", "invalid": "🔴", "missing": "⚪"}


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def sidebar() -> str:
    with st.sidebar:
        st.markdown("### NPHCDA Zero-Dose Platform")
        st.caption(clean("CIDRE and Quantium Insights LLC, in technical support of NPHCDA. "
                         "Funders and reviewers: GAVI and UNICEF."))
        page = st.radio("Navigate", ["Home", "Data and Quality",
                                     "Domain 1 - Coverage", "Domain 2 - Dropout",
                                     "Domain 5 - Zero-dose", "Program Q&A (RAG)"],
                        label_visibility="collapsed")
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
         "Upload the routine immunization data and run the Domain 1, 2 and 5 models in real time.",
         ["Domain 1 - Coverage forecasting", "Domain 2 - Dropout dynamics",
          "Domain 5 - Zero-dose hotspots", "Bayesian + Prophet + spatial"])

    c1, c2 = st.columns([2, 1])
    with c1:
        section("What this tool does")
        st.markdown(clean(
            "- **Domain 1.** National Prophet forecasts of BCG, Penta1, Penta3 and Measles1 as a "
            "percent of the 2024 baseline, against the 80 percent target, plus an LGA at-risk screen.\n"
            "- **Domain 2.** Prophet forecasts of Penta1-to-Penta3, Penta1-to-Measles1 and "
            "Measles1-to-Measles2 dropout, with LASSO-selected drivers and state-by-year heatmaps.\n"
            "- **Domain 5.** A Bayesian hierarchical Beta regression of state zero-dose rates with "
            "credible intervals, population-weighted LGA burden, Pareto prioritization and Getis-Ord "
            "Gi* hotspot maps.\n\n"
            "The models run live on the uploaded data. Heavy steps are scoped so each click returns "
            "quickly; full per-LGA and full-posterior runs are available behind explicit controls."))
    with c2:
        section("Start here")
        st.write(clean("Run the demo instantly with the canonical project inputs, or upload your own "
                       "files on the Data and Quality page."))
        if st.button("Use bundled sample data", type="primary", use_container_width=True):
            io.set_sample_data()
            st.rerun()
        if st.session_state.get("data_source"):
            st.success(clean("Data loaded. Open a domain from the sidebar."))

    section("Modelling domains and research questions")
    st.dataframe(pd.DataFrame([
        {"Domain": "1 Coverage forecasting",
         "Research question": "Which antigens fall below the 80% target in 6-12 months?"},
        {"Domain": "2 Dropout dynamics",
         "Research question": "What are predicted dropout rates and what drives incomplete vaccination?"},
        {"Domain": "5 Zero-dose and hotspots",
         "Research question": "Where are zero-dose children most concentrated and why?"},
    ]), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Data and Quality
# --------------------------------------------------------------------------------------
UPLOAD_ORDER = ["dhis2", "ndhs_long", "model_dataset", "under5", "lga_population"]


def page_data():
    st.markdown("## Data and Quality")
    st.caption(clean("Upload the raw sources or use the bundled sample. Each file is validated "
                     "against its expected schema before modelling."))

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
    section("DHIS2 data quality")
    span = q["span"]
    span_txt = f"{span[0]:%b %Y} to {span[1]:%b %Y}" if span[0] is not None else "-"
    kpi_row([
        {"label": "States", "value": str(q["n_states"]), "color": C.NAVY},
        {"label": "LGAs in file", "value": str(q["n_lgas"]), "color": C.STEEL},
        {"label": "Reporting LGAs", "value": f"{q['reporting_lgas']} / {q['total_lgas']}",
         "sub": "non-zero Penta1, latest year", "color": C.GOLD},
        {"label": "Count completeness", "value": f"{q['completeness']:.0f}%", "color": C.STEEL},
        {"label": "Reporting rate", "value": f"{q['reporting_rate']:.0f}%" if pd.notna(q["reporting_rate"]) else "-",
         "sub": clean(span_txt), "color": C.STEEL},
    ])
    st.plotly_chart(dq.missingness_by_state(d), use_container_width=True)

    nat = io.national_monthly(d)
    out = dq.outliers(nat)
    section("Outlier flags (national series, |z| > 2)")
    if out.empty:
        st.success("No national-series outliers flagged.")
    else:
        st.dataframe(out, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------------------
# Program document Q&A (RAG)
# --------------------------------------------------------------------------------------
def page_rag():
    st.markdown("## Program document Q&A (RAG)")
    st.caption(clean("Upload a programme report (PDF or Word). The app indexes it and answers your "
                     "questions in real time, grounded in the document, with page citations and a "
                     "retrieval-confidence score. Answers come only from the uploaded document."))
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
# Router
# --------------------------------------------------------------------------------------
def main():
    page = sidebar()
    data = io.get_active_data()
    if page == "Home":
        page_home()
    elif page == "Data and Quality":
        page_data()
    elif page.startswith("Domain 1"):
        domain1.render(data)
    elif page.startswith("Domain 2"):
        domain2.render(data)
    elif page.startswith("Domain 5"):
        domain5.render(data)
    elif page.startswith("Program Q&A"):
        page_rag()


if __name__ == "__main__":
    main()
