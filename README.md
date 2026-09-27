---
title: NPHCDA Zero-Dose Platform
emoji: 💉
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# NPHCDA Zero-Dose Predictive Modelling Platform

A real-time, no-code decision-support platform for the NPHCDA (National Primary Health Care Development Agency) Digital Innovation Hub. It turns routine immunization data into predictive intelligence - showing where zero-dose children are, why they are missed, and where to act first - and it re-runs every model live on new data with no coding required.

A **zero-dose** child is one who never received the first pentavalent dose (DTP1 / Penta1) - a child the health system never reached.

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA; funders and reviewers GAVI and UNICEF.

---

## 1. What is in this handover

| Asset | Where | Purpose |
| --- | --- | --- |
| **Web application** | https://amobionovo-nphcda-zerodose.hf.space/ (also deployable on Render) | The live, no-code tool - run models, view results, export |
| **Source code (this repo)** | github.com/onovo007/nphcda-zerodose | The full app; `app.py` is the entry point |
| **Analytical notebooks** | `notebooks/` in this repo (and Google Drive) | Reproducible Google Colab notebooks for Domains 1-7 |
| **Final report** | `handover/Final_NPHCDA_Consolidated_Report_UPDATED_08_09_2026.docx` | The consolidated technical report |
| **Final slide deck** | `handover/NPHCDA_ZeroDose_RI_Team_Presentation_9.27.2026.pptx` | The RI-team presentation |
| **Datasets (large / harmonized)** | Google Drive (link below), organized by domain | GitHub does not host large data; the Drive folder is the data of record |

**Google Drive datasets:** https://drive.google.com/drive/folders/107pWT0m4_A9Sk4aVji-t8LV3m7gfoLzx

---

## 2. Using the web application

### Getting in and loading data
1. Open the platform link and **sign in** with your name and an authorized email.
2. On the **Home** page, click **"Use bundled sample data"** (a single click). The app ships with the canonical sample inputs in `data/sample/`, so it runs end to end with no upload.
3. Give a page a few seconds to warm up on first use; results then load and later pages are faster.

### The pages (what each answers)
| Page | Answers |
| --- | --- |
| **Home** | Overview and the research questions; load data here |
| **Data and Quality** | Schema validation, completeness, reporting rates; upload your own files here |
| **Coverage Forecasting (Domain 1)** | Which antigens are projected to fall below 80% of their 2024 level (at-risk-of-decline early-warning). Includes an **Additional antigens** tab (OPV3, IPV1, PCV3, Yellow Fever, Men A as early-warning; IPV2 and Rotavirus 1-3 as uptake) |
| **Dropout and Completion (Domain 2)** | Antigen-pair dropout forecasts and the LASSO-selected drivers |
| **Zero-Dose and Hotspots (Domain 5)** | Modelled state and LGA zero-dose rates and burden, priority tiers, Getis-Ord Gi* hotspot maps |
| **LGA Priority and Archetypes** | Every local government ranked worst-to-best by burden, tagged with its archetype, equity tier and matched intervention |
| **Triangulation and Cross-Checks** | External and convergent validation (NmDHS 2025-26; IHME DTP1 surface) |
| **Exploratory Data Analysis** | Correlations, distributions, hypothesis tests |
| **Ask the Analyst / Program Q&A (RAG) / ZARA** | AI interpretation and document-grounded Q&A (needs an API key - see note) |
| **Reports and Briefs** | One-click factsheet and policy brief from the live results |
| **User Guide (SOP) / Methods and Validation** | In-app documentation |

### Reading the results (key framing)
- **At-risk-of-decline early-warning:** a projection below **80% of the antigen's 2024 level** (2024 is the last complete year). This is a decline signal, **not** true coverage of the child population.
- **Priority ranking:** rank by **zero-dose burden (worst-to-best)**; the **equity tier** and **archetype** are shown alongside each local government. High burden plus Tier-1 deprivation is the top priority.
- **Every estimate is a model estimate** with an uncertainty range; local-government figures are calibrated, not directly observed.

### The AI interpretation / ZARA feature
The grounded AI interpretation and the ZARA assistant require an OpenAI API key entered in the sidebar (per session). Without a key the whole platform still works - only the AI-written interpretation text is hidden. To enable AI for all reviewers without each entering a key, add an `OPENAI_API_KEY` server-side secret (and set a spending cap).

---

## 3. Running the app locally

Prerequisites: Python 3.12 and Git.
```
git clone https://github.com/onovo007/nphcda-zerodose.git
cd nphcda-zerodose
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```
Open http://localhost:8501, sign in (any name/email locally), and click "Use bundled sample data".

**Do not change these pinned versions in `requirements.txt`:** `numpy==1.26.4` and `pyarrow==16.1.0`. The pyarrow pin fixes a native crash (a numpy-2-era pyarrow segfaults on every table render against numpy 1.26). With numpy pinned below 2, keep every compiled numpy-dependent wheel on a numpy-1.x build.

---

## 4. The notebooks (Google Colab)

The `notebooks/` folder holds the reproducible analysis, one per domain:

| Notebook | Domain |
| --- | --- |
| `D1_D2__domains_1_2_final.ipynb` | Domains 1 and 2 - coverage forecasting and dropout |
| `D1_Additional_Antigens_Forecast.ipynb` | Domain 1 extension - OPV, IPV, PCV, Yellow Fever, Men A, Rotavirus |
| `D3__equity_zero_dose_stratification.ipynb` | Domain 3 - equity (state-level, original) |
| `D3_LGA_Equity_Analysis.ipynb` | Domain 3 - equity rebuilt at LGA level (composite index + MGWR) |
| `D4__Domain4_Vaccine_Hesitancy_Methodology.ipynb` | Domain 4 - vaccine hesitancy (Composite Hesitancy Index) |
| `D5__zero_dose_analysis5.ipynb` | Domain 5 - Bayesian zero-dose model (state and LGA) |
| `D6__Copy_of_domain_6_polio_final.ipynb` | Domain 6 - polio surveillance and certification |
| `D7__domain_7_surveillance.ipynb` | Domain 7 - outbreak risk and coverage-to-mortality |
| `NPHCDA_LGA_Archetype_Modeling.ipynb` | LGA archetypes (agglomerative clustering of 15 covariates) |
| `NPHCDA_Cross_Checks_Triangulation.ipynb` | External and convergent validation |

### How to run a notebook after updating the data
1. Open the notebook in **Google Colab** (Colab -> File -> Open notebook -> GitHub / Upload, or open from the Drive folder).
2. When a cell asks for a data file, **upload the updated dataset** (or mount the Google Drive folder) - the file must keep the **same structure** as the sample (see Section 5).
3. **Run all cells** (Runtime -> Run all). Outputs (tables, figures) regenerate from your data.
4. Save the notebook and export any figures you need.

---

## 5. Dataset structure required for upload and re-run

The app validates every upload against its expected schema. Keep the **column names and structure exactly as below** (extra columns are ignored; missing required columns produce a clear error, not a crash).

### DHIS2 routine export (Domains 1, 2, 5) - `dhis2_data_all_states.csv` shape
- **Required columns:** `zone, state, lga, period, penta_1_count, penta_3_count`
- **Recommended:** `bcg_count, pent_2_count, measles_1_count, measles_2_count`
- **Additional antigens (optional):** `opv_0_count, opv_1_count, opv_2_count, opv_3_count, pcv_1_count, pcv_2_count, pcv_3_count, ipv_1_count, ipv_2_count, rota_1_count, rota_2_count, rota_3_count, yellow_fever_count, men_a_count`
- **`period` format:** month-year like `Jan-21`, `Feb-21`, ... one row per LGA per month.
- Sample row: `North West, Kano, Dala, Jan-21, 1450, 1180`

### NDHS zero-dose longitudinal (Domain 5) - `nigeria_ndhs_zero_dose_VERIFIED_long.csv` shape
- **Required:** `state, zone, year, zero_dose_pct, n_children_12_23m`
- One row per state per survey year (2008, 2013, 2018, 2024). `zero_dose_pct` = percent with no Penta1.

### Under-five population (state cohort) - `under_5_2024.csv` shape
- Columns: zone abbreviation, state name, under-five population count. The 12-to-23-month cohort = under-five / 5.

### Administrative LGA population - `administrative_lga_population.csv` shape
- **Required:** `State, Name, Status, PopulationProjection2022-03-21` (Status = "Local Government Area" for LGA rows). Used to weight burden within a state.

### Equity / archetype covariates (Domain 3 and archetypes)
- The LGA covariate master (education, child stunting/wasting/underweight, DPT1-3 dropout, poverty, exclusive breastfeeding, ORS, antenatal care 4+, facility delivery, improved water, travel time, relative wealth, conflict events, conflict fatalities) - see the Google Drive Domain 3 / archetype folder. These feed the LGA equity index and the archetypes.

**Tip:** the safest way to update data is to open the matching sample file in `data/sample/`, replace the values while keeping the headers, and upload the result. Start from the bundled sample first, then upload once your file matches the template.

---

## 6. Google Drive - the dataset folder (data of record)

GitHub does not store large datasets, so the full and harmonized datasets live in the Google Drive folder, organized by domain:
https://drive.google.com/drive/folders/107pWT0m4_A9Sk4aVji-t8LV3m7gfoLzx

Recommended folder organization (by domain), so each notebook's inputs are easy to find:
```
NPHCDA_ZeroDose_Datasets/
  Domain1_2_Coverage_Dropout/     (DHIS2 routine export; consolidated additional-antigens file)
  Domain3_Equity_LGA/             (LGA covariate master; equity index inputs - the state->LGA switch)
  Domain4_Hesitancy/              (ACSM register)
  Domain5_ZeroDose/               (NDHS zero-dose long; under-five; LGA population; GRID3 boundaries)
  Domain6_Polio/                  (AFP line list)
  Domain7_Surveillance/           (IHME GBD cause fractions)
  Archetypes_LGA/                 (the 15-covariate LGA master and archetype outputs)
  README_dataset_dictionary       (source, date and column notes per file)
```
Ensure the **latest harmonized datasets** are present, in particular for the Domain 3 switch from state-level to **LGA-level** equity (the LGA covariate master) and the **archetype** analysis inputs, and that each notebook points to its domain folder.

---

## 7. Reproducibility and methods (brief)
- **Domain 1:** Prophet forecasting per antigen; at-risk-of-decline early-warning at 80% of the 2024 level.
- **Domain 2:** Prophet dropout forecasting across antigen pairs; LASSO driver selection.
- **Domain 3:** LGA composite equity-deprivation index (four quartile tiers) and MGWR (multiscale geographically weighted regression).
- **Domain 4:** Composite Hesitancy Index from the ACSM register, tested against zero-dose.
- **Domain 5:** Bayesian hierarchical Beta regression (PyMC) of state zero-dose, distributed to LGAs by Penta1 throughput and population; Getis-Ord Gi* hotspots; agglomerative (Ward) clustering for archetypes.
- **Domain 6:** Bernoulli spatial scan on the AFP line list; three-year certification survival view.
- **Domain 7:** Pre-specified, false-discovery-rate-corrected coverage-to-mortality tests against IHME GBD.
- **Validation:** six-month hold-out back-tests (2.7-4.3% error); NmDHS 2025-26 out-of-sample check (state rho = 0.88); IHME DTP1 admin-2 surface (LGA rho = 0.60).

Boundary geometry is GRID3 (NPHCDA vaccination boundaries), shipped as simplified GeoJSON. LGA population denominator: National Population Commission (2006 Census, 2022 projection), via City Population.

---

## 8. Deployment notes
- **Hugging Face Space:** built from the `Dockerfile` (the `README.md` frontmatter above configures the Space).
- **Render:** built from `render.yaml` (Python runtime, binds to `$PORT`, `autoDeploy: true` - pushing to GitHub `main` redeploys automatically). Do not change the `$PORT` binding.
- **Docker (portable):** the `Dockerfile` runs the full stack on any Docker host (Render, Railway, Fly, Cloud Run) and can be integrated into another platform.

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA. Funders and reviewers: GAVI and UNICEF.
