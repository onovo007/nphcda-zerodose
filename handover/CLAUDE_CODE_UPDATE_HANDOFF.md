# Claude Code update handoff - NPHCDA Zero-Dose Platform

**Read this fully before making changes.** It is written for a Claude Code session on a new laptop that will make significant corrections and updates to the web app (Hugging Face), the notebooks, the report, and the slide deck. House style throughout: hyphens only (no em/en dashes), American spelling but keep "modelling" and "programme", plain language, spell out acronyms, label model outputs "model estimate".

---

## 0. THE PRIORITY TASK - data correction (do this first, it changes headline numbers)

**What was wrong:** We previously excluded **44 local governments** (LGAs) from the zero-dose modelling for "missing Penta1 (DTP1) data", leaving **730 modelled**. This was **incorrect**. The Penta1 data for **43 of those 44 LGAs was actually reported** - the problem was a **name-mapping failure** between the DHIS2 export and the GRID3 / platform admin-2 names. The **fuzzy string matching** used to bridge names did not align 43 of the 44. **Only ONE LGA truly did not report Penta1.**

**Consequence:**
- The correct count is **773 modelled / 1 excluded** (not 730 / 44).
- Additional **data-row issues** were also found, so the **total zero-dose burden increases** (currently reported as ~2.08 million / 2,085,312 across 730 LGAs; it must be re-computed and will be higher).
- This flows into the **report, the slide deck, and the web app** - all three must be updated, plus the 44-LGA data-quality documents.

**WORK ALREADY IN PROGRESS (continue from here, do not redo):** a prior session recovered **30 LGAs** lost to name mismatches in the **archetype burden join** (archetype files now reach the 2,085,312 total) and fixed the FCT area councils that were filed under Enugu. STILL TO DO: (a) extend the name-mapping fix into the **Domain 5 zero-dose MODEL** itself, where the 730/44 exclusion lives (not just the archetype files) so the burden total is recomputed on 773 LGAs; (b) reconcile the true exclusion to **1 LGA**; (c) propagate the new totals to the report, deck, web app and the excluded-LGA documents. Check `git log` for the latest state before starting.

### Where the name-matching / inclusion logic lives (fix here)
The exclusion was a side effect of fuzzy matching failing, so Penta1 looked absent. The fix is a **deterministic DHIS2-to-GRID3 admin-2 crosswalk** (a lookup table), replacing or backstopping the fuzzy matching at every site:

| File | Line(s) | What it does | Action |
| --- | --- | --- | --- |
| `models/d5_zerodose.py` | ~356 | `lga_all = lga_all[lga_all["penta_1_count"] > 0]` - drops LGAs with no Penta1 after the name join | The drop is correct in principle; the problem is upstream name mapping. Ensure the DHIS2 LGA names map to the platform names BEFORE this filter |
| `models/d5_zerodose.py` | ~427 | `difflib.get_close_matches(..., cutoff=0.80)` - fuzzy LGA-population match | Replace with crosswalk lookup first; fuzzy only as last resort |
| `names.py` | `clean_lga_name`, `nlga`, `nstate`, `tok`, `LGA_ALIAS`, `FCT6` | name normalization + a tiny alias dict (currently only 2 aliases) | **Expand `LGA_ALIAS` (or add a crosswalk CSV) to cover the 43 mis-mapped LGAs** |
| `Archtyping at LGA level/arch_geo.py` | ~38, ~41 | fuzzy geojson join (cutoff 0.82, 0.90) | Same - crosswalk first |
| `Archtyping at LGA level/finalize_master.py` | ~141, ~144 | fuzzy zero-dose join (cutoff 0.82, 0.90) | Same - crosswalk first |

**Recommended approach:** build `data/sample/dhis2_grid3_lga_crosswalk.csv` with columns `state, dhis2_lga_name, platform_lga_name` for the 43 (plus any others), load it in `names.py`/`data_io.py`, and apply it before any fuzzy fallback. Validate that all 774 LGAs (773 reporting + 1 true non-reporter) now join. Keep the fuzzy matching only as a backstop with a logged warning.

### Downstream regeneration AFTER the data fix (critical - the app serves precomputed results)
The app loads **precomputed** Domain 5 results for the bundled data (that is the crash-proofing). After the correction you MUST regenerate them or the app will show the OLD numbers:
1. Update the bundled DHIS2 data if the source changed: `data/sample/dhis2_data_all_states.csv` (and the consolidated/additional-antigen files in `03_Datasets/datasets/`).
2. Re-run `python precompute_d5.py` -> regenerates `data/sample/precomputed/` (`state_res.parquet`, `state_diag.parquet`, `state_meta.json`, `lga_clean.parquet`, `lga_pareto.parquet`, `lga_stats.json`, `gi_meta.json`, `lga_gi.parquet`, `state_gi_2026/2027/2028.parquet`). The fingerprint will change - that is expected.
3. Re-run the archetype pipeline in `Archtyping at LGA level/` (build_lga_archetype_master -> cluster_lga_archetypes -> finalize_master -> domain3_composite -> domain3_mgwr) to refresh the 773-LGA master, equity index, MGWR, and figures.
4. Regenerate the LGA priority ranking (`regen_priority.py`) and the bundled `data/sample/lga_archetype_master.csv`, `lga_priority_ranking.csv`, `lga_archetype_summary.csv`.
5. Run `_smoke_test.py` to confirm everything still works end to end.
6. Update the CURRENT artifacts in `handover/` (file names change over time - check `ls handover/`): the report `Final_NPHCDA_Consolidated_Report_UPDATED_10.6.2026.docx`, the deck `NPHCDA_ZeroDose_RI_Team_Presentation_ED_Updates.10.6.2026.pptx`, the excluded-LGA files `NPHCDA_Excluded_44_LGAs_DataQuality.*` (rename/retitle to reflect **1 excluded**), the priority lists `NPHCDA_LGA_Priority_Lists_*.xlsx`, and the archetype list `NPHCDA_LGA_Archetype_List_updated_*.xlsx`. They must all state **1 excluded / 773 modelled** and the revised burden total.
7. Update every place that says "730", "44", "2.08 million" / "2,085,312", "62% in top 20%", "top 146 / top 270" - re-derive from the new results.

**Verify before push:** `_smoke_test.py` passes; the Zero-Dose page total reflects the new burden; the MGWR map white cells drop to ~1 (the true non-reporter) plus any missing-covariate LGAs.

---

## 1. Web app architecture (Hugging Face)

- **Framework:** Streamlit 1.54 on Python 3.12, in a Docker Space. Entry point `app.py`.
- **Deployment:** Hugging Face Space (built from `Dockerfile`; the `README.md` frontmatter configures the Space) and Render (built from `render.yaml`, Python runtime, binds `$PORT`, autoDeploy on GitHub push).
- **CRASH-PROOFING (do not undo):**
  - `requirements.txt` pins `numpy==1.26.4` and **`pyarrow==16.1.0`**. The pyarrow pin is the ROOT fix for the exit-139 segfault (a numpy-2-era pyarrow crashes `st.dataframe` -> Arrow against numpy 1.26). With numpy<2, pin every compiled numpy-dependent wheel to a numpy-1.x build.
  - `faulthandler` is enabled in `app.py` so any native crash names the faulting line in the HF logs.
  - Dockerfile sets thread limits (OMP/OPENBLAS/MKL/NUMBA/RAYON) to avoid thread-pool oversubscription.
  - Domain 5 live PyMC runs in an isolated child process (`isolation.py` + `model_worker.py`) so a native fault cannot take the app down; bundled data uses the precomputed results (fingerprint-gated), so no live sampling for the demo.
  - Recovery if it crash-loops: plain **Restart** (~1 min), NOT factory rebuild.
- **Page modules** (nav in `app.py` `sidebar()`): Home, Data and Quality, Coverage Forecasting (`domain1.py`), Dropout & Completion (`domain2.py`), Zero-Dose & Hotspots (`domain5.py`), LGA Priority & Archetypes (`lga_priority.py`), Triangulation & Cross-Checks (`crosschecks.py`, `triangulation.py`), Exploratory Data Analysis (`impsci.py`), Ask the Analyst (`ai.py`), Reports & Briefs (`reports.py`), Program Q&A RAG (`rag.py`), User Guide (SOP) and Methods & Validation.
- **Core modules:** `config.py` (paths, antigen sets, palettes, MCMC settings, SCHEMAS for upload), `data_io.py` (prep_dhis2, national_monthly, prep_under5, schema read/validate), `data_quality.py`, `names.py` (admin-2 name harmonization - KEY for the fix), `spatial.py` (GRID3 geojson + Getis-Ord Gi*; precomputed-gated), `viz.py` (plotly figures), `theme.py`, `stats_infer.py`, `auth.py` (email allow-list), `llm.py` (OpenAI calls), `ai.py` (grounded interpretation blocks).
- **Models:** `models/d1_forecast.py` (Prophet; antigens parameterised via `_antigens`; established vs recently-introduced antigen sets), `models/d2_dropout.py`, `models/d5_zerodose.py` (Bayesian hierarchical Beta via PyMC/nutpie + LGA burden distribution + precompute hooks).

## 2. ZARA assistant and AI interpretation

- **ZARA** = Zero-dose Analytics and Risk Assistant. Implemented in `llm.py` (`interpret`, `chat`, `tts`, `embed`, `rag_answer`, `compose_brief`) and surfaced via `ai.py` `ai_block(...)` under each output, plus the "Ask the Analyst" and "Program Q&A (RAG)" pages.
- **Models:** OpenAI - `gpt-4o-mini` (default), `gpt-4o`, `gpt-5`, `gpt-4.1-mini`. **Languages:** English, French, Spanish, Hausa, Nigerian Pidgin, Yoruba, Igbo, Swahili. TTS: English/French/Spanish/Swahili.
- **Key:** the API key is read ONLY from the sidebar per session (`ai._cfg` -> `st.session_state["llm"]`). There is NO shared-key fallback yet. To enable AI for all users without each entering a key, add an `OPENAI_API_KEY` server-side secret fallback in `ai._cfg`/sidebar and set a spending cap. Without a key the app still works; only AI text is hidden. ZARA is instructed never to invent numbers (grounded on the live outputs).

## 3. Datasets (exact CSV names used in modelling)

Bundled with the app in `data/sample/`:
- `dhis2_data_all_states.csv` - monthly LGA dose counts (BCG, Penta1-3, Measles1-2, OPV0-3, PCV1-3); Domains 1, 2, 5. (+ additional antigens live in `03_Datasets/datasets/dhis2_data_additional_antigens.csv` and the harmonized `dhis2_data_consolidated.csv`.)
- `dhis2_data_live_births.csv` - monthly live births by LGA.
- `nigeria_ndhs_zero_dose_VERIFIED_long.csv` - state zero-dose by survey wave (2008-2024); Domain 5 Bayesian model.
- `nigeria_zero_dose_model_dataset.csv` - 37-state equity covariates; Domain 2 LASSO, original state-level Domain 3.
- `under_5_2024.csv`, `under_5_2025.csv` - under-five population by state (cohort = under-five / 5).
- `administrative_lga_population.csv` - LGA population (NPC 2022 projection); within-state burden weighting.
- `ndhs_antigens2024.csv` - NDHS antigen coverage 2024 (admin-vs-survey checks).
- `lga_archetype_master.csv` - 15 LGA covariates + archetype + modelled zero-dose (Domain 3 LGA equity + archetypes).
- `lga_archetype_summary.csv`, `lga_priority_ranking.csv` - archetype summary and ranked priority list (outputs).
- `domain2_lasso_coefficients_lga.csv` - LGA LASSO driver coefficients.
- `ihme_dtp1_admin2_2018.csv` - IHME DTP1 admin-2 surface (LGA external validation).
- `geo/nga_lgas.geojson`, `geo/nga_states.geojson` - GRID3 boundaries.

Full and harmonized datasets (data of record) are in Google Drive, organized by domain: see `handover/DATA_DICTIONARY.md` for source, date, and key columns of each. Drive: https://drive.google.com/drive/folders/107pWT0m4_A9Sk4aVji-t8LV3m7gfoLzx

## 4. Notebooks (Google Colab, in `notebooks/`)
- `D1_D2__domains_1_2_final.ipynb` - coverage forecasting + dropout.
- `D1_Additional_Antigens_Forecast.ipynb` - OPV/IPV/PCV/Yellow Fever/Men A/Rotavirus extension.
- `D3__equity_zero_dose_stratification.ipynb` - equity, state level (original).
- `D3_LGA_Equity_Analysis.ipynb` - equity rebuilt at LGA level (composite index + MGWR).
- `D4__Domain4_Vaccine_Hesitancy_Methodology.ipynb` - Composite Hesitancy Index.
- `D5__zero_dose_analysis5.ipynb` - Bayesian zero-dose model (state + LGA).
- `D6__Copy_of_domain_6_polio_final.ipynb` - polio surveillance + certification.
- `D7__domain_7_surveillance.ipynb` - outbreak risk + coverage-to-mortality.
- `NPHCDA_LGA_Archetype_Modeling.ipynb` - agglomerative clustering of 15 covariates.
- `NPHCDA_Cross_Checks_Triangulation.ipynb` - external + convergent validation.

## 5. Methods per domain (brief)
- **D1** Prophet forecasting per antigen; at-risk-of-decline early-warning at 80% of the 2024 level (established antigens); recently introduced antigens (IPV2, Rotavirus) shown as uptake, no decline flag.
- **D2** Prophet dropout across antigen pairs; bootstrap-stable LASSO drivers.
- **D3** LGA composite equity-deprivation index (travel time, low education, poverty, low wealth; quartile tiers); MGWR (OLS 0.39 -> MGWR 0.55).
- **D4** Composite Hesitancy Index from the ACSM register; decoupled from zero-dose (rho = -0.33).
- **D5** Bayesian hierarchical Beta regression (PyMC/nutpie) of state zero-dose on NDHS 2008-2024; distributed to LGAs by Penta1 throughput and population; Getis-Ord Gi* hotspots; Ward clustering for archetypes.
- **D6** Bernoulli spatial scan on the AFP line list; three-year certification survival view.
- **D7** Pre-specified, false-discovery-rate-corrected coverage-to-mortality tests vs IHME GBD 2023.
- **Validation** six-month hold-out back-tests (2.7-4.3% error); NmDHS 2025-26 (state rho = 0.88); IHME DTP1 surface (LGA rho = 0.60). Note: these validation numbers may shift slightly after the 773-LGA correction - re-run the cross-checks.

## 6. Access and deployment (for pushing updates)

- **GitHub:** `github.com/onovo007/nphcda-zerodose` (branch `main`). On the new laptop: `git clone`, then first push prompts for Username `onovo007` + a **GitHub Personal Access Token** (fine-grained, repo Contents read/write). Render auto-deploys on push to `main`.
- **Hugging Face Space:** `huggingface.co/spaces/amobionovo/nphcda-zerodose`. Add it as a second remote and push separately:
  `git remote add hf https://huggingface.co/spaces/amobionovo/nphcda-zerodose` then `git push hf main` (Username `amobionovo` + an **HF write token**).
- **SECURITY:** do NOT commit tokens or keys. The HF token used during development must be **revoked** and a fresh one generated on the new laptop. The app's email allow-list is the `ALLOWED_EMAILS` Space secret; the OpenAI key for ZARA is a per-session sidebar entry (or a future server-side secret).
- **Both remotes mirror the same code**; push to `origin` (GitHub -> Render) and `hf` (Hugging Face) to update both.

## 7. Run and test locally
```
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501     # app; any name/email locally
python _smoke_test.py                                 # end-to-end check of Domains 1,2,5 + spatial
python precompute_d5.py                               # regenerate precomputed results after a data change
```

## 8. Guardrails (do not break)
- Keep `numpy==1.26.4` and `pyarrow==16.1.0` pinned; keep `faulthandler` on; keep the precompute + isolation.
- After any change to the bundled data or the model, re-run `precompute_d5.py` or the app will serve stale numbers.
- Keep `render.yaml` binding to `$PORT`.
- Run `_smoke_test.py` and verify the app renders (sign-in -> Use bundled sample data -> open each page) before pushing.
- When the correction lands, update the report, deck, figures, and the 44-LGA documents consistently - the headline numbers (773 modelled / 1 excluded / new burden total) must match everywhere.

## 9. Exact propagation map - where the 730/44/2.09M numbers live (current 10.6.2026 files)

After recomputing on 773 LGAs, every item below must change to **773 modelled / 1 excluded** and the **revised (higher) burden total**. Re-derive the percentages and LGA counts from the new results (do not hand-edit without re-running).

**Deck - `NPHCDA_ZeroDose_RI_Team_Presentation_ED_Updates.10.6.2026.pptx`:**
- Slide 2: "All 730 reporting LGAs... 146 LGAs cover 62 percent"
- Slide 9 (whole slide): "The 44 local government areas we could not model"; "730 of the 774... met the standard"; the breakdown "23 North-West, 16 North-East, 3 North-Central, 2 South-South" -> rewrite to **1 excluded / 773 modelled**, and reframe the 43 as a DHIS2-to-GRID3 **name-mapping correction**, not missing data
- Slide 10: "591 of the 730 reporting areas" (recompute denominator)
- Slide 17: "730 modelled local government areas"; "62% top 20%"; "270 hold 80%"
- Slide 18: "top 150 reach 62 percent; top 270 reach 80 percent"
- Slides 20 and 22: "146"
- Slide 26 (cross-check): "Only 44 were far apart" vs the IHME surface - re-run the concordance
- Slide 27: "730 reporting... national total 2.09 million"
- Slide 31: "about 150 LGAs reach roughly 62%"

**Report - `Final_NPHCDA_Consolidated_Report_UPDATED_10.6.2026.docx`:**
- Table 3b "The 44 LGAs excluded from DHIS2-based modelling" -> revise to the **1 true non-reporter** (optionally add a table of the 43 recovered by the name-mapping fix)
- "Of Nigeria's 774 LGAs, 730 met the Penta1 reporting standard... Forty-four were excluded" -> **773 / 1**
- The remediation narrative ("in most of these LGAs other antigens report normally... the Penta1 data element itself has been dropped or misconfigured") -> **reframe**: 43 were a DHIS2-to-GRID3 admin-2 name-mapping mismatch (now corrected); only 1 is a genuine non-reporter
- Figure 14 "Pareto... across 730 reporting LGAs (total approx 2.09 million)" -> 773 + new total
- "top 146 LGAs... close to two-thirds"; "top 270" -> recompute
- "Archetypes 1 and 2 hold approximately 1.50 million of the roughly 2.09 million" -> recompute
- "Forty-four LGAs (Table 3b)... concentrated in the North-West and North-East" -> revise

**Web app:** these numbers are produced live/precomputed, so they update automatically once the model is re-run and `precompute_d5.py` regenerates `data/sample/precomputed/` - no hand-editing needed, but verify the Zero-Dose page total and the LGA Priority counts after regeneration.

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA; funders and reviewers GAVI and UNICEF.
