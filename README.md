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

# NPHCDA Zero-Dose Predictive Modelling Platform (Domains 1, 2, 5)

A real-time, no-code modelling tool for the NPHCDA Digital Innovation Hub. Upload the raw
immunization data sources and the app runs the Domain 1, 2 and 5 models live and renders the
same tables and visualizations as the analytical notebooks.

- **Domain 1 - Antigen coverage forecasting.** National Prophet forecasts of BCG, Penta1,
  Penta3 and Measles1 as a percent of the 2024 baseline, against the 80 percent target line;
  at-risk-LGA screen.
- **Domain 2 - Dropout and completion dynamics.** Prophet forecasts of Penta1 to Penta3,
  Penta1 to Measles1 and Measles1 to Measles2 dropout, with LASSO-selected drivers and
  state-by-year dropout heatmaps.
- **Domain 5 - Zero-dose modelling and hotspots.** A Bayesian hierarchical Beta regression
  (PyMC) of state zero-dose rates with credible intervals, population-weighted LGA burden,
  Pareto prioritization, and Getis-Ord Gi* hotspot maps.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

The app ships with the canonical sample inputs in `data/sample/`, so it runs end to end with no
upload. Use the "Use bundled sample data" button on the Home page, or upload your own files on the
Data and Quality page.

## Data sources expected on upload

| File | Role |
| --- | --- |
| DHIS2 routine export (`dhis2_data_all_states.csv` shape) | Domains 1, 2, 5 dose counts |
| NDHS zero-dose longitudinal (`nigeria_ndhs_zero_dose_VERIFIED_long.csv`) | Domain 5 Bayesian model |
| Zero-dose model dataset (equity covariates) | Domain 2 LASSO drivers |
| Under-five population 2024 | State 12-to-23-month cohort |
| Administrative LGA population (NPC 2022) | Domain 5 within-state burden weighting |

## Performance notes

Heavy models run live but are scoped so each click returns quickly: the Domain 5 state Bayesian
model and the national Prophet forecasts fit in seconds to about a minute. Full per-LGA Prophet
and the full 3000-draw posterior are available behind an explicit "run full" control.

## Reproducibility

Modelling logic is lifted from the project notebooks (Domains 1, 2, 5). Boundary geometry is
GRID3 (NPHCDA vaccination boundaries), shipped as simplified GeoJSON. LGA population denominator:
City Population (citypopulation.de), compiled from the National Population Commission of Nigeria
(2006 Census, 2022 projection).

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA. Funders and
reviewers: GAVI and UNICEF.
