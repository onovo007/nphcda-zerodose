# Domain 2 update (11 October 2026): dropout and completion

Domain 2 rebuilt with DHIS2 counts read correctly. DHIS2 exports counts of 1,000 or more with a thousands separator ("1,234"); these are now read as numbers rather than as missing values.

| Folder | Content |
|---|---|
| `code/d2_rebuild.py` | Rebuilds the Domain 2 figures and tables with the web application's own functions (`data_io.prep_dhis2`, `models/d2_dropout.py`), so they match the Dropout & Completion page |
| `code/fix_and_run_d1d2.py` | Applies the parsing correction to `notebooks/D1_D2__domains_1_2_final.ipynb` and executes it end to end |
| `figures/` | D2_F1 national dropout forecasts; D2_F2 state x year heatmap (observed 2021-2025, forecast 2026-2027); D2_F3 drivers across states (signed LASSO, 2024); D2_F4 drivers across LGAs (signed LASSO, 2021-2024). PNG (300 dpi) and PDF |
| `results/` | National dropout summary, state x year matrices, LASSO coefficients, driver stability, parsing impact and `summary.json` |

**Key results (model estimates).** National Measles1-to-Measles2 dropout fell from 60% in 2021 to 34% in 2025 (forecast about 26% for December 2026) and remains the largest gap; Penta1-to-Penta3 is about 5-7% and Penta1-to-Measles1 about 11-16%. In 2025, 34 of 37 states lost more than 10% of children between the two measles doses.

The scripts expect the local layout used for the project (this repository cloned next to the `Zero-Dose Predictive Modeling` folder). The figures appear in the 10.11.2026 report (Section 5.2, Figures 2-4b) and deck (slides 13-14).
