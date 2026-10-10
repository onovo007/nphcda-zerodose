# Domain 5: zero-dose children for all LGAs, two Bayesian methods (October 2026)

Domain 5 of the NPHCDA Predictive Modelling of Zero-Dose Children programme estimates where zero-dose children are concentrated, who they are, and where to act first. This folder contains the October 2026 Domain 5 deliverables.

Zero-dose children are children aged 12-23 months who have not received the first dose of the pentavalent vaccine. All figures are model estimates for 2026.

## The two methods

| | Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation | Method 2: Bayesian small-area estimation (SAE) |
|---|---|---|
| State level | Beta regression on NDHS 2008-2024, with a DHIS2 Penta1 trend term | Survey-anchored aggregation likelihood on NDHS 2008-2024 |
| LGA level | State estimate distributed by LGA share of reported Penta1 doses and NPC population | Each LGA modelled directly using six local covariates and a BYM2 spatial term |
| LGAs estimated | 773 (Guzamala, Borno, reported no Penta1 doses in 2021-2024) | 774 |
| Zero-dose children, 2026 | 2.10 million (95% interval 1.90-2.31) | 2.15 million (95% CrI 1.86-2.46) |
| LGAs needed to reach 50% / 60% / 80% of the burden | 151 / 199 / 342 | 138 / 181 / 319 |
| NmDHS 2025-26, 37 states (Spearman rho) | 0.89 | 0.87 |
| IHME 2018, LGAs (overall / within state) | 0.65 / 0.08 | 0.89 / 0.54 |

## Contents

| Folder | Content |
|---|---|
| `report_and_deck/` | RI team presentation and consolidated report (Word and PDF). The 10.10.2026 versions (54 slides; report Sections 5.5.6 and 5.5.7) add external validation, calibration, temporal holdouts and uncertainty-aware prioritization for Method 2, and contextual profiles, dominant barriers and candidate intervention packages. Profile labels describe only what the indicators measure. |
| `government_workbook/` | `NPHCDA_ZeroDose_LGA_Estimates_2026_Method1_vs_SAE.xlsx`. It compares the two methods for every LGA, with national, zone and state tables, priority lists, burden concentration, validation, archetypes and definitions. |
| `Updated Datasets for Domain 5/` | All inputs needed to run the three notebooks, one subfolder per notebook. See its README. |
| `notebooks/` | Executed notebooks with their outputs: |
| | - `D5_Method1_Bayesian_Hierarchical_Model_LGA_Allocation.ipynb` |
| | - `D5_Method2_Bayesian_Small_Area_Estimation.ipynb`, written for readers new to SAE |
| | - `D5_LGA_Archetypes_Unsupervised_Clustering.ipynb` |
| `results/` | Combined tables for both methods: LGA, state, zone and national estimates; validation; Pareto scenarios; archetype burden |
| `figures/` | Publication figures (PNG, SVG, PDF) with source data |
| `code/` | Scripts that combine the notebook outputs and build the figures, workbook, deck and report. `update_10_10.py` builds the 10.10.2026 deck and report; `build_app_evidence_10_10.py` writes the web-app evidence files in `data/sample/two_methods/evidence/`. |

## Reproducing

1. Open a notebook in Google Colab.
2. Choose Runtime > Run all.
3. Upload the files from the matching subfolder of `Updated Datasets for Domain 5`.

The notebooks save their outputs, including posterior draws. `code/final_results.py` combines those outputs into `results/`.
