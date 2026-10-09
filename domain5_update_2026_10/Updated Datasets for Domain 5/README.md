# Updated Datasets for Domain 5

Domain 5 of the NPHCDA Predictive Modelling of Zero-Dose Children programme asks where zero-dose children are concentrated, who they are, and where to act first. This folder holds every input needed to reproduce the Domain 5 results from start to finish in Google Colab or on a local computer. There is one subfolder for each notebook:

| Folder | Notebook | What it produces |
|---|---|---|
| `Method1_Bayesian_Hierarchical_Model/` | `D5_Method1_Bayesian_Hierarchical_Model_LGA_Allocation.ipynb` | State zero-dose forecasts for 2026-2028, and 2026 estimates for 773 LGAs (local government areas) with 95% intervals |
| `Method2_Bayesian_Small_Area_Estimation/` | `D5_Method2_Bayesian_Small_Area_Estimation.ipynb` | 2026 zero-dose estimates for all 774 LGAs, each with a 95% credible interval and a priority probability |
| `LGA_Archetypes/` | `D5_LGA_Archetypes_Unsupervised_Clustering.ipynb` | Five structural archetypes for all 774 LGAs, with zero-dose burden by archetype under both methods |

The three notebooks are in the `notebooks/` folder next to this README.

Zero-dose children are children aged 12-23 months who have not received the first dose of the pentavalent vaccine (Penta1/DTP1). Every output is a model estimate.

## How to run in Google Colab

1. Open the notebook in Colab (File > Upload notebook).
2. Choose Runtime > Run all.
3. When the first cell asks, select all the files in the notebook's subfolder; you can select them all at once.
   - Alternatively, copy this folder to Google Drive and follow "Option B" in the first cell.
4. Outputs (CSV tables, figures and posterior draws) are written to `outputs/`. You can download them from the Colab file panel.

Approximate run times on a standard Colab CPU:

| Notebook | Approximate run time |
|---|---|
| Method 1 | about 2-5 minutes |
| Method 2 | about 5-15 minutes |
| Archetypes | under 1 minute |

The first cell installs the packages each notebook needs.

## How to run locally

Use Python 3.11 or later and install the packages:

```
pip install pymc nutpie arviz geopandas libpysal esda scikit-learn openpyxl matplotlib
```

Then point the notebook at its data folder and run it:

```
set D5_DATA_DIR=...\Updated Datasets for Domain 5\Method2_Bayesian_Small_Area_Estimation
set D5_OUT_DIR=outputs_Method2_Bayesian_Small_Area_Estimation
jupyter nbconvert --to notebook --execute D5_Method2_Bayesian_Small_Area_Estimation.ipynb
```

Run Method 1 and Method 2 before the archetype notebook if you want to refresh `lga_zero_dose_estimates_2026_both_methods.csv`.

## The two methods

- **Method 1: Bayesian hierarchical model with DHIS2-calibrated LGA allocation.**
  - A Beta regression on the logit scale estimates each state's zero-dose rate from the Nigeria Demographic and Health Surveys (NDHS) 2008, 2013, 2018 and 2023-24.
    - It uses zone and state partial pooling and a DHIS2 Penta1 trend term.
    - It forecasts to 2026-2028.
  - Each state's estimate is then distributed to its LGAs using each LGA's share of reported Penta1 doses and its National Population Commission (NPC) 2022 population.
  - Guzamala (Borno) reported no Penta1 doses in 2021-2024, the years the allocation uses (partial reporting began in 2025), so Method 1 does not estimate it.
- **Method 2: Bayesian small-area estimation (SAE).**
  - Each LGA has its own zero-dose rate.
  - The population-weighted average of a state's LGA rates is linked to the state's NDHS results through a measurement-error likelihood (design effect 2).
  - Differences between LGAs come from three sources:
    - six local covariates;
    - a BYM2 (Besag-York-Mollie, version 2) spatial effect on the LGA neighbour graph;
    - zone effects.
  - Method 2 does not use routine coverage data, and it estimates all 774 LGAs.

Both methods use the same population base: children aged 12-23 months = state under-five population (2024 projection) / 5, shared among LGAs by NPC 2022 LGA population.

## Files

### Method1_Bayesian_Hierarchical_Model/

| File | Content | Source |
|---|---|---|
| `nigeria_ndhs_zero_dose_VERIFIED_long.csv` | State zero-dose (%) by NDHS round, 37 states x 4 rounds | NDHS 2008-2024 (DHS Program) |
| `dhis2_data_all_states.csv` | Monthly antigen doses by LGA, 2021-2025. Dose counts above 999 contain thousands separators (for example "1,234"); the notebook reads them as numbers | NPHCDA DHIS2 |
| `under_5_2024.csv`, `under_5_2025.csv` | State under-five population projections | NPC projections |
| `administrative_lga_population.csv` | LGA population, 2022 | NPC |
| `nga_lgas.geojson`, `nga_states.geojson` | LGA and state boundaries | GRID3 |
| `ihme_dtp1_admin2_2018.csv` | DTP1 coverage by LGA, 2018 (validation only) | IHME Local Burden of Disease |
| `nmdhs_2025_26_penta1_by_state_zone.csv` | Penta1 coverage by state and zone (validation only) | Nigeria mini DHS 2025-26 |

### Method2_Bayesian_Small_Area_Estimation/

| File | Content |
|---|---|
| `sae_lga_inputs_774.csv` | One row per LGA (774) with three groups of fields (detailed below) |
| `lga_adjacency_grid3_queen.csv` | LGA neighbour pairs (Queen contiguity on GRID3 boundaries; 2,165 pairs) |
| `nigeria_ndhs_zero_dose_VERIFIED_long.csv`, `under_5_2024.csv`, `under_5_2025.csv`, `nmdhs_2025_26_penta1_by_state_zone.csv`, geojsons | As above |

`sae_lga_inputs_774.csv` contains:
- **Identifiers and population:**
  - LGA identifier;
  - zone, state and LGA;
  - NPC 2022 population and share of state;
  - cohort aged 12-23 months.
- **Covariates:** ANC4+, facility delivery, improved water, Relative Wealth Index, travel time to a health facility, conflict events and poverty.
- **Reference fields:**
  - DHIS2 Penta1 summaries, used for comparison only and not in the model;
  - archetype;
  - IHME 2018 zero-dose, used for validation only.

### LGA_Archetypes/

| File | Content |
|---|---|
| `lga_archetype_covariates_774.csv` | 15 social, health-service, nutrition, geographic and security covariates for 774 LGAs, with the archetype assignment |
| `archetype_covariate_sources.csv` | Definition, year, source and direction of each covariate |
| `lga_zero_dose_estimates_2026_both_methods.csv` | 2026 LGA zero-dose estimates from Method 1 and Method 2 (attached to describe the archetypes; not used to form them) |
| `lga_map_keys_774.csv` | Keys linking each LGA to the GRID3 boundary file |
| `nga_lgas.geojson`, `nga_states.geojson` | Boundaries |

## Covariate sources

- DHS Spatial Data Repository modelled surfaces (2018): antenatal care with four or more visits, facility delivery, improved water.
- Meta Relative Wealth Index.
- Travel time to healthcare: Weiss et al., Malaria Atlas Project.
- Armed Conflict Location and Event Data (ACLED), 2021-2024.
- Poverty surface.
- IHME modelled surfaces (archetype covariates).

All are summarised to GRID3 LGA boundaries.

## Data use

These files combine:
- public survey and geospatial products;
- NPHCDA routine immunization data (DHIS2).

Use and share the DHIS2-derived files in line with NPHCDA data-sharing arrangements. Cite the original sources when using the files.
