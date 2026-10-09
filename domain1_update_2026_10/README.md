# Domain 1: antigen early-warning forecasts for all LGAs (October 2026)

Domain 1 asks which routine immunization antigens are projected to fall below 80% of their 2024 delivery level in the next 6 to 12 months, nationally and in each local government area (LGA).

## Method

The method is the same as in the original Domain 1 analysis:
- a Prophet time-series model for each national and LGA series, with yearly and semi-annual seasonality and a 95% prediction interval;
- each forecast is expressed as % of the same series' 2024 mean monthly doses;
- an early-warning flag is raised when the forecast falls below 80% in months 6-12 after December 2025.

The nine additional antigens use the additional-antigen notebook settings: national series and a 30-month horizon.

## Results (model estimates)

| | |
|---|---|
| National tracer antigens (BCG, Penta1, Penta3, Measles1) | All stay above their 2024 level. Lowest forecasts: 100%, 106%, 107% and 106%. |
| LGA forecasts | 3,092 (773 LGAs x 4 antigens). Guzamala (Borno) has no 2024 data. |
| Early-warning flags | 1,302 in 513 LGAs: Measles1 393, Penta1 320, Penta3 312, BCG 277. |
| Flags with the decline already visible in 2025 | 195 (2025 doses below 80% of 2024). These are first on the worklist. |
| LGAs flagged on all four tracer antigens | 155 |
| Additional antigens | No decline warning for OPV3, IPV1, PCV3, Yellow Fever or Meningitis A. IPV2 is rising; rotavirus doses were 89-91% of 2024 in 2025. |
| Coverage of the eligible cohort, 2026 (routine data) | 125-139% for the four tracers, against 55-69% in NDHS 2024. Routine doses exceed the projected cohort. |

## Contents

| Folder | Content |
|---|---|
| `government_workbook/` | `NPHCDA_Domain1_Antigen_EarlyWarning_LGA_2026.xlsx`, with these sheets: |
| | - national tracers; |
| | - additional antigens; |
| | - coverage vs survey; |
| | - prioritised worklist; |
| | - all LGA forecasts; |
| | - state summary; |
| | - Guzamala month-by-month check. |
| `notebooks/` | Executed Colab notebook `D1_Antigen_EarlyWarning_Forecasts_773_LGAs.ipynb` and its outputs |
| `Updated Datasets for Domain 1/` | The four input files and a README |
| `figures/` | Publication figures (PNG, SVG, PDF) with source data |
| `outputs/corrected/` | Forecast tables from the scripts |
| `code/` | Forecast, figure, workbook and notebook scripts |
