# Updated Datasets for Domain 1

These are the inputs for the notebook `D1_Antigen_EarlyWarning_Forecasts_773_LGAs.ipynb` (in `notebooks/`). The notebook answers the Domain 1 question: which antigens are projected to fall below 80% of their 2024 delivery level in the next 6 to 12 months, nationally and in each local government area (LGA).

| File | Content | Source |
|---|---|---|
| `dhis2_data_all_states.csv` | Monthly doses by LGA, January 2021 to December 2025: BCG, Penta1-3, Measles1-2, OPV0-3, PCV1-3 | NPHCDA DHIS2 |
| `dhis2_data_additional_antigens.csv` | Monthly doses by LGA: IPV1-2, Rotavirus 1-3, Yellow Fever, Meningitis A | NPHCDA DHIS2 |
| `under_5_2024.csv` | State under-five population projection, 2024 (eligible cohort = under-five / 5) | NPC |
| `ndhs_antigens2024.csv` | State vaccination coverage among children aged 12-23 months, NDHS 2023-24 | DHS Program |

## How to run in Google Colab

1. Open the notebook.
2. Choose Runtime > Run all.
3. Select the four files when asked.

The LGA step fits about 3,100 models. It takes about 10-20 minutes on Colab and about 2 minutes on a multi-core computer.

## Points to know about the data

- **Thousands separators.** DHIS2 prints counts of 1,000 or more with a thousands separator (for example `"1,234"`). Read the files with `thousands=","`; otherwise these values are lost.
- **Late-starting states.** Katsina, Kogi, Kwara, Lagos and Niger have no rows before January 2023.
- **Guzamala (Borno).**
  - It reported no doses of any antigen from January 2021 to January 2025, so it has no 2024 baseline and no forecast.
  - Penta1 was reported in April and June to November 2025.
  - Its month-by-month record is in the notebook (section 8) and in the Domain 1 government workbook ("Guzamala check").

## Data use

These files contain NPHCDA routine immunization data. Share them in line with NPHCDA data-sharing arrangements.
