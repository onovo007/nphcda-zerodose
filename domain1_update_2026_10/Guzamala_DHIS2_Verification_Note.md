# DHIS2 data check: Guzamala LGA, Borno State

**Purpose.** Guzamala is the only one of Nigeria's 774 local government areas (LGAs) without first-dose pentavalent (Penta1) data for 2021-2024. As a result:
- it has no Domain 5 Method 1 estimate;
- it has no Domain 1 forecast.

Please verify the record in the DHIS2 system at source.

| Item | Detail |
|---|---|
| State / zone | Borno, North East |
| DHIS2 organisation unit | `bo Guzamala Local Government Area` |
| Period with no data for any antigen | January 2021 to January 2025 (49 consecutive months) |
| First month with any antigen reported | February 2025 (BCG, Penta3, Measles1; Penta1 blank) |
| Months with Penta1 reported | April, June, July, August, September, October and November 2025: 7, 6, 9, 18, 18, 27 and 18 doses (103 in total) |
| 2025 months with no data for any antigen | January, March and May 2025 |
| December 2025 | BCG, OPV1 and Rotavirus 1 reported; Penta1 blank |

## Questions for the DHIS2 team

1. Were Guzamala's health facilities active and reporting in 2021-2024, with their data entered under another organisation unit (for example a neighbouring LGA or the state)?
2. Are the 2025 values complete, or are late reports still to be entered?
3. Does the Penta1 data element exist for all of Guzamala's facilities?

The full month-by-month record for all 20 antigens is in `NPHCDA_Domain1_Antigen_EarlyWarning_LGA_2026.xlsx` (sheet "Guzamala check") and in `outputs/Guzamala_Borno_DHIS2_records_2021_2025.csv`.

*Source: NPHCDA DHIS2 routine immunization export by LGA and month, January 2021 to December 2025.*
