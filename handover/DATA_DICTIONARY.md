# NPHCDA Zero-Dose Platform - Dataset Dictionary

Source, reference date, and key columns for every dataset used across the notebooks and the web application. The large and harmonized datasets live in the Google Drive folder (GitHub does not host large files), organized by domain to match this dictionary.

**Google Drive datasets:** https://drive.google.com/drive/folders/107pWT0m4_A9Sk4aVji-t8LV3m7gfoLzx

Consortium: CIDRE and Quantium Insights LLC, in technical support of NPHCDA; funders and reviewers GAVI and UNICEF.

---

## Domain 1 and 2 - Coverage forecasting and dropout
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| `dhis2_data_all_states.csv` | Monthly LGA dose counts (BCG, Penta1-3, Measles1-2, OPV0-3, PCV1-3) | DHIS2 routine | 2021-2025 | zone, state, lga, period, `*_count` |
| `dhis2_data_additional_antigens.csv` | Additional antigens: IPV1-2, Rotavirus 1-3, Yellow Fever, Men A | DHIS2 routine | 2021-2025 | zone, state, lga, period, `*_count` |
| `dhis2_data_consolidated.csv` (harmonized) | All antigens merged into one file (the file of record for forecasting) | DHIS2 routine | 2021-2025 | zone, state, lga, period, all `*_count` |
| `dhis2_data_live_births.csv` | Monthly live births by LGA | DHIS2 routine | 2021-2025 | zone, state, lga, period, live_births |

`period` format is month-year, e.g. `Jan-21`; one row per LGA per month.

## Domain 3 - Equity (state-to-LGA switch)
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| `lga_archetype_master.csv` / `LGA_archetype_master.xlsx` | The 15 LGA covariates + archetype + modelled zero-dose; the basis of the LGA-level equity rebuild | IHME, DHS, Meta, Weiss, ACLED | see covariate dates below | platform_State, platform_LGA, archetype, zero_dose_*, 15 covariates |
| `lga_equity_ranking.csv` | LGA composite deprivation index (0-100) and four quartile tiers | Derived (Domain 3) | 2026 | equity_rank, equity_deprivation_index, equity_tier |
| `lga_combined_priority_ranking.csv` | Zero-dose burden rank + equity tier + archetype + priority flag | Derived (Domain 3/5) | 2026 | burden rank, equity_tier, archetype_type |
| `nigeria_zero_dose_model_dataset.csv` | The original STATE-level equity covariates (37-state file) | NDHS + covariates | 2018-2024 | state, 30+ predictors |

## Domain 4 - Vaccine hesitancy
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| `acsm_cleaned.csv` | 2,593 advocacy and social-mobilization activities, 36 states | NPHCDA ACSM monitoring register | project period | state, concern fields, activity |

## Domain 5 - Zero-dose modelling
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| `nigeria_ndhs_zero_dose_VERIFIED_long.csv` | State zero-dose percent by survey wave | NDHS | 2008-2024 | state, zone, year, zero_dose_pct, n_children_12_23m |
| `under_5_2024.csv`, `under_5_2025.csv` | Under-five population by state (cohort = under-five / 5) | National Population Commission | 2022 projection | zone, state, under-five |
| `administrative_lga_population.csv` | LGA population for within-state burden weighting | National Population Commission | 2022 projection | State, Name, Status, PopulationProjection2022-03-21 |
| `ndhs_antigens2024.csv` | NDHS antigen coverage by state (for admin-vs-survey checks) | NDHS | 2024 | state, antigen coverage |
| `lga_modeled_zero_dose.csv` | Modelled LGA zero-dose rate and children (output) | Derived (Domain 5) | 2026 | State, LGA, zero_dose_rate_pct, zero_dose_children |
| `lga_priority_ranking.csv` | Ranked LGA priority list with archetype, tier, intervention (output) | Derived (Domain 5) | 2026 | Burden rank, State, LGA, Archetype, Equity tier |

## Domain 6 - Polio surveillance
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| AFP national line list | Case-level acute flaccid paralysis surveillance (~103,000 records) | National AFP surveillance | 2015-2025 | date, location, classification |

## Domain 7 - Surveillance and mortality
| File | What it contains | Source | Reference date | Key columns |
| --- | --- | --- | --- | --- |
| IHME GBD cause fractions | State cause-of-death fractions (mortality proxy) | IHME Global Burden of Disease | 2023 | state, cause, fraction |

## Archetype and equity covariate source files (Archetypes_LGA / Domain 3)
| Covariate | Source | Reference date |
| --- | --- | --- |
| Education attainment | IHME admin-2 | 2000-2017 |
| Child growth (stunting, wasting, underweight) | IHME admin-2 | 2000-2017 |
| DPT1-3 coverage/dropout | IHME admin-2 (Africa) | 2000-2016 |
| MCV1 coverage | IHME admin-2 | 2000-2019 |
| Exclusive breastfeeding | IHME admin-2 (Africa) | 2000-2017 |
| ORS coverage | IHME admin-2 | project period |
| Antenatal care 4+ visits | NDHS (DHS Spatial Data Repository) | 2018 |
| Facility delivery | NDHS (DHS Spatial Data Repository) | 2018 |
| Improved water source | NDHS (DHS Spatial Data Repository) | 2018 |
| Relative wealth | Meta Relative Wealth Index | 2021 |
| Travel time to care (remoteness) | Weiss et al. (Malaria Atlas Project) | 2018/2020 |
| Conflict events and fatalities | ACLED (Armed Conflict Location and Event Data) | 2021-2024 |

## Geometry
| File | What it contains | Source |
| --- | --- | --- |
| GRID3 GeoJSON (admin-1 states, admin-2 LGAs) | NPHCDA vaccination boundaries for maps and spatial statistics | GRID3 |

---

**Notes**
- Every modelled figure is a model estimate; local-government values are calibrated, not directly observed.
- 730 of 774 local governments were modelled for zero-dose; 44 were excluded for no usable Penta1 record in 2021-2024.
- U-5 mortality and LLS 2018-2019 microdata were reviewed but excluded from the archetype covariates (health outcome / not LGA-representative).
