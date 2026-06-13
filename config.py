"""
Central configuration for the NPHCDA Zero-Dose real-time modelling app.

Holds file paths, dataset schemas (for upload validation), model constants lifted
verbatim from the Domain 1/2/5 notebooks, and the colour palettes used across every
figure so the live charts match the report and deck exactly.

House style: no em dashes, no en dashes, no section signs anywhere in user-facing text.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "sample"
GEO_DIR = DATA_DIR / "geo"
ASSETS_DIR = APP_DIR / "assets"

# Bundled project sample inputs (so the demo runs with zero upload)
SAMPLE_FILES = {
    "dhis2": DATA_DIR / "dhis2_data_all_states.csv",
    "ndhs_long": DATA_DIR / "nigeria_ndhs_zero_dose_VERIFIED_long.csv",
    "model_dataset": DATA_DIR / "nigeria_zero_dose_model_dataset.csv",
    "under5": DATA_DIR / "under_5_2024.csv",
    "lga_population": DATA_DIR / "administrative_lga_population.csv",
    "ndhs_antigens": DATA_DIR / "ndhs_antigens2024.csv",
    "live_births": DATA_DIR / "dhis2_data_live_births.csv",
}

# Map each tracer antigen to its survey-coverage column in ndhs_antigens2024.csv (the central
# source for the admin-vs-survey reference lines).
SURVEY_ANTIGEN_COLS = {
    "BCG": "BCG vaccination received",
    "Penta1": "Pentavalent 1 vaccination received",
    "Penta3": "Pentavalent 3 vaccination received",
    "Measles1": "Measles vaccination received",
}
GEO_STATES = GEO_DIR / "nga_states.geojson"
GEO_LGAS = GEO_DIR / "nga_lgas.geojson"

# Alternative under-five population vintage for the Domain 5 population sensitivity view.
UNDER5_2025_PATH = DATA_DIR / "under_5_2025.csv"

# --------------------------------------------------------------------------------------
# Model constants (verbatim from the notebooks)
# --------------------------------------------------------------------------------------
# Domain 5 - Bayesian hierarchical Beta (PyMC). Notebook used 3000/3000; the live default
# is tuned down so a click returns in about a minute. The full run is exposed as a toggle.
MCMC_DRAWS_LIVE = 1000
MCMC_TUNE_LIVE = 1000
MCMC_DRAWS_FULL = 3000
MCMC_TUNE_FULL = 3000
MCMC_CHAINS = 2
TARGET_ACCEPT = 0.92
FORECAST_YEARS = [2026, 2027, 2028]
LGA_KNN = 5
NDHS_YEARS = [2008, 2013, 2018, 2024]
YEAR_CENTER = 2024  # year_mean used to standardise the year covariate

# Domain 1 / 2 - Prophet
FORECAST_MONTHS = 18
THRESHOLD_PCT = 80.0
AT_RISK_WINDOW_MONTHS = (6, 12)
PI_80_FACTOR = 0.654  # 80% PI half-width = z(0.90)/z(0.975) = 1.2816/1.9600 of the 95% PI half-width

# NDHS national survey coverage (% of children 12-23 months), used as the admin-vs-survey
# reference line in Coverage Forecasting (WHO admin-coverage mode). Penta1 is recomputed live from
# the uploaded zero-dose data (100 - national zero-dose); the other antigens use these published
# NDHS 2023-24 national figures. EDIT to match the exact values in your NDHS report.
SURVEY_COVERAGE = {"BCG": 70.0, "Penta1": 62.0, "Penta3": 57.0, "Measles1": 59.0}
SURVEY_COVERAGE_SOURCE = "NDHS 2023-24 (national, 12-23 months)"

# Data dictionary / provenance shown on the Data and Quality page (source, vintage, what it holds).
PROVENANCE = [
    ("DHIS2 routine immunization", "NPHCDA DHIS2 export", "2021-2025 monthly", "Antigen dose counts by LGA-month"),
    ("NDHS zero-dose (state-year)", "NDHS, verified from report PDFs", "2008-2024 rounds", "Survey zero-dose rate by state"),
    ("NDHS antigen coverage", "NDHS 2023-24", "2024", "State survey coverage per antigen"),
    ("Zero-dose model dataset", "NDHS / NPC / composite", "to 2024", "State equity & socioeconomic covariates"),
    ("Under-five population", "City Population (NPC 2022 projection)", "2024", "Under-5 by state; birth cohort = /5"),
    ("DHIS2 live births", "NPHCDA DHIS2 export", "2021-2025 monthly", "Facility-reported live births by LGA"),
    ("LGA population", "City Population / NPC 2022 projection", "2022", "Population by LGA for burden weighting"),
]

ANTIGEN_TS = {
    "BCG": "bcg_count",
    "Penta1": "penta_1_count",
    "Penta3": "penta_3_count",
    "Measles1": "measles_1_count",
}
COUNT_COLS = [
    "bcg_count", "penta_1_count", "pent_2_count", "penta_3_count",
    "measles_1_count", "measles_2_count",
]

# Domain 2 - LASSO driver features (verbatim from the D1_D2 notebook, cell 29)
LASSO_FEATURES = [
    "anc_4plus", "delivered_in_hf", "pct_c12to23_vax_card_seen",
    "pct_cu5_birth_registered", "pct_using_improved_water_source",
    "pct_cu5_slept_under_itn", "pct_problem_accessing_hfdistance",
    "pct_cu5_stunted", "pct_women_no_education", "pct_lowest_wealth_quintile",
    "total_fertility_rate", "pct_urban", "pct_muslim",
    "pct_women_with_mobile_phone", "pct_media_at_least_once_week",
    "pct_women_curr_employed", "pct_women_say_wife_beating_justified",
]
DROPOUT_TARGETS = {
    "dropout_p1p3": "Penta1 to Penta3",
    "dropout_p1m1": "Penta1 to Measles1",
    "dropout_m1m2": "Measles1 to Measles2",
}

# --------------------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------------------
# Report / deck palette
NAVY = "#1F3B57"
STEEL = "#2E6E8E"
ACCENT = "#C0392B"
GOLD = "#C8902A"
MUTE = "#6B7A88"
LIGHT = "#EAF1F5"
INK = "#1A1A1A"
PAPER = "#FFFFFF"
PANEL = "#F8F9FA"

# NPHCDA brand green (from the agency logo) - used as a secondary accent that blends with navy.
NPHCDA_GREEN = "#1C7A3D"
NPHCDA_GREEN_DK = "#0F5226"

# Optional branding images (drop these into assets/; the app falls back gracefully if absent).
LOGO_PATH = ASSETS_DIR / "nphcda_logo.png"
HERO_PATH = ASSETS_DIR / "hero.jpg"
FLAG_PATH = ASSETS_DIR / "flag.png"
LOGIN_BANNER_PATH = ASSETS_DIR / "_banner_login.jpg"

ANTIGEN_PAL = {
    "BCG": "#1565C0",
    "Penta1": "#2E7D32",
    "Penta3": "#E65100",
    "Measles1": "#880E4F",
}
# Okabe-Ito colour-blind-safe categorical palette (north = warm, south = cool, all distinguishable
# under deuteranopia/protanopia).
ZONE_COLORS = {
    "North West": "#D55E00",
    "North East": "#E69F00",
    "North Central": "#CC79A7",
    "South West": "#0072B2",
    "South East": "#009E73",
    "South South": "#56B4E9",
}
TIER_COLORS = {
    "Tier 1: Critical": "#D73027",
    "Tier 2: High": "#FC8D59",
    "Tier 3: Moderate": "#FEE090",
    "Tier 4: Lower": "#91BFDB",
}
# Short tier labels used in the LGA table (Tier 1..Tier 4)
TIER_COLORS_SHORT = {
    "Tier 1": "#D73027",
    "Tier 2": "#FC8D59",
    "Tier 3": "#FEE090",
    "Tier 4": "#91BFDB",
}
HOTSPOT_COLORS = {
    "Hot Spot (p<0.01)": "#D73027",
    "Hot Spot (p<0.05)": "#FC8D59",
    "Hot Spot (p<0.10)": "#FEE090",
    "Not Significant": "#CCCCCC",
    "Cold Spot (p<0.10)": "#ABD9E9",
    "Cold Spot (p<0.05)": "#74ADD1",
    "Cold Spot (p<0.01)": "#4575B4",
}
DROPOUT_COLORS = {
    "dropout_p1p3": "#E65100",
    "dropout_p1m1": "#880E4F",
    "dropout_m1m2": "#4A148C",
}
YEAR_COLORS = {
    2024: "#4C72B0",
    2026: "#E67E50",
    2027: "#5BA86A",
    2028: "#D7574E",
}
ZONE_ORDER = [
    "North West", "North East", "North Central",
    "South West", "South East", "South South",
]

# Sans-serif throughout (headings use the same family, heavier weight).
FONT_HEAD = "'IBM Plex Sans', 'Segoe UI', Arial, sans-serif"
FONT_BODY = "'IBM Plex Sans', 'Segoe UI', Arial, sans-serif"

# --------------------------------------------------------------------------------------
# Upload schemas (required columns for validation)
# --------------------------------------------------------------------------------------
SCHEMAS = {
    "dhis2": {
        "label": "DHIS2 routine immunization export",
        "required": ["zone", "state", "lga", "period", "penta_1_count", "penta_3_count"],
        "recommended": ["bcg_count", "measles_1_count", "measles_2_count", "pent_2_count"],
        "note": "LGA-month dose counts. 'period' like Jan-21. Feeds Domains 1, 2 and 5.",
    },
    "ndhs_long": {
        "label": "NDHS zero-dose longitudinal (state-year)",
        "required": ["state", "zone", "year", "zero_dose_pct", "n_children_12_23m"],
        "recommended": ["dtp1_coverage"],
        "note": "Survey-anchored state zero-dose rates by year. Feeds the Domain 5 Bayesian model.",
    },
    "model_dataset": {
        "label": "Zero-dose model dataset (equity covariates)",
        "required": ["state_name", "zone_name"],
        "recommended": LASSO_FEATURES,
        "note": "State equity and socioeconomic covariates. Feeds the Domain 2 LASSO drivers.",
    },
    "under5": {
        "label": "Under-five population (2024)",
        "required": [],
        "recommended": [],
        "note": "Zone, State, Under 5 columns (header on row 2). Sets the state 12-23m cohort.",
    },
    "lga_population": {
        "label": "Administrative LGA population (NPC 2022)",
        "required": ["State", "Name", "Status", "PopulationProjection2022-03-21"],
        "recommended": [],
        "note": "LGA population used to distribute the state cohort across LGAs (Domain 5).",
    },
    "ndhs_antigens": {
        "label": "NDHS antigen coverage (2024, by state)",
        "required": ["State"],
        "recommended": ["BCG vaccination received", "Pentavalent 1 vaccination received",
                        "Pentavalent 3 vaccination received", "Measles vaccination received"],
        "note": "State NDHS 2024 survey coverage per antigen. Sets the admin-vs-survey reference lines.",
    },
    "live_births": {
        "label": "DHIS2 live births (monthly, by LGA)",
        "required": ["state", "lga", "period", "live_births_count"],
        "recommended": ["zone"],
        "note": "Monthly DHIS2-reported live births. Optional eligible-infant denominator for coverage.",
    },
}

# Validation ground-truth (for the optional self-check banner)
GROUND_TRUTH = {
    "national_lga_burden": 2_085_872,
    "lga_count": 730,
    "sokoto_2026": 71.9,
    "kano_burden_k": 195,
    "penta3_min": 87.5,
    "pareto_top20_pct": 62,
}
