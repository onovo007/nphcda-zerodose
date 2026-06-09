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

# Bundled sample inputs (the canonical raw sources, so the demo runs with zero upload)
SAMPLE_FILES = {
    "dhis2": DATA_DIR / "dhis2_data_all_states.csv",
    "ndhs_long": DATA_DIR / "nigeria_ndhs_zero_dose_VERIFIED_long.csv",
    "model_dataset": DATA_DIR / "nigeria_zero_dose_model_dataset.csv",
    "under5": DATA_DIR / "under_5_2024.csv",
    "lga_population": DATA_DIR / "administrative_lga_population.csv",
}
GEO_STATES = GEO_DIR / "nga_states.geojson"
GEO_LGAS = GEO_DIR / "nga_lgas.geojson"

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
PI_80_FACTOR = 0.53  # 80% PI derived from the 95% PI in the notebook figures

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

ANTIGEN_PAL = {
    "BCG": "#1565C0",
    "Penta1": "#2E7D32",
    "Penta3": "#E65100",
    "Measles1": "#880E4F",
}
ZONE_COLORS = {
    "North West": "#D73027",
    "North East": "#FC8D59",
    "North Central": "#FDAE61",
    "South West": "#1A9850",
    "South East": "#91CF60",
    "South South": "#66C2A5",
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

FONT_HEAD = "'IBM Plex Serif', Georgia, 'Times New Roman', serif"
FONT_BODY = "'IBM Plex Sans', 'Segoe UI', Calibri, sans-serif"

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
