"""
Name-normalization and small numeric helpers, lifted verbatim from the notebooks so the
joins across DHIS2, NDHS, the LGA population file and the GRID3 geometry behave identically.
"""
from __future__ import annotations

import re

import numpy as np


def normalise_name(s) -> str:
    """Standardise a state/LGA name for joining (D5 cell 12)."""
    return str(s).strip().title().replace("-", " ").replace("_", " ")


def clean_lga_name(s) -> str:
    """Remove the 2-letter prefix code and 'Local Government Area' suffix (D5 cell 12)."""
    s = str(s).strip()
    s = re.sub(r"^[a-z]{2}\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*local\s+government\s+area\s*$", "", s, flags=re.IGNORECASE)
    return s.strip().title()


def nstate(s) -> str:
    """Normalised state key used in the population matcher and GeoJSON (D5 cell 45)."""
    s = str(s).lower().strip()
    if "federal capital" in s or s in ("fct", "fct, abuja"):
        return "fct"
    return re.sub(r"\(.*?\)", "", s).strip()


def nlga(s) -> str:
    """Normalised LGA key (D5 cell 45)."""
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(s).lower().strip())
    return re.sub(r"\s+", " ", s.replace("'", "").replace("/", " ")
                  .replace("-", " ").replace(".", "")).strip()


def tok(s) -> str:
    """Token-sorted key for fuzzy LGA matching (D5 cell 45)."""
    return " ".join(sorted(nlga(s).split()))


# Explicit LGA aliases and the six FCT LGAs mislabelled under Enugu in the population file
LGA_ALIAS = {("kebbi", "arewa"): "arewa dandi", ("imo", "ezinihitte mbaise"): "ezinihitte"}
FCT6 = {"Abaji", "Abuja Municipal Area Council", "Bwari", "Gwagwalada", "Kuje", "Kwali"}


def hotspot_class(z, p) -> str:
    """Map a Gi* z-score and permutation p-value to a labelled category (D5 cell 12)."""
    if p <= 0.01 and z > 0:
        return "Hot Spot (p<0.01)"
    if p <= 0.05 and z > 0:
        return "Hot Spot (p<0.05)"
    if p <= 0.10 and z > 0:
        return "Hot Spot (p<0.10)"
    if p <= 0.01 and z < 0:
        return "Cold Spot (p<0.01)"
    if p <= 0.05 and z < 0:
        return "Cold Spot (p<0.05)"
    if p <= 0.10 and z < 0:
        return "Cold Spot (p<0.10)"
    return "Not Significant"


def minmax_scale(x):
    """Scale a pandas Series to 0-100 (D5 cell 12)."""
    r = x.max() - x.min()
    return (x - x.min()) / r * 100 if r > 0 else x * 0 + 50
