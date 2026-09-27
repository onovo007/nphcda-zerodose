# NPHCDA Zero-Dose Predictive Modelling Platform - Handover Manifest

**Handover to:** National Primary Health Care Development Agency (NPHCDA)
**From:** CIDRE and Quantium Insights LLC consortium, in technical support of NPHCDA; funders and reviewers GAVI and UNICEF.
**Date:** 2026-09-27

This document records exactly what is being handed over.

---

## Items handed over

### 1. The web application (perpetual access)
- Live, no-code platform: **https://amobionovo-nphcda-zerodose.hf.space/**
- Runs all domains on bundled sample data or on uploaded data; NPHCDA retains perpetual access.
- Access is email-controlled; authorized NPHCDA users are added to the allow-list.
- Portable via its Docker container for redeployment on Government infrastructure if desired.

### 2. The source code repository (GitHub)
- **github.com/onovo007/nphcda-zerodose** - the full application, deployment configuration (Docker, Render), and a comprehensive README / user guide.

### 3. The datasets (Google Drive - data of record)
- Drive folder organized by domain (Domains 1-7, Archetypes, Geometry) with a dataset dictionary (source, date, key columns per file).
- Large and harmonized datasets are stored here, since GitHub does not host large files.

### 4. Final technical report
- `Final_NPHCDA_Consolidated_Report_UPDATED_08_09_2026.docx` (repo `handover/`).

### 5. Final presentation deck
- `NPHCDA_ZeroDose_RI_Team_Presentation_9.27.2026.pptx` (repo `handover/`).

### 6. Analytical notebooks (Google Colab - reproducible)
- Full set in the repo `notebooks/`: Domains 1-7, the LGA archetype notebook, and cross-checks / triangulation. Each re-runs on updated data.

### 7. Documentation and guides
- README - using the results, navigating the web app, running the notebooks, and the dataset structure required to upload and re-run successfully.
- `handover/DATA_DICTIONARY.md` - the dataset dictionary.
- `handover/HANDOVER_MANIFEST.md` - this document.

### 8. Methods and validation
- One method per domain; external validation against NmDHS 2025-26 (state rho = 0.88) and the IHME DTP1 surface (LGA rho = 0.60); documented 730-modelled / 44-excluded data-quality accounting.

---

## What stays with the consortium, and why

The **hosting and ongoing maintenance of the live web application** continue with the consortium, while NPHCDA retains full access. This is because a modelling platform is a **living system**: it needs periodic model re-training, dependency and security updates, and paid hosting/compute. The consortium absorbing this during the pilot de-risks it for NPHCDA until a sustainability plan (hosting budget and Digital Innovation Hub capacity) is in place.

**Nothing is locked away.** The intellectual output - source code, notebooks, datasets, report, deck, and this dictionary - is fully handed over and reproducible. Should the Government wish to take full ownership, the application is portable via its Docker container and open code, and can be redeployed on Government infrastructure or integrated into an existing Government platform, with a phased transition and capacity-building for the Digital Innovation Hub team.

---

## Access and security notes
- No credentials (API keys or tokens) are included in this package. Access is granted by adding authorized emails; any keys remain in the hosting environment.
- The GitHub repository and the Google Drive folder mirror each other by design (same domain structure, same dictionary).
