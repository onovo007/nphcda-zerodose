"""Bundle the validation and contextual-profile evidence for the web app (10.10.2026).

1. Replaces the archetype labels in the bundled data with descriptive profile labels that name only what the
   input indicators measure (no indicator measures nomadism, migration, riverine settlement or slums).
2. Writes small text files to data/sample/two_methods/evidence/ from the verified refinement outputs:
   external validation, calibration, temporal holdouts, burden capture, per-LGA barriers and candidate
   packages, within-state variance, package-assignment agreement and package stability.

Run from the repository root:  python domain5_update_2026_10/code/build_app_evidence_10_10.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "sample"
TM = DATA / "two_methods"
OUT = TM / "evidence"
REF = REPO.parent / "Zero-Dose Predictive Modeling" / "WorldBank_Conference_2026_Refinement"
AF, BT = REF / "01_Study_A" / "figures", REF / "02_Study_B" / "tables"

PLAB = {1: "Deprived northern rural", 2: "Low maternal care, high conflict exposure", 3: "Geographically isolated",
        4: "Near-average", 5: "Relatively advantaged"}
OLD = {1: ["Remote Rural / Hard-to-Reach"], 2: ["Conflict-Affected / Nomadic"],
       3: ["Riverine / Geographically Isolated"],
       4: ["Peri-urban / Migrant Dense (transitional)", "Peri-urban / Migrant Dense"],
       5: ["Urban Slums (better-off urban core)", "Urban Slums / Better-off Core", "Urban Slums (better-off core)"]}
COMP = {"OUTREACH": "Outreach and mobile sessions", "MCH_INTEGRATION": "Maternal and child health integration",
        "NUTRITION_INTEGRATION": "Nutrition co-delivery", "DEMAND_COST": "Demand generation and indirect-cost support",
        "SECURITY_ACCESS": "Security-sensitive access", "DEFAULTER_TRACING": "Defaulter tracing and reminder-recall"}
AMB = {"clear (stable and well inside its profile)": "Clear", "intermediate": "Intermediate",
       "mixed (unstable or between two profiles)": "Mixed"}
DOMS = ["Socioeconomic", "Maternal care", "Nutrition", "Access", "Insecurity", "Programme (dropout)"]


def relabel(path):
    """Replace old archetype labels in every text column (values only, so CSV quoting stays valid)."""
    df = pd.read_csv(path)
    m = {o: PLAB[k] for k, olds in OLD.items() for o in olds}
    changed = 0
    for c in df.columns:
        if df[c].dtype == object:
            new = df[c].replace(m)
            changed += int((new != df[c]).sum())
            df[c] = new
    if changed:
        import csv
        with open(path, encoding="utf-8", newline="") as fh:
            df.columns = next(csv.reader(fh))  # keep the original header, including any repeated name
        df.to_csv(path, index=False)
        print("relabelled", path.relative_to(REPO), changed, "cells")


def comps(code_str):
    if not isinstance(code_str, str) or not code_str.strip():
        return "Core services only"
    return "; ".join(COMP[c.strip()] for c in code_str.split("+") if c.strip() in COMP)


def main():
    for f in [DATA / "lga_archetype_master.csv", DATA / "lga_archetype_summary.csv", DATA / "lga_priority_ranking.csv",
              TM / "archetype_both_methods.csv", TM / "lga_archetype_covariates_774.csv", TM / "lga_both_methods_774.csv"]:
        relabel(f)
    # profile packages and evidence text: the same dictionaries the app uses (lga_priority.py)
    import ast
    tree = ast.parse((REPO / "lga_priority.py").read_text(encoding="utf-8"))
    lit = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body
           if isinstance(t, ast.Assign) and getattr(t.targets[0], "id", "") in ("BUNDLE", "EVIDENCE", "ARCH_DEF")}
    sm = pd.read_csv(DATA / "lga_archetype_summary.csv")
    lab = sm["Type of Cluster (setting typology)"]
    sm["Targeted lever / intervention"] = lab.map(lit["BUNDLE"]).str.replace(" + ", "; ", regex=False)
    sm["Evidence base and allocation rationale (citations)"] = lab.map(lit["EVIDENCE"])
    sm["Defining determinants (data-driven signature)"] = sm["Defining determinants (data-driven signature)"] \
        .str.replace(" Well-served overall, with residual gaps in informal settlements.", " Moderate conflict events.", regex=False) \
        .str.replace("Moderate on all dimensions", "Near the national average on every dimension", regex=False)
    sm.to_csv(DATA / "lga_archetype_summary.csv", index=False)
    pr = pd.read_csv(DATA / "lga_priority_ranking.csv")
    pr["Intervention bundle"] = pr["Archetype"].map(lit["BUNDLE"])
    pr["Evidence base (method and citation)"] = pr["Archetype"].map(lit["EVIDENCE"])
    pr.to_csv(DATA / "lga_priority_ranking.csv", index=False)

    OUT.mkdir(parents=True, exist_ok=True)

    # Study A: external validation and holdouts
    v = pd.read_csv(AF / "FA2_comparative_performance_data.csv")
    v["approach"] = v.model.map({"A0": "Latest survey carried forward", "A1": "Routine data only",
                                 "A2": "Fay-Herriot area-level model", "A3": "Method 1 (hierarchical state model)",
                                 "A4": "Method 2 (SAE)", "A5": "Gradient boosting"})
    v["check"] = v.validation.where(v.validation != "holdout", "Predict " + v.target.astype("Int64").astype(str))
    v[["check", "approach", "MAE", "bias", "spearman_rho", "spearman_ci_lo", "spearman_ci_hi", "CRPS", "dMAE_vs_A4",
       "dMAE_lo", "dMAE_hi"]].round(3).rename(columns={"dMAE_vs_A4": "dMAE_vs_SAE"}).to_csv(OUT / "validation_comparators.csv", index=False)
    pd.read_csv(AF / "FA4_calibration_data.csv")[["model", "interval", "nominal", "empirical_coverage", "covered", "n",
                                                  "mean_width", "CRPS"]].round(4).to_csv(OUT / "calibration.csv", index=False)
    t = pd.read_csv(AF / "FA6_temporal_extrapolation_data.csv")
    t.round(2).to_csv(OUT / "temporal_holdout.csv", index=False)
    c = pd.read_csv(AF / "FA5_burden_capture_curves_data.csv")
    c[["K", "rule", "capture_SAE_mean", "capture_SAE_lo95", "capture_SAE_hi95", "capture_state_model_mean"]].round(2) \
        .to_csv(OUT / "capture_curves.csv", index=False)

    # Study B: per-LGA barriers and candidate packages
    s = pd.read_csv(BT / "B_R2_soft_membership_per_lga.csv")
    p = pd.read_csv(BT / "B_R4_lga_packages.csv")
    d = s[["lga_uid", "state", "lga", "primary_archetype", "secondary_profile", "ambiguity", "dominant_barrier_1",
           "dominant_barrier_1_z", "n_domains_above_0_5"] + [f"domain_{x}" for x in DOMS]].merge(
        p[["lga_uid", "barriers_flagged", "n_barriers", "lga_barrier_package"]], on="lga_uid")
    d["profile"] = d.primary_archetype.map(PLAB)
    d["secondary_profile"] = d.secondary_profile.map(PLAB)
    d["membership"] = d.ambiguity.map(AMB)
    d["dominant_barrier"] = d.dominant_barrier_1.where(d.dominant_barrier_1_z >= 0.5, "None above +0.5 SD")
    d["barriers_flagged"] = d.barriers_flagged.fillna("None").str.replace(";", ",")
    d["candidate_components"] = d.lga_barrier_package.map(comps)
    d = d.rename(columns={"primary_archetype": "profile_id", **{f"domain_{x}": f"score_{x}" for x in DOMS}})
    keep = ["lga_uid", "state", "lga", "profile_id", "profile", "membership", "secondary_profile", "dominant_barrier",
            "barriers_flagged", "n_barriers", "candidate_components"] + [f"score_{x}" for x in DOMS]
    d[keep].round(3).to_csv(OUT / "lga_barriers_774.csv", index=False)

    pd.read_csv(BT / "B_R3_within_state_variance.csv").round(4).to_csv(OUT / "within_state_variance.csv", index=False)
    r = pd.read_csv(BT / "B_R4_assignment_rule_comparison.csv")
    r = r[~r.assignment_rule.str.contains("own barrier")].copy()
    r["assignment_rule"] = r.assignment_rule.replace({"By archetype (centroid rule)": "By contextual profile",
                                                      "Hybrid archetype (primary + secondary)": "Profile plus secondary profile"})
    r[["assignment_rule", "mean_jaccard_vs_lga", "burden_weighted_jaccard"]].round(4).to_csv(OUT / "assignment_rules.csv", index=False)
    st = pd.read_csv(BT / "B_R4_decision_stability.csv")
    st[["alternative", "same_primary_archetype", "same_core_components_jaccard_ge_0_67", "same_core_burden_weighted"]] \
        .rename(columns={"same_primary_archetype": "same_primary_profile", "same_core_components_jaccard_ge_0_67":
                         "same_core_lgas", "same_core_burden_weighted": "same_core_children"}).round(3) \
        .to_csv(OUT / "package_stability.csv", index=False)
    m = pd.read_csv(BT / "B_R4_barrier_intervention_matrix.csv")
    m[["barrier", "indicator", "rule", "components", "evidence", "local_validation"]].to_csv(OUT / "barrier_matrix.csv", index=False)
    print("evidence files:", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
