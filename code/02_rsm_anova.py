"""
02_rsm_anova.py
----------------
Fits a full second-order Response Surface Methodology (RSM) model
(linear + two-way interactions + pure quadratic terms) for each of the six
responses, using coded factors (centered, unit-step-scaled) to stabilize the
fit. Runs Type-II ANOVA on each model to get significance (p-value) and
percent-contribution-to-variance for every term, then rolls the nine terms
up into six interpretable groups: Speed, Feed, Diameter, 2-way Interactions,
Quadratic curvature, and Residual (unexplained).

Input : data/AA5083_original.csv
Output: results/02_anova_full_terms.csv      (every model term, every response)
        results/02_anova_grouped_pct.csv     (grouped % contribution, for Figure 1)
        results/02_anova_significant_terms.csv (only terms with p < 0.05)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "AA5083_original.csv")

# Coded factors: centered on the design midpoint, scaled to the level step size.
# This is standard RSM practice and avoids the numerical instability that
# raw (e.g. 1500-3000) units cause in quadratic/interaction terms.
df["Speed"] = (df["Spindle_Speed_RPM"] - 2250) / 500   # -1.5 .. 1.5
df["Feed"]  = (df["Feed_Rate_mm_min"]  - 150)  / 50     # -2   .. 2
df["Dia"]   = (df["Tool_Dia_mm"]       - 9)    / 2      # -1.5 .. 1.5

RESPONSES = ["CTR_mm", "TBT_mm", "BBT_mm", "Circularity_Error_Ce_mm",
             "Cylindricity_Error_Cy_mm", "Conicity_Error_Co_mm"]
FORMULA_TERMS = ("Speed + Feed + Dia + Speed:Feed + Speed:Dia + Feed:Dia "
                  "+ I(Speed**2) + I(Feed**2) + I(Dia**2)")

all_terms = []
grouped_rows = []
sig_rows = []

for resp in RESPONSES:
    model = smf.ols(f"{resp} ~ {FORMULA_TERMS}", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    aov["pct_contribution"] = aov["sum_sq"] / ss_total * 100
    aov["response"] = resp
    aov["R2"] = model.rsquared
    aov["Adj_R2"] = model.rsquared_adj
    aov_reset = aov.reset_index().rename(columns={"index": "term"})
    all_terms.append(aov_reset)

    pct = aov["pct_contribution"]
    grouped_rows.append({
        "Response": resp,
        "R2": round(model.rsquared, 3),
        "Speed_pct": round(pct["Speed"], 2),
        "Feed_pct": round(pct["Feed"], 2),
        "Dia_pct": round(pct["Dia"], 2),
        "Interactions_pct": round(pct[["Speed:Feed", "Speed:Dia", "Feed:Dia"]].sum(), 2),
        "Quadratic_pct": round(pct[["I(Speed ** 2)", "I(Feed ** 2)", "I(Dia ** 2)"]].sum(), 2),
        "Residual_pct": round(pct["Residual"], 2),
    })

    sig = aov_reset[(aov_reset["PR(>F)"] < 0.05) & (aov_reset["term"] != "Residual")]
    for _, row in sig.iterrows():
        sig_rows.append({"response": resp, "term": row["term"],
                          "p_value": round(row["PR(>F)"], 4),
                          "pct_contribution": round(row["pct_contribution"], 2)})

full_terms_df = pd.concat(all_terms, ignore_index=True)
full_terms_df.to_csv(ROOT / "results" / "02_anova_full_terms.csv", index=False)

grouped_df = pd.DataFrame(grouped_rows).set_index("Response")
grouped_df.to_csv(ROOT / "results" / "02_anova_grouped_pct.csv")

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(ROOT / "results" / "02_anova_significant_terms.csv", index=False)

print("Grouped % contribution by response (Figure 1 data):")
print(grouped_df)
print("\nStatistically significant terms (p < 0.05):")
print(sig_df.to_string(index=False))
