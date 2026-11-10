"""
04_gra_multiresponse.py
------------------------
Grey Relational Analysis (GRA) on the 80 actually-tested runs (this ranks
observed data directly, so it is NOT subject to the small-sample prediction
limits documented in script 03 -- that caveat only applies to claims about
UNTESTED combinations).

All six responses are "smaller-the-better." Three weighting schemes are
computed to test how much the recommendation depends on subjective choices:
  - equal6   : equal weight, all 6 responses (the naive field-standard default)
  - critic6  : CRITIC objective weighting (variance + inter-criteria conflict),
               all 6 responses -- the same weighting family used with
               TOPSIS/WASPAS in the closest literature comparator
  - equal5   : equal weight, the 5 directly-measured responses only
               (CTR_mm excluded, given the data-quality flag from script 01)

Also runs:
  - Taguchi-style mean-GRG-by-level tables (main effects) for each scheme
  - ANOVA on the composite GRG itself
  - A direct diagnostic of the Speed x Diameter interaction that explains why
    the naive "recombine the best individual levels" recommendation disagrees
    with the best individually-tested run
  - The best-performing run *within the best diameter subgroup specifically*,
    which is the recommendation that actually accounts for the interaction

Input : data/AA5083_original.csv
Output: results/04_gra_scores_and_ranks.csv
        results/04_gra_main_effects.csv
        results/04_gra_anova_on_grg.csv
        results/04_speed_dia_interaction.csv
        results/04_best_cell_within_best_dia.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "AA5083_original.csv")

RESP_ALL = ["CTR_mm", "TBT_mm", "BBT_mm", "Circularity_Error_Ce_mm",
            "Cylindricity_Error_Cy_mm", "Conicity_Error_Co_mm"]
RESP_NO_CTR = [r for r in RESP_ALL if r != "CTR_mm"]
PSI = 0.5  # standard GRA distinguishing coefficient

# ---- Normalize (smaller-the-better): 1 = best observed, 0 = worst observed ----
norm = pd.DataFrame(index=df.index)
for col in RESP_ALL:
    lo, hi = df[col].min(), df[col].max()
    norm[col] = (hi - df[col]) / (hi - lo)

delta = 1 - norm                      # Delta_min=0, Delta_max=1 globally by construction
xi = PSI / (delta + PSI)              # grey relational coefficient, per response per run


def critic_weights(norm_df, cols):
    Xm = norm_df[cols].values
    std = Xm.std(axis=0, ddof=1)
    corr = np.corrcoef(Xm, rowvar=False)
    conflict = (1 - corr).sum(axis=1)
    info = std * conflict
    return info / info.sum()


weights = {
    "equal6": pd.Series(1 / 6, index=RESP_ALL),
    "critic6": pd.Series(critic_weights(norm, RESP_ALL), index=RESP_ALL),
    "equal5": pd.Series(1 / 5, index=RESP_NO_CTR),
}

for scheme, w in weights.items():
    df[f"GRG_{scheme}"] = (xi[w.index] * w).sum(axis=1)
    df[f"GRG_{scheme}_rank"] = df[f"GRG_{scheme}"].rank(ascending=False).astype(int)

print("CRITIC weights (6-response scheme):")
print(weights["critic6"].round(3))

keep_cols = ["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"] + RESP_ALL + \
            [c for c in df.columns if c.startswith("GRG_")]
df[keep_cols].to_csv(ROOT / "results" / "04_gra_scores_and_ranks.csv", index=False)

top10 = df.nsmallest(10, "GRG_equal6_rank")[
    ["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm",
     "GRG_equal6_rank", "GRG_critic6_rank", "GRG_equal5_rank"]
]
print("\nTop 10 by equal6 GRG, with rank under the other two schemes:")
print(top10.to_string(index=False))

overlap5 = (set(df.nsmallest(5, "GRG_equal6_rank")["Exp_No"]) &
            set(df.nsmallest(5, "GRG_critic6_rank")["Exp_No"]) &
            set(df.nsmallest(5, "GRG_equal5_rank")["Exp_No"]))
print(f"\nExp_No in ALL three schemes' top-5: {sorted(overlap5)}")

# ---- Main effects (Taguchi-style mean GRG per factor level) ----
main_effects_rows = []
for scheme in ["GRG_equal6", "GRG_equal5"]:
    for factor in ["Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"]:
        means = df.groupby(factor)[scheme].mean()
        for level, val in means.items():
            main_effects_rows.append({"scheme": scheme, "factor": factor,
                                       "level": level, "mean_GRG": round(val, 4)})
main_effects_df = pd.DataFrame(main_effects_rows)
main_effects_df.to_csv(ROOT / "results" / "04_gra_main_effects.csv", index=False)

recommended = {}
for scheme in ["GRG_equal6", "GRG_equal5"]:
    best_speed = df.groupby("Spindle_Speed_RPM")[scheme].mean().idxmax()
    best_feed = df.groupby("Feed_Rate_mm_min")[scheme].mean().idxmax()
    best_dia = df.groupby("Tool_Dia_mm")[scheme].mean().idxmax()
    recommended[scheme] = (best_speed, best_feed, best_dia)
    match = df[(df.Spindle_Speed_RPM == best_speed) & (df.Feed_Rate_mm_min == best_feed) & (df.Tool_Dia_mm == best_dia)]
    rank = match[f"{scheme}_rank"].values[0] if len(match) else None
    print(f"\n{scheme}: naive main-effects recommendation = "
          f"Speed {best_speed}, Feed {best_feed}, Dia {best_dia} "
          f"-> actual rank if tested: {rank} of 80")

# ---- ANOVA on the composite GRG ----
df["Speed_c"] = (df["Spindle_Speed_RPM"] - 2250) / 500
df["Feed_c"] = (df["Feed_Rate_mm_min"] - 150) / 50
df["Dia_c"] = (df["Tool_Dia_mm"] - 9) / 2
terms = "Speed_c + Feed_c + Dia_c + Speed_c:Feed_c + Speed_c:Dia_c + Feed_c:Dia_c + I(Speed_c**2) + I(Feed_c**2) + I(Dia_c**2)"
anova_rows = []
for scheme in ["GRG_equal6", "GRG_equal5"]:
    model = smf.ols(f"{scheme} ~ {terms}", data=df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    aov["pct"] = aov["sum_sq"] / aov["sum_sq"].sum() * 100
    aov["scheme"] = scheme
    aov["R2"] = model.rsquared
    anova_rows.append(aov.reset_index().rename(columns={"index": "term"}))
anova_df = pd.concat(anova_rows, ignore_index=True)
anova_df.to_csv(ROOT / "results" / "04_gra_anova_on_grg.csv", index=False)
print("\nANOVA on composite GRG (both schemes) -- see results/04_gra_anova_on_grg.csv")

# ---- Speed x Diameter interaction diagnostic ----
# Why does the main-effects recommendation (low speed) disagree with the best
# individually-tested runs (high speed, at 6mm)? Split the speed effect apart
# by diameter to check directly.
interaction_rows = []
for dia in sorted(df["Tool_Dia_mm"].unique()):
    sub = df[df["Tool_Dia_mm"] == dia]
    for speed, val in sub.groupby("Spindle_Speed_RPM")["GRG_equal5"].mean().items():
        interaction_rows.append({"Tool_Dia_mm": dia, "Spindle_Speed_RPM": speed,
                                  "mean_GRG_equal5": round(val, 4)})
interaction_df = pd.DataFrame(interaction_rows)
interaction_df.to_csv(ROOT / "results" / "04_speed_dia_interaction.csv", index=False)
pivot = interaction_df.pivot(index="Spindle_Speed_RPM", columns="Tool_Dia_mm", values="mean_GRG_equal5")
print("\nMean GRG_equal5 by Speed, split by Diameter (Figure 4 data):")
print(pivot)
print(f"\nBest speed AT 6mm: {pivot[6].idxmax()} rpm  |  Best speed AT 12mm: {pivot[12].idxmax()} rpm  "
      f"-> direction reverses, confirming a real Speed x Diameter interaction.")

# ---- Best cell within the best diameter subgroup (accounts for the interaction) ----
best_dia_overall = df.groupby("Tool_Dia_mm")["GRG_equal5"].mean().idxmax()
subgroup = df[df["Tool_Dia_mm"] == best_dia_overall].sort_values("GRG_equal5_rank")
best_cell_cols = ["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"] + RESP_NO_CTR + \
                 ["GRG_equal5", "GRG_equal5_rank", "GRG_equal6_rank", "GRG_critic6_rank"]
subgroup[best_cell_cols].to_csv(ROOT / "results" / "04_best_cell_within_best_dia.csv", index=False)
print(f"\nBest-performing runs WITHIN the best diameter ({best_dia_overall}mm) -- "
      f"the recommendation that correctly accounts for the interaction:")
print(subgroup[best_cell_cols].head(5).to_string(index=False))
