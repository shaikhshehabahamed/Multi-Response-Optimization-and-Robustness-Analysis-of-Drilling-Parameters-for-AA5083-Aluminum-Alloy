"""
05_topsis_crosscheck.py
-------------------------
GRA (script 04) is one multi-criteria method among several used in this
literature -- TOPSIS is the other common one (and is the one paired with
CRITIC weighting in the closest published AA5083 comparator). If the two
methods agree on which runs are genuinely good, that is much stronger
evidence than either method alone. TOPSIS uses vector normalization and a
distance-to-ideal-point logic, which is mathematically unrelated to GRA's
min-max + grey-coefficient logic -- so agreement between them is a real,
independent check, not a restatement of the same computation.

Input : data/AA5083_original.csv
Output: results/05_topsis_scores_and_ranks.csv
        results/05_method_agreement.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "AA5083_original.csv")

RESP_ALL = ["CTR_mm", "TBT_mm", "BBT_mm", "Circularity_Error_Ce_mm",
            "Cylindricity_Error_Cy_mm", "Conicity_Error_Co_mm"]
RESP_NO_CTR = [r for r in RESP_ALL if r != "CTR_mm"]


def critic_weights(df_, cols):
    # Uses the same min-max-normalized-to-"higher-is-better" matrix as script 04
    # so the CRITIC weights themselves are identical across both methods.
    norm = pd.DataFrame({c: (df_[c].max() - df_[c]) / (df_[c].max() - df_[c].min()) for c in cols})
    Xm = norm[cols].values
    std = Xm.std(axis=0, ddof=1)
    corr = np.corrcoef(Xm, rowvar=False)
    conflict = (1 - corr).sum(axis=1)
    info = std * conflict
    return info / info.sum()


weight_schemes = {
    "equal6": pd.Series(1 / 6, index=RESP_ALL),
    "critic6": pd.Series(critic_weights(df, RESP_ALL), index=RESP_ALL),
    "equal5": pd.Series(1 / 5, index=RESP_NO_CTR),
}


def topsis(df_, cols, weights):
    X = df_[cols].values.astype(float)
    # Vector normalization (the TOPSIS convention -- deliberately different
    # from GRA's min-max normalization, so this is a genuinely separate check)
    r = X / np.sqrt((X ** 2).sum(axis=0))
    v = r * weights.values
    # All criteria are smaller-the-better -> ideal best is the column MIN, ideal worst is the column MAX
    ideal_best = v.min(axis=0)
    ideal_worst = v.max(axis=0)
    s_best = np.sqrt(((v - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((v - ideal_worst) ** 2).sum(axis=1))
    closeness = s_worst / (s_best + s_worst)
    return closeness


for scheme, w in weight_schemes.items():
    cols = list(w.index)
    df[f"TOPSIS_{scheme}"] = topsis(df, cols, w)
    df[f"TOPSIS_{scheme}_rank"] = df[f"TOPSIS_{scheme}"].rank(ascending=False).astype(int)

keep_cols = ["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"] + \
            [c for c in df.columns if c.startswith("TOPSIS_")]
df[keep_cols].to_csv(ROOT / "results" / "05_topsis_scores_and_ranks.csv", index=False)

print("Top 10 by TOPSIS (equal5, no CTR):")
print(df.nsmallest(10, "TOPSIS_equal5_rank")[
    ["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm", "TOPSIS_equal5_rank"]
].to_string(index=False))

# ---- Agreement between GRA and TOPSIS ----
gra = pd.read_csv(ROOT / "results" / "04_gra_scores_and_ranks.csv")
merged = df.merge(gra[["Exp_No", "GRG_equal6_rank", "GRG_critic6_rank", "GRG_equal5_rank"]], on="Exp_No")

agreement_rows = []
for scheme in ["equal6", "critic6", "equal5"]:
    gra_col, topsis_col = f"GRG_{scheme}_rank", f"TOPSIS_{scheme}_rank"
    spearman_r = merged[gra_col].corr(merged[topsis_col], method="spearman")
    top5_gra = set(merged.nsmallest(5, gra_col)["Exp_No"])
    top5_topsis = set(merged.nsmallest(5, topsis_col)["Exp_No"])
    agreement_rows.append({
        "scheme": scheme,
        "spearman_rank_correlation": round(spearman_r, 3),
        "top5_overlap_count": len(top5_gra & top5_topsis),
        "top5_overlap_ExpNo": sorted(top5_gra & top5_topsis),
    })
agreement_df = pd.DataFrame(agreement_rows)
agreement_df.to_csv(ROOT / "results" / "05_method_agreement.csv", index=False)
print("\nGRA vs TOPSIS agreement by weighting scheme:")
print(agreement_df.to_string(index=False))
