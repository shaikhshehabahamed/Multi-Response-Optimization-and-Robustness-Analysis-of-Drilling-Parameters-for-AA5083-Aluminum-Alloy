"""
06_robustness_checks.py
-------------------------
Two further robustness checks on the Phase 2 ranking, beyond comparing
weighting schemes and comparing GRA against TOPSIS:

1. Monte Carlo noise-perturbation stability. Script 03 showed that most
   responses' variation is dominated by noise rather than by the three DOE
   factors. That noise was actually present in every run -- it just wasn't
   attributed to anything. This check asks: if you reran these 80 experiments
   and got typical measurement/process noise again, how often would each
   candidate run still land in the top 5? Each response is perturbed with
   Gaussian noise at its own empirical standard deviation (a conservative,
   defensible noise scale given script 03's finding that little of that
   standard deviation is attributable to the controllable factors anyway),
   1000 times, and the GRA ranking is recomputed on each perturbed draw.

2. Distinguishing-coefficient (psi) sensitivity. GRA's grey relational
   coefficient has one free parameter, psi (standard default = 0.5, used
   throughout this analysis). This checks whether the top-5 changes across
   the plausible range psi in {0.1, 0.3, 0.5, 0.7, 0.9}.

Input : data/AA5083_original.csv
Output: results/06_rank_robustness_montecarlo.csv
        results/06_psi_sensitivity.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "AA5083_original.csv")
RESP_ALL = ["CTR_mm", "TBT_mm", "BBT_mm", "Circularity_Error_Ce_mm",
            "Cylindricity_Error_Cy_mm", "Conicity_Error_Co_mm"]
RESP_NO_CTR = [r for r in RESP_ALL if r != "CTR_mm"]
rng = np.random.default_rng(seed=42)


def grg_from_matrix(values_df, cols, psi=0.5, weights=None):
    norm = pd.DataFrame({c: (values_df[c].max() - values_df[c]) / (values_df[c].max() - values_df[c].min())
                          for c in cols})
    delta = 1 - norm
    xi = psi / (delta + psi)
    if weights is None:
        weights = pd.Series(1 / len(cols), index=cols)
    return (xi[cols] * weights).sum(axis=1)


# ---- 1. Monte Carlo perturbation ----
N_ITER = 1000
noise_sd = {c: df[c].std() for c in RESP_ALL}
top5_counts_6 = {exp: 0 for exp in df["Exp_No"]}
top5_counts_5 = {exp: 0 for exp in df["Exp_No"]}

for _ in range(N_ITER):
    perturbed = df.copy()
    for c in RESP_ALL:
        perturbed[c] = df[c] + rng.normal(0, noise_sd[c], size=len(df))
        perturbed[c] = perturbed[c].clip(lower=0)  # these are all physically non-negative quantities
    grg6 = grg_from_matrix(perturbed, RESP_ALL)
    grg5 = grg_from_matrix(perturbed, RESP_NO_CTR)
    for exp in grg6.nlargest(5).index:
        top5_counts_6[df.loc[exp, "Exp_No"]] += 1
    for exp in grg5.nlargest(5).index:
        top5_counts_5[df.loc[exp, "Exp_No"]] += 1

mc_df = pd.DataFrame({
    "Exp_No": list(top5_counts_6.keys()),
    "pct_top5_with_CTR": [v / N_ITER * 100 for v in top5_counts_6.values()],
    "pct_top5_without_CTR": [top5_counts_5[k] / N_ITER * 100 for k in top5_counts_6.keys()],
}).merge(df[["Exp_No", "Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"]], on="Exp_No")
mc_df = mc_df.sort_values("pct_top5_without_CTR", ascending=False)
mc_df.to_csv(ROOT / "results" / "06_rank_robustness_montecarlo.csv", index=False)

print(f"Monte Carlo rank-robustness ({N_ITER} noise-perturbed redraws), top 10 most stable runs:")
print(mc_df.head(10).to_string(index=False))
print(f"\nFor reference -- Exp 43 (the CTR-driven outlier flagged earlier): "
      f"in top-5 {top5_counts_6[43]/N_ITER*100:.1f}% of draws with CTR included, "
      f"{top5_counts_5[43]/N_ITER*100:.1f}% without.")

# ---- 2. Psi sensitivity ----
psi_rows = []
top5_sets = {}
for psi in [0.1, 0.3, 0.5, 0.7, 0.9]:
    grg5 = grg_from_matrix(df, RESP_NO_CTR, psi=psi)
    top5 = set(df.loc[grg5.nlargest(5).index, "Exp_No"])
    top5_sets[psi] = top5
    psi_rows.append({"psi": psi, "top5_ExpNo": sorted(top5)})
psi_df = pd.DataFrame(psi_rows)
psi_df.to_csv(ROOT / "results" / "06_psi_sensitivity.csv", index=False)
print("\nGRA top-5 across the plausible range of the distinguishing coefficient (psi):")
print(psi_df.to_string(index=False))
common = set.intersection(*top5_sets.values())
print(f"\nExp_No in the top-5 at EVERY tested psi value: {sorted(common)}")
