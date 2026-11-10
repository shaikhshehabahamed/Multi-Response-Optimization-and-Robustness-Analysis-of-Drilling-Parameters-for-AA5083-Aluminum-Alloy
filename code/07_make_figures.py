"""
07_make_figures.py
--------------------
Generates all five static figures from the results/*.csv files produced by
scripts 02-06. Run this last (or independently, since it only reads CSVs).

Output: figures/fig1_variance_contribution.png
        figures/fig2_insample_vs_loocv.png
        figures/fig3_gra_main_effects.png
        figures/fig4_speed_dia_interaction.png
        figures/fig5_montecarlo_robustness.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})
BLUE, ORANGE, TEAL, GRAY, AMBER, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#eda100", "#e87ba4"
SHORT_LABELS = {"CTR_mm": "CTR", "TBT_mm": "TBT", "BBT_mm": "BBT",
                "Circularity_Error_Ce_mm": "Circularity", "Cylindricity_Error_Cy_mm": "Cylindricity",
                "Conicity_Error_Co_mm": "Conicity"}

# ================= Figure 1: variance contribution =================
df1 = pd.read_csv(RES / "02_anova_grouped_pct.csv", index_col=0)
df1.index = [SHORT_LABELS.get(r, r) for r in df1.index]
cats = ["Speed_pct", "Feed_pct", "Dia_pct", "Interactions_pct", "Quadratic_pct", "Residual_pct"]
cat_labels = ["Speed", "Feed", "Tool diameter", "2-way interactions", "Quadratic", "Residual (unexplained)"]
colors = [BLUE, ORANGE, TEAL, AMBER, MAGENTA, GRAY]

fig, ax = plt.subplots(figsize=(9, 5))
left = np.zeros(len(df1))
for cat, lab, col in zip(cats, cat_labels, colors):
    ax.barh(df1.index, df1[cat], left=left, label=lab, color=col)
    left += df1[cat].values
ax.set_xlabel("% of variance explained")
ax.set_title("Where each drilling defect's variance actually comes from")
ax.set_xlim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
ax.invert_yaxis()
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig1_variance_contribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ================= Figure 2: in-sample vs LOOCV =================
df2 = pd.read_csv(RES / "03_loocv_comparison.csv", index_col=0)
df2.index = [SHORT_LABELS.get(r, r) for r in df2.index]
x = np.arange(len(df2))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w, df2["InSample_R2_RSM"], w, label="In-sample R\u00b2 (optimistic)", color=BLUE)
ax.bar(x, df2["LOOCV_Q2_RSM"], w, label="LOOCV Q\u00b2 \u2014 RSM (honest)", color=ORANGE)
ax.bar(x + w, df2["LOOCV_Q2_GP"], w, label="LOOCV Q\u00b2 \u2014 Gaussian process (honest)", color=TEAL)
ax.axhline(0, color="#52514e", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(df2.index)
ax.set_ylabel("R\u00b2 / Q\u00b2")
ax.set_title("In-sample R\u00b2 overstates what's actually predictable")
ax.legend(loc="upper right", frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig2_insample_vs_loocv.png", dpi=150, bbox_inches="tight")
plt.close()

# ================= Figure 3: GRA main effects =================
df3 = pd.read_csv(RES / "04_gra_main_effects.csv")
factors = [("Spindle_Speed_RPM", "Spindle speed (rpm)"), ("Feed_Rate_mm_min", "Feed rate (mm/min)"),
           ("Tool_Dia_mm", "Tool diameter (mm)")]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (factor, flabel) in zip(axes, factors):
    for scheme, col, lab in [("GRG_equal6", GRAY, "incl. CTR"), ("GRG_equal5", TEAL, "excl. CTR")]:
        sub = df3[(df3.factor == factor) & (df3.scheme == scheme)].sort_values("level")
        ax.plot(sub["level"].astype(str), sub["mean_GRG"], marker="o", color=col,
                linestyle="--" if scheme == "GRG_equal6" else "-", label=lab)
    ax.set_title(flabel, fontsize=11)
    ax.set_ylim(0.6, 0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("Mean grey relational grade")
axes[0].legend(loc="lower left", frameon=False, fontsize=8)
fig.suptitle("Best level per factor is stable whether or not CTR is included", y=1.03, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "fig3_gra_main_effects.png", dpi=150, bbox_inches="tight")
plt.close()

# ================= Figure 4: Speed x Diameter interaction =================
df4 = pd.read_csv(RES / "04_speed_dia_interaction.csv")
pivot = df4.pivot(index="Spindle_Speed_RPM", columns="Tool_Dia_mm", values="mean_GRG_equal5")
fig, ax = plt.subplots(figsize=(7.5, 5))
dia_colors = {6: TEAL, 8: AMBER, 10: BLUE, 12: MAGENTA}
for dia in pivot.columns:
    ax.plot(pivot.index.astype(str), pivot[dia], marker="o", label=f"{dia}mm", color=dia_colors.get(dia, GRAY))
ax.set_xlabel("Spindle speed (rpm)")
ax.set_ylabel("Mean grey relational grade (excl. CTR)")
ax.set_title("The best speed direction reverses depending on tool diameter")
ax.legend(title="Tool diameter", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig4_speed_dia_interaction.png", dpi=150, bbox_inches="tight")
plt.close()

# ================= Figure 5: Monte Carlo robustness =================
df5 = pd.read_csv(RES / "06_rank_robustness_montecarlo.csv").head(10).sort_values("pct_top5_without_CTR")
labels = [f"Exp {int(r.Exp_No)} ({int(r.Spindle_Speed_RPM)}rpm/{int(r.Feed_Rate_mm_min)}mm-min/{int(r.Tool_Dia_mm)}mm)"
          for r in df5.itertuples()]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(labels, df5["pct_top5_without_CTR"], color=TEAL)
ax.set_xlabel("% of 1,000 noise-perturbed redraws landing in top 5")
ax.set_title("No single run dominates \u2014 a robust cluster, not one winner")
ax.set_xlim(0, 45)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "fig5_montecarlo_robustness.png", dpi=150, bbox_inches="tight")
plt.close()

print("All 5 figures written to figures/")
for f in sorted(FIG.glob("*.png")):
    print(" -", f.name)
