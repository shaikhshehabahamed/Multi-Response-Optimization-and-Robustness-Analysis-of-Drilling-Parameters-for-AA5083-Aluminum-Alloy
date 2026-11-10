"""
08_build_master_dataset.py
----------------------------
Joins the original data with every derived score/rank from scripts 04-06
into one convenience file. Run after 04, 05, and 06.

Output: data/AA5083_master_with_all_derived_metrics.csv
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
orig = pd.read_csv(ROOT / "data" / "AA5083_original.csv")
gra = pd.read_csv(ROOT / "results" / "04_gra_scores_and_ranks.csv")
topsis = pd.read_csv(ROOT / "results" / "05_topsis_scores_and_ranks.csv")
mc = pd.read_csv(ROOT / "results" / "06_rank_robustness_montecarlo.csv")[
    ["Exp_No", "pct_top5_with_CTR", "pct_top5_without_CTR"]
]

merged = (orig
          .merge(gra[["Exp_No"] + [c for c in gra.columns if c.startswith("GRG_")]], on="Exp_No")
          .merge(topsis[["Exp_No"] + [c for c in topsis.columns if c.startswith("TOPSIS_")]], on="Exp_No")
          .merge(mc, on="Exp_No"))

out_path = ROOT / "data" / "AA5083_master_with_all_derived_metrics.csv"
merged.to_csv(out_path, index=False)
print(f"Wrote {out_path} -- shape {merged.shape}")
