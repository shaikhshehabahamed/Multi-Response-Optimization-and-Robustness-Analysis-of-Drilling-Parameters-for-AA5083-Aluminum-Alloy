"""
01_data_overview.py
--------------------
Structural inspection of the AA5083 drilling dataset: shape, missing values,
duplicate checks, confirmation of the full-factorial design, and descriptive
statistics for all six response variables. Also documents the CTR_mm data
quality flag (high decimal precision + exact-value duplication across
unrelated factor combinations, inconsistent with a directly gauged
measurement).

Input : data/AA5083_original.csv
Output: results/01_descriptive_stats.csv
        results/01_factorial_check.csv
        results/01_ctr_duplicate_values.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "AA5083_original.csv")

RESPONSES = ["CTR_mm", "TBT_mm", "BBT_mm", "Circularity_Error_Ce_mm",
             "Cylindricity_Error_Cy_mm", "Conicity_Error_Co_mm"]
FACTORS = ["Spindle_Speed_RPM", "Feed_Rate_mm_min", "Tool_Dia_mm"]

print(f"Shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ---- Confirm full-factorial design: every combination appears exactly once ----
combo_counts = df.groupby(FACTORS).size()
factorial_check = pd.DataFrame({
    "n_factor_levels": [df[f].nunique() for f in FACTORS],
    "levels": [sorted(df[f].unique()) for f in FACTORS],
}, index=FACTORS)
factorial_check.loc["SUMMARY", "n_factor_levels"] = combo_counts.shape[0]
factorial_check.to_csv(ROOT / "results" / "01_factorial_check.csv")
print(f"\nUnique factor combinations: {combo_counts.shape[0]} "
      f"(min count/combo = {combo_counts.min()}, max = {combo_counts.max()})")

# ---- Descriptive statistics per response ----
desc = df[RESPONSES].describe().T
desc["decimals_observed"] = [
    df[c].apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0).max()
    for c in RESPONSES
]
desc.to_csv(ROOT / "results" / "01_descriptive_stats.csv")
print("\nDescriptive statistics (see results/01_descriptive_stats.csv):")
print(desc.round(4))

# ---- CTR_mm data-quality flag: duplicated values across unrelated runs ----
vc = df["CTR_mm"].value_counts()
dupes = vc[vc > 1]
dup_rows = df[df["CTR_mm"].isin(dupes.index)][
    ["Exp_No"] + FACTORS + ["CTR_mm"]
].sort_values("CTR_mm")
dup_rows.to_csv(ROOT / "results" / "01_ctr_duplicate_values.csv", index=False)
print(f"\nCTR_mm: {df['CTR_mm'].nunique()} unique values across {len(df)} rows "
      f"({len(dupes)} values repeat, covering {dup_rows.shape[0]} rows).")
print("Full detail in results/01_ctr_duplicate_values.csv")
