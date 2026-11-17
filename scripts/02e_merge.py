"""
02e_merge.py
Join the radar, snow depth, climate, and terrain data into one table with a
row per station per Sentinel-1 pass, remove wet-snow observations, and check
how each predictor relates to snow depth.
Output: data/processed/model_table.csv
"""

import pandas as pd
from scipy.stats import spearmanr
import config as c

c.make_dirs()
wet_db = -2.0   # VV drop (dB) vs the snow-free reference that signals wet snow (Lievens et al., 2022)

# ---------------- Load ----------------
s1 = pd.read_csv(c.processed / "s1_station.csv", parse_dates=["date"])
snotel = pd.read_csv(c.processed / "snotel_clean.csv", parse_dates=["date"])
daymet = pd.read_csv(c.processed / "daymet_features.csv", parse_dates=["date"])
terrain = pd.read_csv(c.processed / "terrain_stations.csv")

# ---------------- Radar + snow depth ----------------
# Inner join keeps only passes with a good SNOTEL depth on the same day
snotel = snotel[snotel["good"]]
df = s1.merge(snotel[["id", "date", "depth_cm", "swe_mm", "tmax_c", "tmin_c"]],
              on=["id", "date"], how="inner")
print(f"Radar passes: {len(s1):,} -> {len(df):,} with a good SNOTEL depth "
      f"({s1['id'].nunique()} -> {df['id'].nunique()} stations)")

# ---------------- Wet-snow filter ----------------
# 1. Temperature at the station, matched to the pass time:
#    morning (descending) pass -> wet if it stayed above freezing overnight (Tmin > 0)
#    evening (ascending) pass  -> wet if it got above freezing that day (Tmax > 0)
temp = df["tmin_c"].where(df["orbit"] == "desc", df["tmax_c"])
wet_temp = temp > 0        # missing temperature -> False (radar check still applies)
# 2. Radar: wet snow absorbs the signal, so VV drops below the snow-free reference
wet_radar = df["dvv"] < wet_db
wet = wet_temp | wet_radar

print(f"\nWet-snow filter ({len(df):,} passes):")
print(f"  temperature rule: {wet_temp.sum():,}")
print(f"  radar rule:       {wet_radar.sum():,}")
print(f"  both:             {(wet_temp & wet_radar).sum():,}")
print(f"  removed (either): {wet.sum():,} ({wet.mean():.0%})")
print(f"  missing temperature (radar rule only): {temp.isna().sum()}")

# Keep a copy with wet rows to compare correlations before/after
all_passes = df.copy()
df = df[~wet]

# ---------------- Add climate and terrain ----------------
df = df.merge(daymet[["id", "date", "cum_prcp", "prcp_7d", "pdd"]], on=["id", "date"], how="left")
df = df.merge(terrain[["id", "name", "elev", "slope", "northness", "eastness", "tpi", "canopy"]],
              on="id", how="left")

missing = df[c.features].isna().any(axis=1)
print(f"\nRows with a missing predictor (dropped): {missing.sum()}")
df = df[~missing].reset_index(drop=True)

# ---------------- Predictor relationship assessment ----------------
rows = []
for f in c.features:
    rho, p = spearmanr(df[f], df["depth_cm"])
    rows.append({"feature": f, "rho": rho, "p": p})
corr = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
print("\nSpearman correlation with snow depth (dry passes):")
print(corr.round(3).to_string(index=False))

rho_all, _ = spearmanr(all_passes["dratio"], all_passes["depth_cm"])
rho_dry, _ = spearmanr(df["dratio"], df["depth_cm"])
print(f"\ndratio vs depth: rho = {rho_all:.3f} all passes -> {rho_dry:.3f} dry passes")

# ---------------- Save ----------------
cols = ["id", "name", "date", "wy", "track", "orbit", "depth_cm", "swe_mm"] + c.features
out = c.processed / "model_table.csv"
df[cols].round(3).to_csv(out, index=False)
print(f"\nSaved {len(df):,} rows from {df['id'].nunique()} stations to {out.name}")