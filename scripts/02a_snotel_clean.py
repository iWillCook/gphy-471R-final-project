"""
02a_snotel_clean.py
Clean the raw SNOTEL daily data: convert units, flag bad snow depth values,
and keep stations with enough usable winters.
Output: data/processed/snotel_clean.csv
"""

import pandas as pd
import config as c

c.make_dirs()

# Quality-control thresholds
max_depth = 600              # cm, deeper than any Wasatch SNOTEL record
max_rise = 50                # cm/day; a rise this big with no SWE gain is a sensor spike
min_swe_rise = 10            # mm; real storms add at least this much SWE
dens_range = (0.04, 0.60)    # plausible snow density (g/cm3)
min_dens_depth = 15          # cm; density is unreliable for very shallow snow

# ---------------- Load data and convert units ----------------
raw = pd.read_csv(c.snotel_dir / "daily.csv", parse_dates=["date"])

df = pd.DataFrame({
    "id": raw["id"],
    "date": raw["date"],
    "depth_cm": raw["SNOWDEPTH"] * 2.54,               # inches -> cm
    "swe_mm": raw["SWE"] * 25.4,                       # inches -> mm
    "tmax_c": (raw["MAX AIR TEMP"] - 32) * 5 / 9,      # deg F -> deg C
    "tmin_c": (raw["MIN AIR TEMP"] - 32) * 5 / 9,
    "tavg_c": (raw["AVG AIR TEMP"] - 32) * 5 / 9,
})
df = df.sort_values(["id", "date"]).reset_index(drop=True)
df["wy"] = df["date"].dt.year + (df["date"].dt.month >= 10)   # Oct-Dec -> next water year

# ---------------- Flag bad depth values ----------------
# Day-to-day changes, computed separately for each station
rise = df.groupby("id")["depth_cm"].diff()
swe_rise = df.groupby("id")["swe_mm"].diff()
dens = df["swe_mm"] / (df["depth_cm"] * 10)   # mm / (cm * 10) = g/cm3

df["flag"] = ""
df.loc[df["depth_cm"].isna(), "flag"] = "missing"
df.loc[(df["depth_cm"] < 0) | (df["depth_cm"] > max_depth), "flag"] = "range"
df.loc[(rise > max_rise) & (swe_rise < min_swe_rise), "flag"] = "spike"
df.loc[(df["depth_cm"] > min_dens_depth) &
       ((dens < dens_range[0]) | (dens > dens_range[1])), "flag"] = "density"

df["good"] = df["flag"] == ""
print("Flagged values:")
print(df["flag"].replace("", "ok").value_counts())

# ---------------- Keep stations with enough usable winters ----------------
# A usable winter = good depth on at least 80% of Dec-Mar days
accum = df[df["date"].dt.month.isin([12, 1, 2, 3])]
frac_good = accum.groupby(["id", "wy"])["good"].mean()
usable = (frac_good >= 0.8).groupby("id").sum()

keep = usable[usable >= c.min_winters].index
dropped = sorted(set(df["id"]) - set(keep))
print(f"\nStations kept: {len(keep)}, dropped (< {c.min_winters} usable winters): {dropped}")
df = df[df["id"].isin(keep)]

# ---------------- Save ----------------
out = c.processed / "snotel_clean.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} station-days ({df['good'].sum():,} good) to {out.name}")