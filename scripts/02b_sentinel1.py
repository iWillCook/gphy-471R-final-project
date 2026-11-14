"""
02b_sentinel1.py
Turn the Sentinel-1 pixel chips into one radar observation per station per pass:
mask bad pixels, average the 3x3 window around each station, merge duplicate
passes from overlapping bursts, convert to dB, and compute VH/VV and the
change from each station's snow-free reference on the same orbit track.
Output: data/processed/s1_station.csv
"""

import numpy as np
import pandas as pd
import config as c

c.make_dirs()
min_pix = 5   # need at least 5 of the 9 window pixels to be valid
min_ref = 2   # need at least 2 snow-free passes to build a reference

# ---------------- Load all water years ----------------
chips = pd.concat([pd.read_parquet(c.s1_dir / f"chips_wy{wy}.parquet") for wy in c.wys],
                  ignore_index=True)
print(f"Loaded {len(chips):,} pixel rows")

# ---------------- Mask bad pixels ----------------
# mask: 0 = valid, 1 = shadow, 2 = layover, 3 = both, 255 = no data.
# Backscatter must be > 0 to convert to dB later.
bad = (chips["mask"] != 0) | ~(chips["vv"] > 0) | ~(chips["vh"] > 0)
chips.loc[bad, ["vv", "vh"]] = np.nan

# ---------------- Average the 3x3 window ----------------
# Keep the inner 3x3 (offsets -1 to 1) and average in LINEAR units
win = chips[(chips["dr"].abs() <= 1) & (chips["dc"].abs() <= 1)]
keys = ["id", "granule", "time", "track", "orbit", "period", "wy"]
obs = win.groupby(keys).agg(
    vv=("vv", "mean"),
    vh=("vh", "mean"),
    n_pix=("vv", "count"),   # count skips NaN = number of valid pixels
).reset_index()
obs = obs[obs["n_pix"] >= min_pix]

# ---------------- Local date of each pass ----------------
# Convert to Mountain Standard Time (UTC-7) to match SNOTEL and Daymet dates.
obs["date"] = (obs["time"] - pd.to_timedelta(7, unit="h")).dt.floor("D")

# ---------------- Merge duplicate passes from overlapping bursts ----------------
n_before = len(obs)
obs = obs.groupby(["id", "track", "date"]).agg(
    orbit=("orbit", "first"),
    period=("period", "first"),
    wy=("wy", "first"),
    vv=("vv", "mean"),
    vh=("vh", "mean"),
    n_pix=("n_pix", "sum"),
).reset_index()
print(f"Station passes: {n_before:,} -> {len(obs):,} after merging burst overlaps")

# ---------------- Convert to dB and compute the cross-ratio ----------------
obs["vv_db"] = 10 * np.log10(obs["vv"])
obs["vh_db"] = 10 * np.log10(obs["vh"])
obs["ratio_db"] = obs["vh_db"] - obs["vv_db"]

# ---------------- Snow-free reference (per station, track, and water year) ----------------
# Average the fall passes in LINEAR units, then convert to dB
sf = obs[obs["period"] == "snow_free"]
ref = sf.groupby(["id", "track", "wy"]).agg(
    ref_vv=("vv", "mean"),
    ref_vh=("vh", "mean"),
    n_ref=("vv", "count"),
).reset_index()
ref = ref[ref["n_ref"] >= min_ref]
ref["ref_vv_db"] = 10 * np.log10(ref["ref_vv"])
ref["ref_vh_db"] = 10 * np.log10(ref["ref_vh"])
ref["ref_ratio_db"] = ref["ref_vh_db"] - ref["ref_vv_db"]

# ---------------- Change from the reference (winter passes only) ----------------
acc = obs[obs["period"] == "accum"].merge(
    ref[["id", "track", "wy", "n_ref", "ref_vv_db", "ref_vh_db", "ref_ratio_db"]],
    on=["id", "track", "wy"], how="left")
no_ref = acc["ref_vv_db"].isna()
print(f"Winter passes: {len(acc):,}, without a snow-free reference (dropped): {no_ref.sum()}")
acc = acc[~no_ref].copy()
acc["n_ref"] = acc["n_ref"].astype(int)

acc["dvv"] = acc["vv_db"] - acc["ref_vv_db"]          # change in VV (dB)
acc["dvh"] = acc["vh_db"] - acc["ref_vh_db"]          # change in VH (dB)
acc["dratio"] = acc["ratio_db"] - acc["ref_ratio_db"] # change in VH/VV (dB)

# ---------------- Save ----------------
cols = ["id", "date", "wy", "track", "orbit", "n_pix", "n_ref",
        "vv_db", "vh_db", "ratio_db", "dvv", "dvh", "dratio"]
out = c.processed / "s1_station.csv"
acc[cols].round(3).to_csv(out, index=False)
print(f"Saved {len(acc):,} station passes from {acc['id'].nunique()} stations to {out.name}")
print(acc[["vv_db", "vh_db", "ratio_db", "dvv", "dvh", "dratio"]].describe().round(2))