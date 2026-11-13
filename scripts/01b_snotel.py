"""
01b_snotel.py
Download daily SNOTEL snow depth, SWE, and air temperature for every
station in the training region, using the metloom package.
"""

import datetime as dt
import geopandas as gpd
import pandas as pd
from metloom.pointdata import SnotelPointData
import config as c

c.make_dirs()

# metloom's names for the SNOTEL sensors we want
v = SnotelPointData.ALLOWED_VARIABLES
variables = [v.SNOWDEPTH, v.SWE, v.TEMPMAX, v.TEMPMIN, v.TEMPAVG]

# ---------------- Find stations ----------------
region = gpd.read_file(c.study_gpkg, layer="region")

# Stations inside the region that measure snow depth
points = SnotelPointData.points_from_geometry(region, [v.SNOWDEPTH], snow_courses=False)

# Only return SNOTEL sites - metloom includes other networks
points = [p for p in points if p.id.endswith(":SNTL")]

# Build a station table. p.metadata is a point: x = lon, y = lat, z = elevation (ft)
stations = pd.DataFrame([
    {"id": p.id, "name": p.name,
     "lon": p.metadata.x, "lat": p.metadata.y, "elev_ft": p.metadata.z}
    for p in points
])
stations = stations.sort_values("elev_ft")
print(stations.to_string(index=False))
print(f"{len(stations)} SNOTEL stations")

stations.to_csv(c.snotel_dir / "stations.csv", index=False)

# ---------------- Download daily SNOTEL data ----------------
start, end = c.date_span()
print(f"\nDownloading {start} to {end}")

frames = []
for p in points:
    try:
        df = p.get_daily_data(
            dt.datetime.combine(start, dt.time()),
            dt.datetime.combine(end, dt.time()),
            variables,
        )
    except Exception as err:   # Adds exception for bad data so run doesn't stop
        print(f"  ! {p.name}: {err}")
        continue
    if df is None:
        print(f"  - {p.name}: no data")
        continue

    df = df.reset_index().drop(columns="geometry")
    n = df["SNOWDEPTH"].notna().sum() if "SNOWDEPTH" in df else 0
    print(f"  + {p.name}: {len(df)} days, {n} with depth")
    frames.append(df)

daily = pd.concat(frames, ignore_index=True)
daily = daily.rename(columns={"site": "id"})

# Convert from UTC to Mountain Standard Time so each value lands on the right date.
ts = pd.to_datetime(daily["datetime"])
if ts.dt.tz is not None:
    ts = ts.dt.tz_convert("Etc/GMT+7")
daily["date"] = ts.dt.date

daily.to_csv(c.snotel_dir / "daily.csv", index=False)
print(f"Saved {len(daily):,} station-days")