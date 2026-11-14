"""
01f_daymet.py
Download Daymet V4 daily precipitation and min/max temperature (1 km grid)
at each SNOTEL station using the Daymet Single Pixel Extraction API (no login).
"""

import io
import time
import numpy as np
import pandas as pd
import requests
import config as c

c.make_dirs()


def get_point(lat, lon, start, end):
    """Download the daily Daymet time series for the 1 km cell containing a point.
    lat, lon: decimal degrees
    start, end: datetime.date
    Returns a DataFrame: date, prcp (mm/day), tmax and tmin (deg C), elev_m (cell elevation)
    """
    params = {"lat": lat, "lon": lon, "vars": ",".join(c.daymet_vars),
              "start": str(start), "end": str(end)}

    # Try up to 3 times
    for attempt in range(3):
        try:
            r = requests.get(c.daymet_url, params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5)

    # Response has a few metadata lines and a CSV table starting at "year,yday"
    text = r.text
    header = text.find("year,yday")
    df = pd.read_csv(io.StringIO(text[header:]))
    df.columns = [col.split(" ")[0] for col in df.columns]   # "prcp (mm/day)" -> "prcp"
    df["date"] = pd.to_datetime(df["year"] * 1000 + df["yday"], format="%Y%j").dt.date

    # Grab the Daymet cell elevation from the metadata lines
    elev = np.nan
    for line in text[:header].splitlines():
        if line.lower().startswith("elevation"):
            elev = float(line.split(":")[1].split()[0])
    df["elev_m"] = elev

    return df.drop(columns=["year", "yday"])


# ---------------- Download for every station ----------------
stations = pd.read_csv(c.snotel_dir / "stations.csv")
start, end = c.date_span()
print(f"Daymet for {len(stations)} stations, {start} to {end}")

frames = []
for st in stations.itertuples():
    try:
        df = get_point(st.lat, st.lon, start, end)
    except Exception as err:
        print(f"  ! {st.name}: {err}")
        continue
    df.insert(0, "id", st.id)   # station ID as first column
    frames.append(df)
    print(f"  + {st.name}: {len(df)} days, Daymet cell {df['elev_m'].iloc[0]:.0f} m "
          f"vs station {st.elev_ft * 0.3048:.0f} m")
    time.sleep(0.5)

daymet = pd.concat(frames, ignore_index=True)
out = c.daymet_dir / "daymet_stations.csv"
daymet.to_csv(out, index=False)
print(f"Saved {len(daymet):,} station-days to {out}")