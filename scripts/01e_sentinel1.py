"""
01e_sentinel1.py
Read OPERA RTC-S1 Sentinel-1 backscatter (VV, VH, layover/shadow mask) at each
SNOTEL station using earthaccess. Requires NASA Earthdata login.

First-time setup: the Earthdata account must be authorized for ASF data.
If reading files fails with a 403 error or "not recognized as being in a
supported file format", log in at https://urs.earthdata.nasa.gov and go to
Applications > Authorized Apps > Approve More Applications, then approve
"Alaska Satellite Facility Data Access" and accept any ASF EULA.
"""

import datetime as dt
import re
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import rasterio
import threading
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Point, Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as c

c.make_dirs()

# Log in. Asks for username/password the first time and saves them to ~/.netrc.
# so later runs log in automatically.
earthaccess.login(strategy="all", persist=True)

# Search area = box around all stations, plus a small margin
stations = pd.read_csv(c.snotel_dir / "stations.csv")
bbox = (stations.lon.min() - 0.05, stations.lat.min() - 0.05,
        stations.lon.max() + 0.05, stations.lat.max() + 0.05)

# Splits an OPERA file name into its parts (track, burst, dates, satellite)
pattern = re.compile(
    r"T(?P<track>\d{3})-(?P<burst>\d{6})-(?P<swath>IW\d)_"
    r"(?P<acq>\d{8}T\d{6})Z_(?P<prod>\d{8}T\d{6})Z_(?P<sat>S1[A-D])"
)


def footprint(g):
    """Lon/lat outline of a granule from its catalog metadata."""
    geom = g["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
    pts = geom["GPolygons"][0]["Boundary"]["Points"]
    return Polygon([(p["Longitude"], p["Latitude"]) for p in pts])


def fix_link(url, name):
    """Point old 'datapool' links to the 'cumulus' server.
    Granules processed in April 2025 only list datapool.asf.alaska.edu links
    which GDAL can't open (403). The same files are on the cumulus server."""
    if "datapool.asf.alaska.edu" in url:
        file = url.split("/")[-1]
        return f"https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/{name}/{file}"
    return url


def search(start, end):
    """Search OPERA RTC-S1 and return a table with one row per burst acquisition."""
    results = earthaccess.search_data(
        short_name=c.s1_name,
        bounding_box=bbox,
        temporal=(str(start), str(end + dt.timedelta(days=1))),   # +1 day so end date is included
    )
    rows = []
    for g in results:
        name = g["umm"]["GranuleUR"]
        m = pattern.search(name)
        if m is None:
            continue

        # Find VV, VH, and mask file links
        links = g.data_links()
        urls = {}
        for layer in ["VV", "VH", "mask"]:
            hits = [u for u in links if u.endswith(f"_{layer}.tif")]
            if hits:
                urls[layer] = fix_link(hits[0], name)
        if len(urls) < 3:   # skips granules missing a layer
            continue

        t = dt.datetime.strptime(m["acq"], "%Y%m%dT%H%M%S")
        rows.append({
            "granule": name,
            "track": int(m["track"]),
            "burst": f"{m['track']}-{m['burst']}-{m['swath']}",
            "sat": m["sat"],
            "time": t,
            "prod": dt.datetime.strptime(m["prod"], "%Y%m%dT%H%M%S"),
            "orbit": "asc" if t.hour < 6 or t.hour > 20 else "desc",   # Utah overpass times
            "vv": urls["VV"], "vh": urls["VH"], "mask": urls["mask"],
            "footprint": footprint(g),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Some burts processed more than once, reprocessed version timestamp can differ by
    # a few seconds; burst imaged at most once a day, so match on burst + date and keep
    # only newest version
    df["date"] = df["time"].dt.date
    n = len(df)
    df = df.sort_values("prod").drop_duplicates(["burst", "date"], keep="last")
    print(f"  removed {n - len(df)} older duplicate versions")
    return df.sort_values("time").reset_index(drop=True)


def gdal_env():
    """GDAL settings for reading NASA files over the internet with your Earthdata login.
    Uses the login saved in ~/.netrc. Each thread gets its own cookie file,
    since threads sharing one file overwrite each other's logins."""
    home = Path.home()
    jar = home / f".urs_cookies_{threading.get_ident()}"   # one cookie file per thread
    return rasterio.Env(
        GDAL_HTTP_NETRC="YES",
        GDAL_HTTP_NETRC_FILE=str(home / ".netrc"),
        GDAL_HTTP_COOKIEFILE=str(jar),
        GDAL_HTTP_COOKIEJAR=str(jar),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",   # don't list the remote folder (slow)
        GDAL_HTTP_MAX_RETRY="5",                    # retry if the server hiccups
        GDAL_HTTP_RETRY_DELAY="3",
    )


def stations_in(fp):
    """Stations whose location falls inside a granule footprint."""
    inside = [fp.contains(Point(lon, lat)) for lon, lat in zip(stations.lon, stations.lat)]
    return stations[inside]


def read_chips(g, sts):
    """Read a small VV/VH/mask window around each station from one granule.
    g: one row of the search() table
    sts: stations inside this granule
    Returns a list of dicts, one per pixel."""
    h = c.s1_half
    n = 2 * h + 1
    out = []
    with gdal_env():
        with rasterio.open(g["vv"]) as vv, rasterio.open(g["vh"]) as vh, rasterio.open(g["mask"]) as mk:
            to_utm = Transformer.from_crs(c.geo, vv.crs, always_xy=True)

            for st in sts.itertuples():
                # Station lon/lat -> UTM meters -> row/column in this file
                x, y = to_utm.transform(st.lon, st.lat)
                row, col = vv.index(x, y)

                # Skip if window hangs off edge of the burst
                if row - h < 0 or col - h < 0 or row + h >= vv.height or col + h >= vv.width:
                    continue

                win = Window(col - h, row - h, n, n)   # (column offset, row offset, width, height)
                a_vv = vv.read(1, window=win)
                a_vh = vh.read(1, window=win)
                a_mk = mk.read(1, window=win)

                if np.all(a_mk == 255):   # station is in the burst's no-data area
                    continue

                for i in range(n):
                    for j in range(n):
                        out.append({
                            "id": st.id, "granule": g["granule"], "time": g["time"],
                            "track": g["track"], "orbit": g["orbit"], "burst": g["burst"],
                            "period": g["period"], "wy": g["wy"],
                            "dr": i - h, "dc": j - h,     # pixel offset from the station
                            "vv": float(a_vv[i, j]), "vh": float(a_vh[i, j]),
                            "mask": int(a_mk[i, j]),
                        })
    return out


# ---------------- Read station chips for each water year ----------------
for wy in c.wys:
    out_file = c.s1_dir / f"chips_wy{wy}.parquet"   # Using .parquet instead of .csv for speed and file size
    if out_file.exists():
        print(f"WY{wy}: already done ({out_file.name})")
        continue

    # Search both windows; label each granule with its period and water year
    tables = []
    for period, (start, end) in c.windows(wy).items():
        df = search(start, end)
        df["period"] = period
        df["wy"] = wy
        tables.append(df)
    granules = pd.concat(tables, ignore_index=True)

    # Pair each granule with the stations inside it; skip granules with none
    jobs = []
    for _, g in granules.iterrows():
        sts = stations_in(g["footprint"])
        if len(sts) > 0:
            jobs.append((g, sts))
    print(f"WY{wy}: reading {len(jobs)} of {len(granules)} granules")

    # Read 8 granules at a time to limit waiting for server
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(read_chips, g, sts): (g, sts) for g, sts in jobs}
        for k, fut in enumerate(as_completed(futures), start=1):
            try:
                rows.extend(fut.result())
            except Exception:
                failed.append(futures[fut])  # save it to retry later
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} granules done, {len(failed)} failed so far")

    # Retry failures one at a time
    if failed:
        print(f"  retrying {len(failed)} failed granules...")
        for g, sts in failed:
            try:
                rows.extend(read_chips(g, sts))
            except Exception as err:
                print(f"  ! {g['granule']}: {err}")

    chips = pd.DataFrame(rows)
    chips.to_parquet(out_file, index=False)
    print(f"Saved {len(chips):,} pixel rows to {out_file.name}")