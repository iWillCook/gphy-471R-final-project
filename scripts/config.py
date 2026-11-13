"""
config.py
Shared settings for all project scripts (paths, study area, dates).
"""

from pathlib import Path
import datetime as dt

# ---------------- Paths ----------------
root = Path(__file__).resolve().parents[1]   # repo folder
raw = root / "data" / "raw"
processed = root / "data" / "processed"

study_dir = raw / "study_area"
snotel_dir = raw / "snotel"
study_gpkg = study_dir / "study_area.gpkg"
dem_dir = raw / "dem"

# ---------------- Time period ----------------
wys = [2021]   # water years; expand to list(range(2017, 2026)) later


def windows(wy):
    """Snow-free and accumulation date windows for one water year."""
    return {
        "snow_free": (dt.date(wy - 1, 10, 1), dt.date(wy - 1, 11, 10)),
        "accum": (dt.date(wy - 1, 12, 1), dt.date(wy, 3, 31)),
    }


def date_span():
    """First and last date needed across all water years."""
    return dt.date(min(wys) - 1, 10, 1), dt.date(max(wys), 3, 31)


# ---------------- Study area ----------------
hucs = {   # HUC-8 training basins
    "16020101": "Upper Weber",
    "16020102": "Lower Weber",
    "16020203": "Provo",
    "16020204": "Jordan",
}
cw_huc = "16020204"   # Jordan basin, which contains the Cottonwood canyons
cw_ids = ["160202040201", "160202040202"]   # Big & Little Cottonwood headwaters (canyons only)

geo = "EPSG:4326"     # lat/lon
utm = "EPSG:32612"    # UTM zone 12N, meters
res = 30   # analysis pixel size (m)


def bbox(layer="region", buf=0.02):
    """Lon/lat bounding box of a study area layer, plus a small buffer (degrees)."""
    import geopandas as gpd
    minx, miny, maxx, maxy = gpd.read_file(study_gpkg, layer=layer).to_crs(geo).total_bounds
    return (minx - buf, miny - buf, maxx + buf, maxy + buf)


def make_dirs():
    """Create the data folders if they don't exist yet."""
    for d in [study_dir, snotel_dir, dem_dir, processed]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(root)
    make_dirs()
    print(windows(2021))
    print(date_span())