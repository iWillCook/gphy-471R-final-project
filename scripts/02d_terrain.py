"""
02d_terrain.py
Build the terrain and canopy predictors on the 30 m UTM grid of the DEM:
elevation, slope, northness, eastness, topographic position index (TPI),
and tree canopy cover. Sample them at each SNOTEL station and save a
clip over the Cottonwood canyons for mapping.
Outputs: data/processed/terrain_stations.csv
         data/processed/terrain_cw.tif  (one band per predictor)
"""

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter
import config as c

c.make_dirs()
tpi_size = 21    # TPI window in pixels (21 x 30 m = 630 m across)
flat = 1         # slopes under 1 degree have no meaningful aspect


def slope_aspect(dem, res):
    """Slope (degrees) and aspect (degrees clockwise from north) from a DEM.

    dem: 2-D elevation array (m), north-up
    res: pixel size (m)
    """
    d_row, d_col = np.gradient(dem, res)   # change per meter down the rows and across the columns
    dz_east = d_col
    dz_north = -d_row                      # rows increase southward, so flip the sign
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = np.degrees(np.arctan2(-dz_east, -dz_north)) % 360   # direction the slope faces
    return slope, aspect


# ---------------- DEM and its derivatives ----------------
with rasterio.open(c.dem_dir / "dem.tif") as src:
    dem = src.read(1).astype("float32")
    grid = {"crs": src.crs, "transform": src.transform, "shape": dem.shape}

slope, aspect = slope_aspect(dem, c.res)
northness = np.cos(np.radians(aspect))   # +1 = faces north, -1 = faces south
eastness = np.sin(np.radians(aspect))    # +1 = faces east,  -1 = faces west
northness[slope < flat] = 0
eastness[slope < flat] = 0

# TPI: a pixel's elevation minus the mean elevation around it.
# Positive = ridge/exposed, negative = valley/sheltered.
filled = np.where(np.isnan(dem), np.nanmean(dem), dem)   # the filter can't handle NaN
tpi = dem - uniform_filter(filled, size=tpi_size, mode="nearest")

# ---------------- Canopy, reprojected onto the DEM grid ----------------
with rasterio.open(c.canopy_dir / "canopy_2023.tif") as src:
    cc = src.read(1)
    codes, counts = np.unique(cc[cc > 100], return_counts=True)
    print("Canopy values > 100 (set to NaN):", dict(zip(codes.tolist(), counts.tolist())))

    cc = cc.astype("float32")
    cc[cc > 100] = np.nan                 # 0-100 = percent canopy; higher values are codes
    canopy = np.full(grid["shape"], np.nan, dtype="float32")
    reproject(cc, canopy,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=grid["transform"], dst_crs=grid["crs"],
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

layers = {"elev": dem, "slope": slope, "northness": northness,
          "eastness": eastness, "tpi": tpi, "canopy": canopy}

# ---------------- Sample at the stations ----------------
st = pd.read_csv(c.snotel_dir / "stations.csv")
to_utm = Transformer.from_crs(c.geo, grid["crs"], always_xy=True)
x, y = to_utm.transform(st["lon"].values, st["lat"].values)
rows, cols = rasterio.transform.rowcol(grid["transform"], x, y)
rows, cols = np.array(rows), np.array(cols)

for name, arr in layers.items():
    st[name] = arr[rows, cols]

st["dem_minus_station_m"] = st["elev"] - st["elev_ft"] * 0.3048   # DEM is treetop height in forest
out = c.processed / "terrain_stations.csv"
st.drop(columns=["lon", "lat", "elev_ft"]).round(3).to_csv(out, index=False)
print(f"\nSaved terrain at {len(st)} stations to {out.name}")
print(st[list(layers) + ["dem_minus_station_m"]].describe().round(2))

# ---------------- Clip over the Cottonwoods for mapping ----------------
bounds = transform_bounds(c.geo, grid["crs"], *c.bbox("cw_area", buf=0.01))
win = from_bounds(*bounds, transform=grid["transform"]).round_offsets().round_lengths()
rs = slice(win.row_off, win.row_off + win.height)
cs = slice(win.col_off, win.col_off + win.width)

profile = {"driver": "GTiff", "dtype": "float32", "nodata": np.nan,
           "count": len(layers), "width": win.width, "height": win.height,
           "crs": grid["crs"], "transform": rasterio.windows.transform(win, grid["transform"]),
           "compress": "deflate"}
out = c.processed / "terrain_cw.tif"
with rasterio.open(out, "w", **profile) as dst:
    for i, (name, arr) in enumerate(layers.items(), start=1):
        dst.write(arr[rs, cs].astype("float32"), i)
        dst.set_band_description(i, name)
print(f"Saved Cottonwood clip ({win.height} x {win.width} px, {len(layers)} bands) to {out.name}")