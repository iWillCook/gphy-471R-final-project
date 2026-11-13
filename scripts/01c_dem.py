"""
01c_dem.py
Download the Copernicus GLO-30 DEM for the training region from
Microsoft Planetary Computer and save it on a 30 m UTM 12N grid.
"""

import numpy as np
import odc.stac
import planetary_computer
import pystac_client
import rioxarray
import config as c

c.make_dirs()
bbox = c.bbox("region")

# ---------------- Search the STAC catalog ----------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,   # adds access tokens to each file link
)
items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
print(f"Found {len(items)} DEM tiles")

# ---------------- Load, mosaic, and reproject ----------------
ds = odc.stac.load(
    items,
    bands=["data"],
    bbox=bbox,
    crs=c.utm,               # reproject straight to UTM 12N
    resolution=c.res,        # 30 m pixels
    resampling="bilinear",   # smooth interpolation suits elevation
    dtype="float32",
    nodata=np.nan,
    chunks={"x": 2048, "y": 2048},   # load in pieces to save memory
)
dem = ds["data"].max(dim="time").compute()   # merge tiles into one layer

# ---------------- Save ----------------
dem = dem.rio.write_crs(c.utm).rio.write_nodata(np.nan)
out = c.dem_dir / "dem.tif"
dem.rio.to_raster(out, compress="deflate", tiled=True)

print(f"DEM: {dem.shape[0]} x {dem.shape[1]} pixels, "
      f"{float(dem.min()):.0f} to {float(dem.max()):.0f} m")
print(f"Saved {out}")