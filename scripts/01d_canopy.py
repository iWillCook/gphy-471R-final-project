"""
01d_canopy.py
Download the 2023 NLCD Tree Canopy Cover (USFS v2023-5, CONUS) and
clip it to the training region.
"""

import zipfile
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import config as c

c.make_dirs()
zip_path = c.canopy_dir / c.tcc_url.split("/")[-1]   # keep original file name

# ---------------- Download Canopy Cover (if it doesn't exist yet) ----------------
if zip_path.exists() and zipfile.is_zipfile(zip_path):
    print(f"Already downloaded {zip_path.name}")
else:
    print(f"Downloading {c.tcc_url}")
    tmp = zip_path.with_suffix(".part")
    with requests.get(c.tcc_url, stream=True, timeout=60) as r:
        r.raise_for_status()                              # stops if link is broken
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):   # 8 MB at a time to save memory
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="")
    print()
    tmp.rename(zip_path)   # rename once the download finishes

# ---------------- Find raster inside the zip ----------------
with zipfile.ZipFile(zip_path) as z:
    rasters = [i for i in z.infolist() if i.filename.lower().endswith((".tif", ".img"))]
member = max(rasters, key=lambda i: i.file_size).filename   # biggest raster = main layer
print(f"Raster in zip: {member}")

# ---------------- Read only study area ----------------
out = c.canopy_dir / "canopy_2023.tif"
src_path = f"/vsizip/{zip_path.as_posix()}/{member}"

with rasterio.open(src_path) as src:
    print(f"Full raster: {src.width} x {src.height} px, CRS {src.crs}")

    # Convert lon/lat box into raster's coordinates, then to pixel window
    bounds = transform_bounds(c.geo, src.crs, *c.bbox("region"), densify_pts=21)    #
    window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()

    data = src.read(1, window=window)            # read just that window
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        width=window.width,
        height=window.height,
        transform=src.window_transform(window),  # georeferencing clipped piece
        compress="deflate",
        tiled=True,
    )

with rasterio.open(out, "w", **profile) as dst:
    dst.write(data, 1)

print(f"Clipped canopy: {data.shape}, values {data.min()} to {data.max()}, "
      f"nodata = {profile.get('nodata')}")
print(f"Saved {out}")