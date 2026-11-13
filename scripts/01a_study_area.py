"""
01a_study_area.py
Download watershed boundaries from the USGS Watershed Boundary Dataset
with pygeohydro and save them to data/raw/study_area/study_area.gpkg.
"""

from pygeohydro import WBD
import config as c

c.make_dirs()

# ---------------- Training basins (HUC-8) ----------------
wbd8 = WBD("huc8")                          # connect to the HUC-8 layer
basins = wbd8.byids("huc8", list(c.hucs))   # request our four basins by ID
basins = basins.to_crs(c.geo)
print(basins[["huc8", "name"]])

# Merge the four basins into one polygon (used to select SNOTEL stations)
region = basins.dissolve()[["geometry"]]

# Save both as layers in one GeoPackage
basins.to_file(c.study_gpkg, layer="basins")
region.to_file(c.study_gpkg, layer="region")

# Area check (area needs meters, so convert to UTM first)
area = basins.to_crs(c.utm).area.sum() / 1e6
print(f"Training region: {area:,.0f} km²")

# ---------------- Cottonwood canyons (HUC-12) ----------------
wbd12 = WBD("huc12")
# Every HUC-12 code starts with its parent HUC-8 code, so this grabs all
# HUC-12s inside the Jordan basin
units = wbd12.bysql(f"huc12 LIKE '{c.cw_huc}%'")
units = units.to_crs(c.geo)

# Keep the ones with "Cottonwood" in the name
cw = units[units["name"].str.contains("Cottonwood", case=False)]

# Keep only the canyon (headwaters) units; the "Outlet" units are valley floor
cw = units[units["huc12"].isin(c.cw_ids)]
print(cw[["huc12", "name", "areasqkm"]])

cw.to_file(c.study_gpkg, layer="cw_units")
cw.dissolve()[["geometry"]].to_file(c.study_gpkg, layer="cw_area")

