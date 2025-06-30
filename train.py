import numpy as np
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.ops import transform as shp_transform
from shapely.affinity import rotate as shp_rotate
from scipy.ndimage import rotate as ndi_rotate
import math

# 1) Locate your TIFF + all KMLs
DATA_ROOT = Path("Data")
tif_path  = next(DATA_ROOT.rglob("*.tif"))
kml_dir   = tif_path.parent / "Numbers"
kml_paths = list(kml_dir.glob("*.kml"))

# 2) Read & reproject each KML
with rasterio.open(tif_path) as src:
    src_crs = src.crs

gdfs = []
for kml in kml_paths:
    gdf = gpd.read_file(kml, driver="KML")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdfs.append(gdf.to_crs(src_crs))

# 3) Union all polygons into one for cropping bounds
all_poly = gdfs[0].geometry.union_all()
for gdf in gdfs[1:]:
    all_poly = all_poly.union(gdf.geometry.union_all())

# 4) Compute a square buffer (half–diagonal)
minx, miny, maxx, maxy = all_poly.bounds
cx, cy = (minx+maxx)/2, (miny+maxy)/2
r = math.hypot(maxx-minx, maxy-miny)/2
minx, maxx = cx-r, cx+r
miny, maxy = cy-r, cy+r

# 5) Crop the raster to that square
with rasterio.open(tif_path) as src:
    win       = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    cropped   = src.read(1, window=win)
    cropped_tf= src.window_transform(win)

# 6) Convert each polygon to pixel coords in the cropped window
inv_tf = ~cropped_tf
def geo2pix(x, y, z=None):
    return inv_tf*(x, y)

pix_polys = [shp_transform(geo2pix, gdf.unary_union) for gdf in gdfs]

# 7) Compute the pixel‐centre
h, w = cropped.shape
centre = ((w-1)/2, (h-1)/2)

# 8) Plot helper
def plot_overlay(img, polys, title):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img, cmap="gray", origin="upper")
    for poly in polys:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, linewidth=2, color="orange")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# 9) Show original
plot_overlay(cropped, pix_polys, "Original cropped + all KMLs")

# 10) Rotate image & all polygons (no reshape)
for ang in (15, 30, 45):
    img_r   = ndi_rotate(cropped, ang, reshape=False)
    rot_polys = [shp_rotate(poly, -ang, origin=centre) for poly in pix_polys]
    plot_overlay(img_r, rot_polys, f"Rotated {ang}° (all KMLs)")
