#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.plot import show
from rasterio.warp import transform_geom
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.affinity import rotate as shp_rotate

from scipy.ndimage import rotate as ndi_rotate

def parse_kml_polygons(kml_path):
    """
    Pure-Python KML parser: extract all <Polygon> geometries as Shapely Polygons.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    ns = {'k': root.tag.split('}')[0].strip('{')}
    polys = []
    for placemark in root.findall('.//k:Placemark', ns):
        for poly in placemark.findall('.//k:Polygon', ns):
            coords_text = poly.find('.//k:coordinates', ns).text.strip()
            pts = []
            for pair in coords_text.split():
                lon, lat, *_ = pair.split(',')
                pts.append((float(lon), float(lat)))
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    return polys

def main():
    # adjust these as needed:
    DATA_ROOT = Path("Data")
    margin_fraction = 0.1     # 0.1 = 10% extra around KML crop
    rotate_angle = 30         # degrees

    # 1) find TIFF + KMLs
    tif_path = next(DATA_ROOT.rglob("*.tif"))
    kml_dir = tif_path.parent / "Numbers"
    kml_paths = sorted(kml_dir.glob("*.kml"))
    if not kml_paths:
        raise RuntimeError(f"No KMLs found in {kml_dir}")

    # 2) load full TIFF
    with rasterio.open(tif_path) as src:
        full_img = src.read(1)
        full_tf = src.transform
        full_crs = src.crs

    # 3) parse & reproject all KMLs
    geoms_full = []
    for kml in kml_paths:
        for poly in parse_kml_polygons(kml):
            gj = mapping(poly)
            reproj = transform_geom("EPSG:4326", full_crs, gj)
            geoms_full.append(shape(reproj))

    # 4) plot full TIFF + boundaries
    fig1, ax1 = plt.subplots(figsize=(8,8))
    show(full_img, transform=full_tf, ax=ax1, cmap="gray")
    for geom in geoms_full:
        xs, ys = geom.exterior.xy
        ax1.plot(xs, ys, color="orange", linewidth=1.5)
    ax1.set_title("1) Full TIFF + ALL KML boundaries")
    ax1.axis("off")

    # 5) rotate full TIFF + boundaries
    full_r = ndi_rotate(full_img, rotate_angle, reshape=False)
    h, w = full_r.shape
    centre = ((w-1)/2, (h-1)/2)
    pix_geoms = [ shp_transform(lambda x,y: (~full_tf)*(x,y), g) for g in geoms_full ]
    rot_pix_geoms = [ shp_rotate(g, -rotate_angle, origin=centre) for g in pix_geoms ]
    fig2, ax2 = plt.subplots(figsize=(8,8))
    ax2.imshow(full_r, cmap="gray", origin="upper")
    for geom in rot_pix_geoms:
        xs, ys = geom.exterior.xy
        ax2.plot(xs, ys, color="orange", linewidth=1.5)
    ax2.set_title(f"2) Full TIFF + ALL boundaries rotated {rotate_angle}°")
    ax2.axis("off")

    # 6) buffered crop around first KML
    single = geoms_full[0]
    minx, miny, maxx, maxy = single.bounds
    dx = (maxx-minx)*margin_fraction
    dy = (maxy-miny)*margin_fraction
    minx_m, maxx_m = minx-dx, maxx+dx
    miny_m, maxy_m = miny-dy, maxy+dy
    with rasterio.open(tif_path) as src:
        window = from_bounds(minx_m, miny_m, maxx_m, maxy_m, transform=src.transform)
        patch = src.read(1, window=window)
        patch_tf = src.window_transform(window)

    inv_tf = ~patch_tf
    pix_single = shp_transform(lambda x,y: inv_tf*(x,y), single)
    ph, pw = patch.shape

    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.imshow(patch, cmap="gray", origin="upper")
    xs, ys = pix_single.exterior.xy
    ax3.plot(xs, ys, color="orange", linewidth=1.5)
    ax3.set_title(f"3) Buffered crop (margin={margin_fraction*100:.0f}%)")
    ax3.axis("off")

    # 7) rotate buffered patch + boundary
    patch_r = ndi_rotate(patch, rotate_angle, reshape=False)
    centre_patch = ((pw-1)/2, (ph-1)/2)
    pix_rot = shp_rotate(pix_single, -rotate_angle, origin=centre_patch)
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.imshow(patch_r, cmap="gray", origin="upper")
    xr, yr = pix_rot.exterior.xy
    ax4.plot(xr, yr, color="orange", linewidth=1.5)
    ax4.set_title(f"4) Rotated buffered patch ({rotate_angle}°)")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
