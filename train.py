#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import transform_geom
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shp_transform, unary_union
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
            text = poly.find('.//k:coordinates', ns).text.strip()
            pts = []
            for coord in text.split():
                lon, lat, *_ = coord.split(',')
                pts.append((float(lon), float(lat)))
            if len(pts) >= 3:
                polys.append(Polygon(pts))
    return polys

def main():
    DATA_ROOT = Path("Data")  # your data folder
    # 1) Locate your TIFF and its KML folder
    tif_path = next(DATA_ROOT.rglob("*.tif"))
    kml_dir  = tif_path.parent / "Numbers"
    kml_paths = sorted(kml_dir.glob("*.kml"))
    if not kml_paths:
        raise RuntimeError(f"No KMLs found in {kml_dir}")

    # 2) Read full TIFF
    with rasterio.open(tif_path) as src:
        full_img = src.read(1)
        full_tf  = src.transform
        full_crs = src.crs

    # 3) Reproject all KMLs into raster CRS
    geoms_full = []
    for kml in kml_paths:
        for poly in parse_kml_polygons(kml):
            gj     = mapping(poly)
            reproj = transform_geom("EPSG:4326", full_crs, gj)
            geoms_full.append(shape(reproj))

    # 4) Plot full TIFF + all boundaries
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    show(full_img, transform=full_tf, ax=ax1, cmap="gray")
    for geom in geoms_full:
        xs, ys = geom.exterior.xy
        ax1.plot(xs, ys, color="orange", linewidth=1.5)
    ax1.set_title("1) Full TIFF with ALL KML boundaries")
    ax1.axis("off")

    # 5) Rotate full TIFF + all KMLs
    angle = 30
    full_r  = ndi_rotate(full_img, angle, reshape=False)
    h_f, w_f = full_r.shape
    centre_f = ((w_f - 1) / 2, (h_f - 1) / 2)

    # transform polygons to pixel-space then rotate
    pix_geoms = [
        shp_transform(lambda x, y: (~full_tf) * (x, y), geom)
        for geom in geoms_full
    ]
    rot_pix_geoms = [
        shp_rotate(geom, -angle, origin=centre_f)
        for geom in pix_geoms
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(full_r, cmap="gray", origin="upper")
    for geom in rot_pix_geoms:
        xs, ys = geom.exterior.xy
        ax2.plot(xs, ys, color="orange", linewidth=1.5)
    ax2.set_title(f"2) Full TIFF + all KMLs rotated {angle}°")
    ax2.axis("off")

    # 6) Crop patch for the first KML
    single_geom = geoms_full[0]
    with rasterio.open(tif_path) as src:
        patch_arr, patch_tf = mask(src, [single_geom], crop=True)
    patch = patch_arr[0]

    # convert polygon to pixel coords in the cropped window
    inv_tf   = ~patch_tf
    pix_geom = shp_transform(lambda x, y: inv_tf * (x, y), single_geom)
    h, w     = patch.shape

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(patch, cmap="gray", origin="upper")
    xs, ys = pix_geom.exterior.xy
    ax3.plot(xs, ys, color="orange", linewidth=1.5)
    ax3.set_title("3) Cropped patch (first KML)")
    ax3.axis("off")

    # 7) Rotate cropped patch + boundary
    patch_r    = ndi_rotate(patch, angle, reshape=False)
    centre     = ((w - 1) / 2, (h - 1) / 2)
    pix_geom_r = shp_rotate(pix_geom, -angle, origin=centre)

    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.imshow(patch_r, cmap="gray", origin="upper")
    xr, yr = pix_geom_r.exterior.xy
    ax4.plot(xr, yr, color="orange", linewidth=1.5)
    ax4.set_title(f"4) Rotated patch & boundary ({angle}°)")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
