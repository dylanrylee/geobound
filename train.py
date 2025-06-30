from pathlib import Path
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt

DATA_ROOT = Path("Data")

for tif_path in DATA_ROOT.rglob("*.tif"):
    numbers_dir = tif_path.parent / "Numbers"
    kml_paths  = list(numbers_dir.glob("*.kml")) if numbers_dir.exists() else []
    if not kml_paths:
        continue

    with rasterio.open(tif_path) as src:
        # 1) Read first band & plot in grayscale
        band1 = src.read(1)
        fig, ax = plt.subplots(figsize=(8, 8))
        show(
            band1,
            transform=src.transform,
            ax=ax,
            cmap="gray",
            title=tif_path.stem
        )

        # 2) Overlay each KML—reprojected into the raster’s CRS
        for kml in kml_paths:
            gdf = gpd.read_file(kml, driver="KML")
            # ensure it’s in the same CRS as the TIFF
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)  # KML defaults to WGS84 lon/lat
            gdf = gdf.to_crs(src.crs)

            gdf.plot(
                ax=ax,
                facecolor="none",
                edgecolor="yellow",
                linewidth=1,
                label=kml.stem
            )

        ax.legend(loc="lower right", fontsize="small")
        plt.tight_layout()
        plt.show()