"""Fetch ESA WorldCover 2021 (v200) labels for the tiles K06 trains and tests on.

WorldCover is a free global 10 m land-cover map. It is published as 3 x 3 degree
Cloud-Optimised GeoTIFFs on AWS, so only the window around each TESSERA tile is
read, not the whole file. Each window is saved, still in WorldCover's own grid
(EPSG:4326), as `worldcover_2021/wc_<lon>_<lat>.tif`; K06 reprojects it onto the
tile's grid.

Run once; K06 reads the files. About 12 s per tile.

Source: Zanaga, D. et al. ESA WorldCover 10 m 2021 v200,
https://doi.org/10.5281/zenodo.7254221
"""
import math
import sys
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import kenya as K  # noqa: E402  (after the path tweak)

OUT = HERE / "worldcover_2021"
URL = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
       "ESA_WorldCover_10m_2021_v200_{}_Map.tif")
MARGIN = 0.01                          # ~1 km extra on each side, for reprojection


def source_name(lon: float, lat: float) -> str:
    """WorldCover's 3-degree file holding a point, named by its south-west corner."""
    south, west = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return f"{'N' if south >= 0 else 'S'}{abs(south):02d}{'E' if west >= 0 else 'W'}{abs(west):03d}"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for lon, lat in [t for tiles in K.CACHED_REGIONS.values() for t in tiles]:
        out = OUT / f"wc_{lon:.2f}_{lat:.2f}.tif"
        if out.exists():
            continue
        with rasterio.open(URL.format(source_name(lon, lat))) as src:
            window = from_bounds(lon - 0.05 - MARGIN, lat - 0.05 - MARGIN,
                                 lon + 0.05 + MARGIN, lat + 0.05 + MARGIN, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            profile = dict(driver="GTiff", dtype="uint8", count=1, crs=src.crs,
                           width=data.shape[1], height=data.shape[0],
                           transform=src.window_transform(window), compress="deflate")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"{out.name}: {data.shape}", flush=True)
    print("done")
