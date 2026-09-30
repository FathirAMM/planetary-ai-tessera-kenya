"""Fetch Natural Earth boundaries and clip them to Kenya and its neighbours.

The full 1:10m Natural Earth files are ~52 MB, almost all of it countries this
project never looks at. This script downloads them, keeps only what the
notebooks draw, and rewrites them in place — the result is a few hundred KB and
small enough to keep alongside the code.

Run once; `build_catalogue.py` and K01 read the clipped files.
"""
import io
import urllib.request
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
        "geojson/ne_10m_admin_{}.geojson")

# Kenya plus every country sharing a border with it, so the context maps in
# K01 still have something to draw. Natural Earth spells Tanzania out in full.
KEEP = ["Kenya", "Ethiopia", "Somalia", "South Sudan", "Uganda",
        "United Republic of Tanzania"]

for level, url_part, name_col, out in [
        ("admin0", "0_countries", "ADMIN", "ne_10m_admin0.geojson"),
        ("admin1", "1_states_provinces", "admin", "ne_10m_admin1.geojson")]:
    url = BASE.format(url_part)
    print(f"downloading {level} ...", flush=True)
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    gdf = gpd.read_file(io.BytesIO(raw))
    # admin0 keeps the neighbours for context; admin1 only needs Kenya.
    wanted = KEEP if level == "admin0" else ["Kenya"]
    gdf = gdf[gdf[name_col].isin(wanted)]
    cols = [c for c in ("ADMIN", "ISO_A3", "admin", "name", "type_en", "geometry")
            if c in gdf.columns]
    gdf[cols].to_file(HERE / out, driver="GeoJSON")
    print(f"  {out}: {len(gdf)} features")
