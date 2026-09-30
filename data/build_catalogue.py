"""Build data/kenya_tiles.csv — every TESSERA tile over Kenya, per year,
tagged with province and broad agro-ecological zone. Run once; notebooks read
the CSV.

Natural Earth's admin1 layer for Kenya is the eight **old provinces**, not the
47 counties of the 2010 constitution. That is coarser than a Kenyan reader
expects, but it is what is available at 1:10m and it is enough to answer the
only question this catalogue exists to answer: is TESSERA's coverage spread
evenly across the country, or is it concentrated somewhere?

The ZONE mapping below collapses those eight into four agro-ecological blocks.
It is deliberately coarse and one entry is a genuine compromise: the old
Eastern province runs from Machakos' farmland all the way up to Marsabit's
desert, so calling all of it ASAL is right for most of its area and wrong for
its southern tip.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geotessera import GeoTessera

HERE = Path(__file__).resolve().parent

ZONE = {
    "Central": "Highlands & Rift",
    "Nairobi": "Highlands & Rift",
    "Rift Valley": "Highlands & Rift",
    "Nyanza": "Lake Victoria Basin",
    "Western": "Lake Victoria Basin",
    "Coast": "Coast",
    "Eastern": "Arid & Semi-Arid (ASAL)",
    "North-Eastern": "Arid & Semi-Arid (ASAL)",
}

adm0 = gpd.read_file(HERE / "ne_10m_admin0.geojson")
kenya = adm0[adm0.ADMIN == "Kenya"].geometry.union_all()
adm1 = gpd.read_file(HERE / "ne_10m_admin1.geojson")
adm1 = adm1[adm1.admin == "Kenya"][["name", "geometry"]].reset_index(drop=True)

gt = GeoTessera(dataset_version="v1", dataset_variant="vultr")
YEARS = list(range(2017, 2026))
BBOX = (33.8, -4.8, 42.0, 5.6)            # generous box around Kenya

# Which (lon, lat) tile centres exist in which years?
per_year = {y: {(round(lo, 2), round(la, 2))
                for _, lo, la in gt.registry.load_blocks_for_region(BBOX, y)}
            for y in YEARS}
all_tiles = sorted(set().union(*per_year.values()))

rows = []
for lon, lat in all_tiles:
    pt = Point(lon, lat)
    if not kenya.contains(pt):
        continue                          # tile centre outside Kenya
    hit = adm1[adm1.geometry.contains(pt)]
    prov = hit.iloc[0]["name"] if len(hit) else "(border)"
    years = [y for y in YEARS if (lon, lat) in per_year[y]]
    rows.append({
        "lon": lon, "lat": lat,
        "province": prov,
        "zone": ZONE.get(prov, "(border)"),
        "n_years": len(years),
        "years": "|".join(map(str, years)),
        "has_2024": 2024 in years,
        "multiyear": len(years) >= 8,
    })

df = pd.DataFrame(rows).sort_values(["lat", "lon"], ascending=[False, True])
df.to_csv(HERE / "kenya_tiles.csv", index=False)
print(f"{len(df)} tiles inside Kenya -> kenya_tiles.csv")
print(df.groupby("zone").agg(tiles=("lon", "size"),
                             deep=("multiyear", "sum")).to_string())
print("\nyear-depth histogram:")
print(df.n_years.value_counts().sort_index().to_string())
