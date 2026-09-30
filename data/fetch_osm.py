"""Fetch OpenStreetMap ground truth for K04 and K05.

Writes two small GeoJSON files next to this script:

* `osm_sgr.geojson`   — standard-gauge track in the K04 study area, tagged with
  which phase it belongs to: `Mombasa-Nairobi` (opened May 2017),
  `Nairobi-Naivasha` (opened October 2019), or `other` (unnamed sidings and
  yards, kept so K04 can exclude them from its background);
* `osm_konza.geojson` — the Konza Techno City plot outline;
* `osm_crops.geojson` — tea and rice fields mapped in three K05 test tiles
  (Kericho, the Mau forest edge, and the Mwea irrigation scheme).

These are what K04 and K05 check their results against, so they come from a
source that is independent of TESSERA. Run once; the notebooks read the files.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
URL = "https://overpass-api.de/api/interpreter"
# Overpass refuses requests without a descriptive User-Agent (HTTP 406).
HEADERS = {"User-Agent": "planetary-ai-tessera-kenya/0.1 (research notebook)"}
BBOX = "-1.9,36.2,-0.8,37.4"          # south, west, north, east


def overpass(query: str, attempts: int = 5) -> list[dict]:
    """Run an Overpass query, retrying: the public server often returns 429/504."""
    data = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(URL, data=data, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=180) as resp:
                return json.load(resp)["elements"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            print(f"  {exc}; retrying in {20 * attempt}s", flush=True)
            time.sleep(20 * attempt)


def phase(name: str) -> str:
    if "Mombasa" in name:
        return "Mombasa-Nairobi"
    if "Naivasha" in name and "Kisumu" not in name:
        return "Nairobi-Naivasha"
    return "other"


print("fetching SGR track ...", flush=True)
ways = overpass(f'[out:json][timeout:120];way["railway"="rail"]["gauge"="1435"]({BBOX});out geom;')
sgr = gpd.GeoDataFrame(
    [{"name": w["tags"].get("name", ""), "phase": phase(w["tags"].get("name", "")),
      "geometry": LineString([(p["lon"], p["lat"]) for p in w["geometry"]])}
     for w in ways if len(w["geometry"]) > 1],
    crs="EPSG:4326")
sgr.to_file(HERE / "osm_sgr.geojson", driver="GeoJSON")
print(f"  osm_sgr.geojson: {len(sgr)} ways", sgr.phase.value_counts().to_dict())

print("fetching Konza Techno City ...", flush=True)
ways = overpass(f'[out:json][timeout:60];way["name"="Konza Techno City"]({BBOX});out geom;')
konza = gpd.GeoDataFrame(
    [{"name": w["tags"]["name"], "landuse": w["tags"].get("landuse", ""),
      "geometry": Polygon([(p["lon"], p["lat"]) for p in w["geometry"]])} for w in ways],
    crs="EPSG:4326")
konza.to_file(HERE / "osm_konza.geojson", driver="GeoJSON")
print(f"  osm_konza.geojson: {len(konza)} outline(s)")

# One bbox per K05 test tile: Kericho (35.25, -0.35), Mau (35.45, -0.45) and
# Mwea (37.35, -0.65).
CROP_TILES = {"Kericho": "-0.4,35.2,-0.3,35.3", "Mau": "-0.5,35.4,-0.4,35.5",
              "Mwea": "-0.7,37.3,-0.6,37.4"}
print("fetching tea and rice fields ...", flush=True)
rows = []
for tile, bbox in CROP_TILES.items():
    for w in overpass(f'[out:json][timeout:120];way["crop"~"^(tea|rice)$"]({bbox});out geom;'):
        ring = [(p["lon"], p["lat"]) for p in w["geometry"]]
        if len(ring) >= 4 and ring[0] == ring[-1]:        # closed outlines only
            rows.append({"tile": tile, "crop": w["tags"]["crop"], "osm_id": w["id"],
                         "geometry": Polygon(ring)})
crops = gpd.GeoDataFrame(rows, crs="EPSG:4326")
crops.to_file(HERE / "osm_crops.geojson", driver="GeoJSON")
print(f"  osm_crops.geojson: {len(crops)} fields",
      crops.groupby(["tile", "crop"]).size().to_dict())
