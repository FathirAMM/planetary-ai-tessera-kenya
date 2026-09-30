"""Shared helpers for the Kenya notebook series (K01–), imported as `K`.

What is fixed here is this repo's conventions:

* one gitignored download directory, so a tile pulled by K01 is not pulled
  again later (see `EMBEDDINGS_DIR` for how to point it at an existing cache);
* `dataset_version="v1"` / `dataset_variant="vultr"` pinned everywhere, because
  embeddings from different dataset versions are **not** comparable;
* the landmask checks, which Kenya's coast and its many lakes make necessary
  (see K01 §4).

Every TESSERA tile over Kenya carries all nine years, 2017–2025, which is what
makes a change-detection series here straightforward. See `DEEP_YEARS` and
K01 §2.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from geotessera import GeoTessera
from rasterio.transform import xy as _pixel_to_projected
from rasterio.warp import transform as _reproject

REPO_ROOT = Path(__file__).resolve().parent.parent   # this file lives in src/

#: Where downloaded tiles land. Gitignored, ~163 MB per tile-year (a ~158 MB
#: embedding plus a ~5 MB scales file), so this grows fast — the full nine-year
#: Kenya archive is ~7 TB.
#:
#: Set GEOTESSERA_EMBEDDINGS_DIR to reuse a cache you already have (or to put it
#: on a bigger disk) rather than re-downloading. Otherwise it sits in the repo
#: root, where it may be a symlink to a shared cache.
EMBEDDINGS_DIR = Path(os.environ.get(
    "GEOTESSERA_EMBEDDINGS_DIR", REPO_ROOT / "global_0.1_degree_representation"))

#: Kenya's bounding box, generous by a little on every side.
KENYA_BBOX = (33.8, -4.8, 42.0, 5.6)

#: The default single-year snapshot. Any year in DEEP_YEARS would do — Kenya
#: has no year that is more complete than the others.
YEAR = 2024

#: Every year TESSERA covers Kenya for. All 4,766 tiles have all nine. See K01.
DEEP_YEARS = list(range(2017, 2026))

#: Places the series has downloaded for all nine years, as {region: [tiles]}:
#: the eight sites of the old restlessness test, Lake Baringo (K02), and the
#: Konza and SGR escarpment tiles (K04). Anything run over these needs no new
#: downloads.
CACHED_REGIONS = {
    "Kitale maize":        [(34.95, 1.05)],
    "Kericho tea":         [(35.25, -0.35)],
    "Mwea rice":           [(37.35, -0.65)],
    "Nairobi":             [(36.85, -1.25)],
    "Mau Forest":          [(35.45, -0.45)],
    "Maasai Mara":         [(35.15, -1.45)],
    "Marsabit desert":     [(37.55, 2.95)],
    "Uasin Gishu":         [(35.85, 0.05)],
    "Lake Baringo":        [(lon, lat) for lat in (0.75, 0.65, 0.55) for lon in (35.95, 36.05, 36.15)],
    "Konza":               [(37.15, -1.65), (37.15, -1.75), (37.05, -1.65)],
    "Rift escarpment":     [(36.55, -1.05), (36.55, -1.15), (36.55, -1.25), (36.65, -1.35)],
}

#: Four clearly separated hues — teal, purple, yellow, orange. The Lake Victoria
#: basin deliberately gets the loudest colour: it is only 204 tiles (4%) in one
#: small western corner, and would vanish on the national map otherwise.
ZONE_COLOURS = {
    "Highlands & Rift": "#2a9d8f",
    "Lake Victoria Basin": "#7209b7",
    "Coast": "#e9c46a",
    "Arid & Semi-Arid (ASAL)": "#e76f51",
}


def client() -> GeoTessera:
    """A GeoTessera pinned to this repo's dataset version and cache."""
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    return GeoTessera(dataset_version="v1", dataset_variant="vultr",
                      embeddings_dir=EMBEDDINGS_DIR)


def catalogue() -> pd.DataFrame:
    """The per-tile coverage table built by `data/build_catalogue.py`.

    Columns: lon, lat, province, zone, n_years, years, has_2024, multiyear.
    """
    return pd.read_csv(REPO_ROOT / "data" / "kenya_tiles.csv")


def tile_of(lon: float, lat: float) -> tuple[float, float]:
    """The 0.1-degree tile *centre* containing a point.

    TESSERA tiles are named by their centre, so a point at (36.82, -1.29) lives
    in tile (36.85, -1.25), not (36.85, -1.35).

    The `round(..., 6)` is not cosmetic. `-1.30 / 0.1` evaluates to
    `-13.000000000000002`, so a bare `floor` drops it a whole tile south and
    silently returns the wrong neighbour. Kenya straddles the equator, so
    roughly half its tiles have a negative latitude and hit this.
    """
    return (round(math.floor(round(lon / 0.1, 6)) * 0.1 + 0.05, 2),
            round(math.floor(round(lat / 0.1, 6)) * 0.1 + 0.05, 2))


def to_lonlat(row: int, col: int, crs, transform) -> tuple[float, float]:
    """Turn a (row, col) pixel inside a tile back into (lon, lat) degrees,
    so a result can be pasted straight into a map."""
    x, y = _pixel_to_projected(transform, row, col)
    lon, lat = _reproject(crs, "EPSG:4326", [x], [y])
    return lon[0], lat[0]


def rowcol_of(lon: float, lat: float, shape, tile_lon: float, tile_lat: float):
    """Approximate (row, col) of a point within its tile.

    Good to a pixel or two — the tile is in UTM, not degrees, so the mapping is
    not exactly linear. Fine for placing a marker; use `to_lonlat` for the
    reverse direction when precision matters.
    """
    h, w = shape[:2]
    row = int((tile_lat + 0.05 - lat) / 0.1 * h)
    col = int((lon - (tile_lon - 0.05)) / 0.1 * w)
    return np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product gives cosine similarity.

    All-zero rows are TESSERA's "no data here" marker (K01 §4). Normalising
    them would divide by zero, so they are left as zeros instead of silently
    becoming NaN.
    """
    vectors = np.atleast_2d(vectors)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors),
                     where=lengths > 0)


def valid_mask(embedding: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of pixels that actually carry data.

    Two distinct failures have to be caught together, because neither one
    announces itself and they need different tests:

    * **outside the landmask** -> a vector of 128 exact zeros (NOT NaN, so
      `np.isnan` misses it and `mean()` happily returns 0.0);
    * **genuinely missing** -> NaN.
    """
    finite = np.isfinite(embedding).all(axis=-1)
    nonzero = np.linalg.norm(np.nan_to_num(embedding), axis=-1) > 0
    return finite & nonzero


#: Two tiles whose identity is not in doubt, used to build the water and land
#: reference fingerprints (K01 §4): open Lake Victoria and central Nairobi.
WATER_REF_TILE = (33.95, -1.05)
LAND_REF_TILE = (36.85, -1.25)


def reference(gt: GeoTessera, lon: float, lat: float, year: int = YEAR) -> np.ndarray:
    """The average fingerprint of one tile, as a unit vector.

    Averages *unit* vectors rather than raw ones, so a few high-magnitude
    pixels cannot dominate the reference.
    """
    emb, _, _ = fetch(gt, lon, lat, year)
    return unit(unit(emb.reshape(-1, emb.shape[-1])).mean(0))[0]


def water_mask(embedding: np.ndarray, water_ref: np.ndarray,
               land_ref: np.ndarray) -> np.ndarray:
    """Boolean (H, W): True where a pixel resembles `water_ref` more than
    `land_ref` (cosine). Pixels without data are False.

    This is the water/land split validated in K01 §4. It is a two-way choice,
    so anything that is neither (swamp, wet mud) goes to whichever it is
    nearer.
    """
    h, w, d = embedding.shape
    u = unit(embedding.reshape(-1, d))
    wet = (u @ water_ref) > (u @ land_ref)
    return wet.reshape(h, w) & valid_mask(embedding)


def step_change(stack: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """When did each pixel change most, and by how much?

    `stack` is (T, N, D): one fingerprint per year for N pixels. For every split
    point b = 1..T-1, the years before and after are averaged (as unit vectors)
    and compared by cosine distance. The split is chosen by that distance
    weighted by sqrt(b * (T - b)), the usual change-point weighting: without it,
    a single odd first or last year would look like a lasting change.

    Returns `(distance, first_new)`, both length N: the cosine distance between
    the before and after averages at the chosen split, and the index of the
    first year *after* it (1 = the second year). Pixels without data in any
    year get distance NaN and index -1.
    """
    T, N, _ = stack.shape
    distance = np.full(N, np.nan, np.float32)
    first_new = np.full(N, -1, np.int8)
    weight = np.sqrt([b * (T - b) for b in range(1, T)]) / (T / 2)
    for start in range(0, N, chunk):
        part = stack[:, start:start + chunk]
        ok = np.all(np.linalg.norm(part, axis=-1) > 0, axis=0)
        u = np.stack([unit(year) for year in part[:, ok]])      # (T, n, D)
        running = np.cumsum(u, axis=0)
        total = running[-1]
        best_score = np.full(ok.sum(), -np.inf, np.float32)
        best_dist = np.zeros(ok.sum(), np.float32)
        best_b = np.zeros(ok.sum(), np.int8)
        for b in range(1, T):
            before, after = unit(running[b - 1]), unit(total - running[b - 1])
            dist = 1 - (before * after).sum(-1)
            score = dist * weight[b - 1]
            better = score > best_score
            best_score[better], best_dist[better], best_b[better] = score[better], dist[better], b
        idx = np.arange(start, min(start + chunk, N))[ok]
        distance[idx], first_new[idx] = best_dist, best_b
    return distance, first_new


def false_colour(gt: GeoTessera, tile, percentile: float = 2.0) -> np.ndarray:
    """PCA a tile's 128 channels down to 3 and stretch to an RGB image.

    `tile` is a `(year, lon, lat, embedding, crs, transform)` tuple as returned
    by `gt.fetch_embeddings`. Colours are only comparable *within* one image —
    the PCA is refitted per tile.
    """
    image = gt.apply_pca_to_embeddings([tile], n_components=3,
                                       standardize=True)[0][3]
    lo, hi = np.nanpercentile(image, percentile), np.nanpercentile(image, 100 - percentile)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def fetch(gt: GeoTessera, lon: float, lat: float, year: int = YEAR):
    """`gt.fetch_embeddings` for one tile, unpacked to the useful three."""
    _, _, _, emb, crs, transform = list(gt.fetch_embeddings([(year, lon, lat)]))[0]
    return emb, crs, transform


def gmaps(lon: float, lat: float) -> str:
    """A Google Maps link for a result, for eyeballing against imagery."""
    return f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},15z/data=!3m1!1e3"


# ---------------------------------------------------------------------------
# Sites used across the series.
#
# Coordinates are hand-picked from satellite imagery. Every site sits on a tile
# with all nine years. K01 §3 verifies that rather than assuming it, and turns up one
# wrinkle: Mombasa's tile has nine years like the rest, but its centre falls in
# the Kilindini channel, so `catalogue()` (which clips by centre) omits it.
# Look that one up via `client().registry`, not the CSV.
# ---------------------------------------------------------------------------
SITES = {
    # --- Rift Valley lakes: the clearest change signal in the country
    "Lake Baringo":        (36.0800,  0.6200, "rift lake that rose ~5 m and flooded its shore after 2010"),
    "Lake Bogoria":        (36.0900,  0.2500, "soda lake, flamingo site, also rising"),
    "Lake Nakuru":         (36.0800, -0.3700, "national park lake that drowned its own gate and fence lines"),
    "Lake Naivasha":       (36.3500, -0.7700, "freshwater lake ringed by export floriculture greenhouses"),
    "Olkaria geothermal":  (36.2900, -0.9000, "Africa's largest geothermal field, still expanding"),
    # --- Highlands: forest, water towers, tea
    "Mount Kenya":         (37.3070, -0.1520, "glaciated peak and its forest belt; the glaciers are nearly gone"),
    "Mau Forest (South West Mau)": (35.4000, -0.4800, "the country's largest water tower, and its longest-running deforestation fight"),
    "Kericho tea":         (35.2800, -0.3700, "the tea belt — a near-monoculture with a very distinctive signature"),
    "Aberdare Range":      (36.7000, -0.4200, "montane forest and moorland above the Central highlands"),
    "Kakamega Forest":     (34.8700,  0.3500, "Kenya's only tract of Guineo-Congolian rainforest, and shrinking"),
    # --- Cities and infrastructure
    "Nairobi":             (36.8200, -1.2900, "the capital; peri-urban sprawl is the fastest land-cover change in Kenya"),
    "Kisumu":              (34.7600, -0.0900, "the lake port, on Winam Gulf"),
    "Mombasa":             (39.6600, -4.0500, "the main port, on an island — a landmask test, see K01"),
    "Konza Technopolis":   (37.1800, -1.6900, "a new city built from scratch on open plain"),
    "SGR Nairobi-Naivasha": (36.5530, -1.1530, "the standard gauge railway cut down the Rift escarpment, opened 2019"),
    # --- ASAL: drought, irrigation, pastoralism
    "Garissa / Tana":      (39.6400, -0.4500, "irrigated riverine agriculture in an otherwise arid county"),
    "Lake Turkana (Ferguson's Gulf)": (35.9000,  3.5500, "the world's largest desert lake"),
    "Lokichar oil basin":  (35.6500,  2.3800, "Turkana's oil discovery area, with new roads and pads since 2015"),
    "Lake Turkana Wind Power": (36.7500,  2.5500, "365 turbines on the Sarima ridge, built 2015-2018"),
    "Marsabit / Chalbi":   (37.5000,  2.9000, "desert margin — the driest end of the gradient"),
    # --- Coast: mangrove, delta, ports
    "Tana River delta":    (40.1500, -2.5500, "delta wetland, mangrove and contested irrigation schemes"),
    "Lamu / Manda Bay":    (40.9000, -2.2700, "mangrove archipelago and the new LAPSSET port"),
    "Amboseli":            (37.2500, -2.6500, "swamp-fed savanna under Kilimanjaro, a drought bellwether"),
    "Maasai Mara":         (35.1500, -1.5000, "savanna rangeland; the wheat frontier presses on its northern edge"),
}
