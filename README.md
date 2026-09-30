# Planetary AI — TESSERA over Kenya

Mapping Kenya with [TESSERA](https://github.com/ucam-eo/geotessera) satellite
embeddings: every 10 m pixel carries a 128-number "fingerprint" of what the
ground looked like across a year. This repo audits what TESSERA has over Kenya
and uses it for change detection.

## Key facts (from K01)

- **4,766 tiles** (0.1° × 0.1°, about 11 × 11 km each) cover Kenya.
- **Every tile has all nine years, 2017–2025.**
- One tile for one year is **~163 MB**, so the full Kenya archive is ~7 TB.
  Notebooks download only the tiles they use.
- The landmask zeroes out the **sea** but **not inland lakes**, so a "usable"
  pixel is not necessarily land.

## Setup

Requires **Python 3.12+** (geotessera 0.9.0 needs it; on 3.11 pip silently
installs an older geotessera that breaks `K.client()`).

```bash
uv sync --group dev          # or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

In VS Code, select the `.venv` (Python 3.12) kernel for the notebooks.

### Tile cache

Downloaded tiles are saved in `global_0.1_degree_representation/` in the repo
root and never downloaded twice. To reuse an existing cache, symlink it there:

```bash
ln -s /path/to/global_0.1_degree_representation global_0.1_degree_representation
```

or set `GEOTESSERA_EMBEDDINGS_DIR` to its path.

## Layout

| Path | What it is |
|---|---|
| `src/kenya.py` | Shared helpers, imported by notebooks as `K` |
| `data/` | Tile catalogue (`kenya_tiles.csv`), border files, and the scripts that built them |
| `experiments/` | Notebooks, starting with `K01_Coverage_Atlas.ipynb` |
| `outputs/` | Results notebooks can rebuild (K02's water maps, K04's change maps). Gitignored. |
| `documentation/` | Notes, including a file-by-file guide in [`FILES.md`](documentation/FILES.md) and all sources in [`REFERENCES.md`](documentation/REFERENCES.md) |

## Notebooks

| Notebook | What it shows |
|---|---|
| [`K01_Coverage_Atlas`](experiments/K01_Coverage_Atlas.ipynb) | How many tiles and years exist over Kenya, how even the coverage is, where the landmask misleads, and a Lake Baringo 2017→2025 change test. Downloads ~1.5 GB on a cold cache. |
| [`K02_Lake_Baringo`](experiments/K02_Lake_Baringo.ipynb) | The whole of Lake Baringo, every year 2017–2025: area per year (181 → 223 km²), when each shore went under water, land flooded (31 km² for good), and a noise check. Downloads ~13 GB on a cold cache. |
| [`K04_Konza_and_SGR`](experiments/K04_Konza_and_SGR.ipynb) | Dating built change with no labels: the Nairobi–Naivasha SGR lights up (dated 2018–2019), the older Mombasa line (the control) does not, and Konza's roads appear as a street grid dated 2020. Downloads ~10 GB. |
| [`K05_Find_Similar_Places`](experiments/K05_Find_Similar_Places.ipynb) | Point at one field, find more like it: one tea field finds 59% of 51 held-out tea fields, one rice field traces the Mwea paddies, with almost no false alarms across 33 tiles. Uses cached tiles only. |
| [`K06_Land_Cover_Map`](experiments/K06_Land_Cover_Map.ipynb) | First trained model: a linear classifier on the embeddings reproduces ESA WorldCover at 68.6% in regions it never saw. Shows where WorldCover itself falls short (tea as trees, paddies as wetland). Uses cached tiles only. |

## Rebuilding the data

Each script is run once; its outputs are kept in `data/`.

```bash
.venv/bin/python data/fetch_boundaries.py   # country/province borders from Natural Earth
.venv/bin/python data/build_catalogue.py    # data/kenya_tiles.csv from the TESSERA registry
.venv/bin/python data/fetch_osm.py          # OpenStreetMap answer keys for K04 and K05
.venv/bin/python data/fetch_worldcover.py   # ESA WorldCover 2021 labels for K06
```
