# georaffer

Download German orthophotos and DSM tiles (NRW, RLP, BB) as GeoTIFF.

## Installation

### pip (macOS, Linux x86, Windows)

```bash
pip install georaffer
```

### conda-forge (all platforms incl. Linux ARM64)

```bash
git clone https://github.com/michaelschleiss/georaffer
cd georaffer
conda create -n georaffer libgdal-jp2openjpeg lazrs-python laspy rasterio -c conda-forge
conda activate georaffer
pip install -e .
```

Or with [pixi](https://pixi.sh):

```bash
git clone https://github.com/michaelschleiss/georaffer
cd georaffer
pixi install
```

## Usage

```bash
# From 4Seasons dataset
georaffer pygeon /data/4seasons/campaign --output ./tiles

# From CSV with coordinates
georaffer csv coords.csv --cols lon,lat --output ./tiles

# From bounding box
georaffer bbox 6.9,50.9,7.1,51.1 --output ./tiles

# From existing GeoTIFF footprint
georaffer tif ./area.tif --output ./tiles

# From specific tile indices
georaffer tiles 362,5604 --output ./tiles
```

See `georaffer --help` for all options.
By default, downloads target NRW + RLP; add `--region bb` to include Brandenburg.

When using the `tif` command, aligned outputs are written to `./tiles/aligned/` and
match the reference GeoTIFF grid (CRS, pixel size, width/height, and bounds).
Alignment currently requires all processed tiles to share the same CRS; mixed-zone
runs (for example NRW/RLP zone 32 plus BB zone 33) will fail and should be split
by zone.

## Python API

```python
from georaffer import TileStore

store = TileStore(path="./tiles", regions=["NRW", "RLP"], imagery_from=(2015, None))

# Query tiles at 1km grid coordinates
tiles = store.query(coords=(350, 5600), tile_type="image")

# Retrieve (downloads + converts if needed)
path = store.get(tiles[0], resolution=2000)

# Batch with pipelined download/convert
paths = store.get_many(tiles, resolution=2000)
```

For single-file conversion without TileStore:

```python
from georaffer.conversion import convert_file

convert_file("raw/image/tile.jp2", "processed", resolution=2000)
```

## File name scheme

Raw downloads keep the provider filenames:

- NRW orthophotos (JP2): `dop10rgbi_32_<grid_x>_<grid_y>_<n>_nw_<year>.jp2`
- NRW DSM (LAZ): `bdom50_32_<grid_x>_<grid_y>_<n>_nw_<year>.laz`
- RLP orthophotos (JP2): `dop20rgb_32_<grid_x>_<grid_y>_2_rp_<year>.jp2`
- RLP DSM (LAZ): `bdom20rgbi_32_<grid_x>_<grid_y>_2_rp.laz`
- BB DSM (ZIP → TIF): `bdom_<zone><grid_x>-<grid_y>.zip` (extracts to `.tif`)

Processed GeoTIFFs use a unified UTM-based pattern:

- `<region>_<zone>_<easting>_<northing>_<year>.tif`

where:

- `<region>` is `nrw`, `rlp`, or `bb`
- `<zone>` is the UTM zone (`32` for NRW/RLP, `33` for BB)
- `<easting>` and `<northing>` are UTM coordinates of the tile’s south‑west corner in meters (for example `350000,5600000`)
- `<year>` is the acquisition year inferred from the source filename or LAZ header (falls back to `latest` when no year is available)

When large source tiles are split into smaller output tiles (for example RLP 2km → 1km grid), filenames still follow this pattern and the easting/northing encode the sub‑tile coordinates, e.g. `rlp_32_362500_5604500_2023.tif`.

## Output directory structure

With `--output ./tiles` the pipeline creates:

```text
./tiles
├── raw
│   ├── image        # Provider JP2 orthophotos (NRW, RLP)
│   └── dsm          # Provider DSM tiles (LAZ for NRW/RLP, TIF for BB)
└── processed
    ├── image
    │   └── <pixels>/      # Orthophoto GeoTIFFs (tile size in pixels, e.g. 2000 for 0.5 m/px on 1 km tiles)
    ├── dsm
    │   └── <pixels>/      # DSM GeoTIFFs with the same convention
```

`<pixels>` matches the internal resolution derived from `--pixel-size` (meters per pixel). For example, with the default `--pixel-size 0.5` on a 1 km grid, georaffer produces `processed/image/2000/` and `processed/dsm/2000/`.

Raw tiles remain cached under `raw/`, so you can re-run conversions with different resolutions without re-downloading.

## Adding New Regions

Subclass `BaseDownloader` and implement:

- `parse_catalog()` - Parse the region's tile feed/catalog
- `grid_to_filename()` - Convert grid coordinates to download URL
- `utm_to_grid_coords()` - Convert UTM to the region's grid system

See `georaffer/downloaders/nrw.py` as reference.

## Data Sources and Licensing

German states publish orthophotos and elevation models as open data—free for commercial and non-commercial use—under [EU High-Value Datasets Regulation 2023/138](https://eur-lex.europa.eu/eli/reg_impl/2023/138) and the [INSPIRE Directive](https://inspire.ec.europa.eu/), coordinated nationally by the [AdV](https://www.adv-online.de/) (Arbeitsgemeinschaft der Vermessungsverwaltungen).

| Region | Provider | License |
|--------|----------|---------|
| NRW | [Geobasis NRW](https://www.opengeodata.nrw.de) | [dl-de/zero-2-0](https://www.govdata.de/dl-de/zero-2-0) |
| RLP | [LVermGeo RLP](https://lvermgeo.rlp.de) | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) ¹ |
| BB | [LGB Brandenburg](https://geobasis-bb.de) | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) ¹ |

¹ Attribution required: `©GeoBasis-DE/<provider> (<year>), dl-de/by-2-0`

## License

MIT
