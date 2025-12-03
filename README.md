# georaffer

Download German orthophotos and DSM tiles (NRW, RLP) as GeoTIFF.

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

# From specific tile indices
georaffer tiles 362,5604 --output ./tiles
```

See `georaffer --help` for all options.

## File name scheme

Raw downloads keep the provider filenames:

- NRW orthophotos (JP2): `dop10rgbi_32_<grid_x>_<grid_y>_<n>_nw_<year>.jp2`
- NRW DSM (LAZ): `bdom50_32_<grid_x>_<grid_y>_<n>_nw_<year>.laz`
- RLP orthophotos (JP2): `dop20rgb_32_<grid_x>_<grid_y>_2_rp_<year>.jp2`
- RLP DSM (LAZ): `bdom20rgbi_32_<grid_x>_<grid_y>_2_rp.laz`

Processed GeoTIFFs use a unified UTM-based pattern:

- `<region>_32_<easting>_<northing>_<year>.tif`

where:

- `<region>` is `nrw` or `rlp`
- `<easting>` and `<northing>` are UTM coordinates (EPSG:25832) of the tile’s south‑west corner in meters (for example `350000,5600000`)
- `<year>` is the acquisition year inferred from the source filename or LAZ header (falls back to `latest` when no year is available)

When large source tiles are split into smaller output tiles (for example RLP 2km → 1km grid), filenames still follow this pattern and the easting/northing encode the sub‑tile coordinates, e.g. `rlp_32_362500_5604500_2023.tif`.

## Output directory structure

With `--output ./tiles` the pipeline creates:

```text
./tiles
├── raw
│   ├── image        # Provider JP2 orthophotos (NRW, RLP)
│   └── dsm          # Provider LAZ DSM tiles
└── processed
    ├── image
    │   └── <pixels>/      # Orthophoto GeoTIFFs (tile size in pixels, e.g. 2000 for 0.5 m/px on 1 km tiles)
    ├── dsm
    │   └── <pixels>/      # DSM GeoTIFFs with the same convention
    └── provenance.csv     # Per-tile provenance metadata
```

`<pixels>` matches the internal resolution derived from `--pixel-size` (meters per pixel). For example, with the default `--pixel-size 0.5` on a 1 km grid, georaffer produces `processed/image/2000/` and `processed/dsm/2000/`.

Raw tiles remain cached under `raw/`, so you can re-run conversions with different resolutions without re-downloading.

## Adding New Regions

Subclass `BaseDownloader` and implement:

- `parse_catalog()` - Parse the region's tile feed/catalog
- `grid_to_filename()` - Convert grid coordinates to download URL
- `utm_to_grid_coords()` - Convert UTM to the region's grid system

See `georaffer/downloaders/nrw.py` as reference.

## License

MIT
