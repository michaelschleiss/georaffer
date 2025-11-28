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

## Adding New Regions

Subclass `BaseDownloader` and implement:

- `parse_catalog()` - Parse the region's tile feed/catalog
- `grid_to_filename()` - Convert grid coordinates to download URL
- `utm_to_grid_coords()` - Convert UTM to the region's grid system

See `georaffer/downloaders/nrw.py` as reference.

## License

MIT
