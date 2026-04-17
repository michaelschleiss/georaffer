"""Map preview of requested tiles on a satellite basemap.

Produces a self-contained Leaflet HTML file showing the requested input
(KML polygons or bbox), the resolved UTM tile grid, and the margin ring.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import utm

from georaffer.config import METERS_PER_KM, OUTPUT_TILE_SIZE_KM

# <coordinates> tag, with or without the KML namespace prefix.
_COORD_TAG_RE = re.compile(r"\{[^}]*\}coordinates$|^coordinates$")


@dataclass(frozen=True)
class PreviewBBox:
    """WGS84 bounding box in lon/lat degrees."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2


@dataclass(frozen=True)
class UtmGrid:
    """Tile grid extent in UTM meters, aligned to the tile size."""

    utm_zone: int
    e_min: int
    n_min: int
    e_max: int  # exclusive (one tile past the last included column)
    n_max: int  # exclusive
    tile_size_m: int

    @property
    def cols(self) -> int:
        return (self.e_max - self.e_min) // self.tile_size_m

    @property
    def rows(self) -> int:
        return (self.n_max - self.n_min) // self.tile_size_m

    @property
    def tile_count(self) -> int:
        return self.cols * self.rows

    def wgs84_ring(self) -> list[list[float]]:
        """Return four corners (SW, NW, NE, SE) as [lat, lon] for Leaflet."""
        corners_utm = [
            (self.e_min, self.n_min),
            (self.e_min, self.n_max),
            (self.e_max, self.n_max),
            (self.e_max, self.n_min),
        ]
        return [list(utm.to_latlon(e, n, self.utm_zone, northern=True)) for e, n in corners_utm]


def parse_kml(path: str | Path) -> tuple[list[list[tuple[float, float]]], PreviewBBox]:
    """Extract polygon rings and WGS84 bbox from a KML file.

    Returns a list of rings (each a list of (lon, lat) tuples) plus the
    overall bbox. Raises ``ValueError`` if no geometry is found.
    """
    text = Path(path).read_text(encoding="utf-8")
    root = ET.fromstring(text)

    rings: list[list[tuple[float, float]]] = []
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")

    for elem in root.iter():
        if not _COORD_TAG_RE.search(elem.tag) or not elem.text:
            continue
        ring: list[tuple[float, float]] = []
        for tok in elem.text.split():
            parts = tok.split(",")
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            ring.append((lon, lat))
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat
        if ring:
            rings.append(ring)

    if not rings:
        raise ValueError(f"No coordinates found in KML: {path}")

    return rings, PreviewBBox(min_lon, min_lat, max_lon, max_lat)


def latlon_bbox_to_utm(
    bbox: PreviewBBox,
) -> tuple[float, float, float, float, int]:
    """Convert a WGS84 bbox to UTM meters, returning (xmin, ymin, xmax, ymax, zone)."""
    min_x, min_y, min_zone, _ = utm.from_latlon(bbox.min_lat, bbox.min_lon)
    max_x, max_y, max_zone, _ = utm.from_latlon(bbox.max_lat, bbox.max_lon)
    if min_zone != max_zone:
        raise ValueError(
            f"bbox spans UTM zones {min_zone} and {max_zone}; split by zone before preview."
        )
    return min_x, min_y, max_x, max_y, min_zone


def utm_bbox_to_wgs84(xmin: float, ymin: float, xmax: float, ymax: float, zone: int) -> PreviewBBox:
    """Convert a UTM bbox to its enclosing WGS84 bbox (corner transforms)."""
    # Transform all four corners and take the extremes, since UTM->lonlat
    # rotates the rectangle slightly relative to meridians/parallels.
    corners = [
        utm.to_latlon(xmin, ymin, zone, northern=True),
        utm.to_latlon(xmin, ymax, zone, northern=True),
        utm.to_latlon(xmax, ymax, zone, northern=True),
        utm.to_latlon(xmax, ymin, zone, northern=True),
    ]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    return PreviewBBox(min(lons), min(lats), max(lons), max(lats))


def utm_grid_for_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    zone: int,
    margin: int = 0,
    tile_size_m: int | None = None,
) -> UtmGrid:
    """Compute the UTM tile grid that covers ``[xmin, xmax] x [ymin, ymax]``.

    Mirrors ``load_from_bbox``'s tile snapping (floor division) and extends
    the extent by ``margin`` rings on each side.
    """
    tile = tile_size_m if tile_size_m is not None else int(OUTPUT_TILE_SIZE_KM * METERS_PER_KM)
    if tile <= 0:
        raise ValueError("tile_size_m must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    e0 = int(xmin // tile) * tile
    n0 = int(ymin // tile) * tile
    # End is exclusive: step one tile past the last cell that contains max.
    e1 = (int(xmax // tile) + 1) * tile
    n1 = (int(ymax // tile) + 1) * tile

    ring = margin * tile
    return UtmGrid(
        utm_zone=zone,
        e_min=e0 - ring,
        n_min=n0 - ring,
        e_max=e1 + ring,
        n_max=n1 + ring,
        tile_size_m=tile,
    )


# The template is a static Leaflet page with a single JSON payload injected
# at `__DATA__`. Keeping it as a plain string avoids f-string brace escaping.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
 html,body,#map{height:100%;margin:0}
 .gr-label{background:rgba(0,0,0,0.78);color:#fff;padding:4px 6px;
   font:12px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;
   border:1px solid #00e0ff;border-radius:3px;white-space:nowrap}
 .gr-margin{border-color:#ff9933;color:#ff9933}
 .gr-legend{position:absolute;z-index:1000;bottom:12px;left:12px;
   background:rgba(0,0,0,0.78);color:#fff;padding:8px 10px;border-radius:4px;
   font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}
 .gr-legend .sw{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}
</style>
</head>
<body>
<div id="map"></div>
<div class="gr-legend" id="legend"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;
const map = L.map('map');
L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'Esri World Imagery', maxZoom: 19}
).addTo(map);

const group = L.featureGroup().addTo(map);
const legendItems = [];

// KML polygons (red).
if (DATA.polygons && DATA.polygons.length) {
  DATA.polygons.forEach(ring => {
    L.polygon(ring, {color:'#ff3333', weight:2, fillOpacity:0.1}).addTo(group);
  });
  legendItems.push(['#ff3333', 'solid', 'KML polygons']);
}

// Requested bbox (cyan dashed).
if (DATA.inputBbox) {
  const b = DATA.inputBbox;
  L.rectangle([[b.min_lat,b.min_lon],[b.max_lat,b.max_lon]],
    {color:'#00e0ff', weight:2, fill:false, dashArray:'6 4'}).addTo(group);
  legendItems.push(['#00e0ff', 'dashed', 'requested bbox']);
}

// Tile grid (yellow solid).
if (DATA.tileGrid) {
  const g = DATA.tileGrid;
  L.polygon(g.ring, {color:'#ffff33', weight:3, fill:false})
    .bindTooltip(`tile grid: ${g.cols}\u00d7${g.rows} = ${g.cols*g.rows} tiles`,
      {sticky:true}).addTo(group);
  legendItems.push(['#ffff33', 'solid', `tile grid (${g.cols}\u00d7${g.rows} = ${g.cols*g.rows} tiles)`]);
}

// Margin ring (orange dashed), only when margin > 0.
if (DATA.marginGrid) {
  const m = DATA.marginGrid;
  L.polygon(m.ring, {color:'#ff9933', weight:3, fill:false, dashArray:'4 4'})
    .bindTooltip(`with margin ${m.margin}: ${m.cols}\u00d7${m.rows} = ${m.cols*m.rows} tiles`,
      {sticky:true}).addTo(group);
  legendItems.push(['#ff9933', 'dashed',
    `margin ${m.margin} (${m.cols}\u00d7${m.rows} = ${m.cols*m.rows} tiles)`]);
}

// UTM corner labels.
(DATA.cornerLabels || []).forEach(l => {
  L.marker([l.lat, l.lon], {opacity:0, interactive:false}).addTo(map)
    .bindTooltip(
      `<div class="gr-label ${l.style||''}">${l.html}</div>`,
      {permanent:true, direction:l.direction||'right',
       offset:l.offset||[8,0], className:'gr-label-wrap'}
    );
});

map.fitBounds(group.getBounds(), {padding:[24,24]});
L.control.scale({imperial:false}).addTo(map);

document.getElementById('legend').innerHTML = legendItems.map(([c, style, text]) => {
  const dash = style === 'dashed' ? 'border-top:2px dashed '+c+';background:none' : 'background:'+c;
  return `<div><span class="sw" style="${dash}"></span>${text}</div>`;
}).join('');
</script>
</body>
</html>
"""


def _corner_labels(
    grid: UtmGrid, *, title: str, style: str, direction_offsets: dict[str, list[int]]
) -> list[dict]:
    """Build NW/SE corner label records for a grid."""
    nw = utm.to_latlon(grid.e_min, grid.n_max, grid.utm_zone, northern=True)
    se = utm.to_latlon(grid.e_max, grid.n_min, grid.utm_zone, northern=True)
    return [
        {
            "lat": nw[0],
            "lon": nw[1],
            "html": (f"{title} NW (UTM {grid.utm_zone}N)<br>E: {grid.e_min}<br>N: {grid.n_max}"),
            "style": style,
            "direction": "right",
            "offset": direction_offsets.get("nw", [8, 0]),
        },
        {
            "lat": se[0],
            "lon": se[1],
            "html": (f"{title} SE (UTM {grid.utm_zone}N)<br>E: {grid.e_max}<br>N: {grid.n_min}"),
            "style": style,
            "direction": "left",
            "offset": direction_offsets.get("se", [-8, 0]),
        },
    ]


def build_preview_data(
    *,
    utm_zone: int,
    utm_xmin: float,
    utm_ymin: float,
    utm_xmax: float,
    utm_ymax: float,
    margin: int,
    tile_size_m: int,
    kml_polygons: list[list[tuple[float, float]]] | None = None,
    title: str = "georaffer preview",
) -> dict:
    """Assemble the JSON payload consumed by the HTML template."""

    tile_grid = utm_grid_for_bbox(
        utm_xmin, utm_ymin, utm_xmax, utm_ymax, utm_zone, margin=0, tile_size_m=tile_size_m
    )
    input_bbox_wgs84 = utm_bbox_to_wgs84(utm_xmin, utm_ymin, utm_xmax, utm_ymax, utm_zone)

    tile_grid_record = {
        "cols": tile_grid.cols,
        "rows": tile_grid.rows,
        "ring": tile_grid.wgs84_ring(),
        "e_min": tile_grid.e_min,
        "n_min": tile_grid.n_min,
        "e_max": tile_grid.e_max,
        "n_max": tile_grid.n_max,
    }

    labels = _corner_labels(
        tile_grid,
        title="tile",
        style="",  # default cyan border
        direction_offsets={"nw": [8, 0], "se": [-8, 0]},
    )

    margin_record = None
    if margin > 0:
        margin_grid = utm_grid_for_bbox(
            utm_xmin,
            utm_ymin,
            utm_xmax,
            utm_ymax,
            utm_zone,
            margin=margin,
            tile_size_m=tile_size_m,
        )
        margin_record = {
            "margin": margin,
            "cols": margin_grid.cols,
            "rows": margin_grid.rows,
            "ring": margin_grid.wgs84_ring(),
            "e_min": margin_grid.e_min,
            "n_min": margin_grid.n_min,
            "e_max": margin_grid.e_max,
            "n_max": margin_grid.n_max,
        }
        labels.extend(
            _corner_labels(
                margin_grid,
                title=f"margin {margin}",
                style="gr-margin",
                direction_offsets={"nw": [8, 30], "se": [-8, -30]},
            )
        )

    polygons = None
    if kml_polygons:
        # Convert (lon, lat) -> [lat, lon] for Leaflet.
        polygons = [[[lat, lon] for lon, lat in ring] for ring in kml_polygons]

    return {
        "title": title,
        "polygons": polygons,
        "inputBbox": {
            "min_lon": input_bbox_wgs84.min_lon,
            "min_lat": input_bbox_wgs84.min_lat,
            "max_lon": input_bbox_wgs84.max_lon,
            "max_lat": input_bbox_wgs84.max_lat,
        },
        "tileGrid": tile_grid_record,
        "marginGrid": margin_record,
        "cornerLabels": labels,
    }


def render_preview_html(data: dict) -> str:
    """Render the Leaflet HTML page from a preview data dict."""
    title = html.escape(str(data.get("title", "georaffer preview")))
    payload = json.dumps(data, separators=(",", ":"))
    # The payload is embedded inside <script>; prevent HTML/JS parser confusion
    # on any `</script>` or stray `<!--` sequences that may appear in string fields.
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--")
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


def write_preview_html(data: dict, output: str | Path) -> Path:
    """Write the rendered preview HTML and return the path."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_preview_html(data), encoding="utf-8")
    return out
