"""RLP (Rhineland-Palatinate) tile downloader."""

import xml.etree.ElementTree as ET

import requests
import urllib3

from georaffer.config import METERS_PER_KM, RLP_GRID_SIZE, RLP_JP2_PATTERN, RLP_LAZ_PATTERN, Region
from georaffer.downloaders.base import RegionDownloader

# Suppress SSL warnings (RLP has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class RLPDownloader(RegionDownloader):
    """RLP (Rhineland-Palatinate) downloader.

    Note: RLP does not have historic imagery feeds, so imagery_from is ignored.
    """

    def __init__(
        self,
        output_dir: str,
        imagery_from: tuple[int, int | None] | None = None,
        session: requests.Session | None = None,
    ):
        # RLP ignores imagery_from (no historic imagery available)
        super().__init__(Region.RLP, output_dir, imagery_from=None, session=session)
        self._jp2_feed_url = "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php?id=2b009ae4-aa3e-ff21-870b-49846d9561b2&type=DATASET&generateFrom=remotelist"
        self._laz_feed_url = "https://www.geoportal.rlp.de/mapbender/php/mod_inspireDownloadFeed.php?id=3d2dda7d-b4b5-47d2-b074-dd45edd36738&type=DATASET&generateFrom=remotelist"

    def utm_to_grid_coords(
        self, utm_x: float, utm_y: float
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """RLP uses 2km grid with km-based coordinates.

        Filenames use km coordinates (e.g., 362, 5604).
        Formula: tile_index * km_per_tile
        Example: 362500m // 2000 = 181 → 181 * 2 = 362km
        """
        km_per_tile = RLP_GRID_SIZE // METERS_PER_KM  # 2
        grid_x = int(utm_x // RLP_GRID_SIZE) * km_per_tile
        grid_y = int(utm_y // RLP_GRID_SIZE) * km_per_tile
        return (grid_x, grid_y), (grid_x, grid_y)

    @property
    def jp2_feed_url(self) -> str:
        return self._jp2_feed_url

    @property
    def laz_feed_url(self) -> str:
        return self._laz_feed_url

    @property
    def verify_ssl(self) -> bool:
        return False  # RLP has SSL certificate issues

    def _parse_jp2_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse RLP JP2 feed using INSPIRE/Atom namespace."""
        jp2_tiles = {}
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for link_elem in root.findall('.//atom:link[@type="image/jp2"]', ns):
            url = link_elem.get("href")
            if url and url.endswith(".jp2"):
                filename = url.split("/")[-1]
                match = RLP_JP2_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"RLP JP2 '{filename}' doesn't match pattern. "
                        f"Expected: dop20rgb_32_XXX_YYYY_2_rp_YEAR.jp2"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                jp2_tiles[(grid_x, grid_y)] = url

        return jp2_tiles

    def _parse_laz_feed(
        self, session: requests.Session, root: ET.Element
    ) -> dict[tuple[int, int], str]:
        """Parse RLP LAZ feed using INSPIRE/Atom namespace."""
        laz_tiles = {}
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for link_elem in root.findall(".//atom:link", ns):
            url = link_elem.get("href")
            if url and url.endswith(".laz"):
                filename = url.split("/")[-1]
                match = RLP_LAZ_PATTERN.match(filename)
                if not match:
                    raise ValueError(
                        f"RLP LAZ '{filename}' doesn't match pattern. "
                        f"Expected: bdom20rgbi_32_XXX_YYYY_2_rp.laz"
                    )
                grid_x = int(match.group(1))
                grid_y = int(match.group(2))
                laz_tiles[(grid_x, grid_y)] = url

        return laz_tiles
