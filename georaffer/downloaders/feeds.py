"""Shared utilities for fetching XML feeds."""

import time
import xml.etree.ElementTree as ET

import requests

from georaffer.config import FEED_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_BASE, RETRY_MAX_WAIT


def fetch_xml_feed(
    session: requests.Session,
    url: str,
    timeout: float = FEED_TIMEOUT,
    wrap_content: bool = False,
) -> ET.Element:
    """Fetch XML feed with retries and exponential backoff.

    Args:
        session: HTTP session for requests
        url: Feed URL to fetch
        timeout: Request timeout in seconds
        wrap_content: If True, wrap content in <root> tags (for feeds without root element)

    Returns:
        Parsed XML root element

    Raises:
        RuntimeError: If fetch fails after MAX_RETRIES attempts
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = min(RETRY_BACKOFF_BASE ** (attempt - 1), RETRY_MAX_WAIT)
                time.sleep(delay)

            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            content = response.content
            if wrap_content:
                content = b"<root>" + content + b"</root>"

            return ET.fromstring(content)

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to fetch feed {url} after {MAX_RETRIES} retries: {last_error}")
