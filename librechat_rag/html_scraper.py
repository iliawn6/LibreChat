"""
HTML scraping utilities for the LibreChat RAG project.

This module provides a small script to download and cache HTML pages from
``https://stallman.org``. The downloaded files are stored under
``data/html`` (configured via :mod:`librechat_rag.config`) and can then be
loaded as documents using :func:`librechat_rag.data_loading.load_html_docs`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Set

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from .config import HTML_DIR


def save_html(url: str, directory: Path) -> None:
    """
    Download a single HTML page and save it to ``directory``.

    The filename is derived from the URL path and stored without an explicit
    extension, mirroring the original approach.
    """

    response = requests.get(url)
    response.raise_for_status()

    filename = urlparse(url).path.strip("/").replace("/", "_") or "index"
    filepath = directory / filename

    directory.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as file:
        file.write(response.text)

    print(f"[html_scraper] Saved: {filepath}")


def is_internal_link(href: str) -> bool:
    """
    Determine whether a link should be treated as an internal Stallman.org HTML page.
    """

    parsed_href = urlparse(href)

    return (
        not href.startswith("#")  # not an in-page identifier
        and not href.startswith("tel:")  # not a telephone link
        and href.endswith(".html")  # only HTML pages
        and (
            parsed_href.netloc == ""  # relative link
            or "stallman.org" in parsed_href.netloc  # same domain
        )
    )


def extract_links(base_url: str) -> Set[str]:
    """
    Extract all internal HTML links from ``base_url`` on ``stallman.org``.
    """

    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if is_internal_link(href):
            full_url = urljoin(base_url, href)
            links.add(full_url)

    print(f"[html_scraper] Found {len(links)} internal HTML links")
    return links


def scrape_stallman_site(base_url: str = "https://stallman.org") -> None:
    """
    Download HTML content from ``stallman.org`` into :data:`HTML_DIR`.

    This function:

    1. Extracts internal HTML links from ``base_url``.
    2. Saves each linked page into the configured HTML directory.
    """

    directory = HTML_DIR
    links = extract_links(base_url)

    for link in links:
        save_html(link, directory)


if __name__ == "__main__":
    # Allow running this module as a script:
    #   python -m librechat_rag.html_scraper
    scrape_stallman_site()

