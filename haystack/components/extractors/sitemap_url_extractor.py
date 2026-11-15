# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source
from haystack.core.serialization import default_to_dict
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

# The sitemap protocol limits a sitemap to 50 MB (uncompressed). Larger files, including gzip files that unpack to
# more than this, are skipped. See https://www.sitemaps.org/protocol.html
MAX_SITEMAP_SIZE = 50 * 1024 * 1024

_GZIP_MAGIC_NUMBER = b"\x1f\x8b"
_DOCTYPE_PATTERN = re.compile(rb"<!DOCTYPE", re.IGNORECASE)


@component
class SitemapURLExtractor:
    """
    Extracts page URLs from XML sitemaps.

    A sitemap (`sitemap.xml`) lists the pages of a website. This component reads sitemaps and returns their URLs,
    so you can fetch and index a whole website: connect a `LinkContentFetcher` that downloads the sitemap to this
    component, and connect its `urls` output to a second `LinkContentFetcher` that downloads the pages.

    It supports both kinds of sitemaps:
    - a URL set (`<urlset>`), which lists pages. Its URLs are returned in the `urls` output.
    - a sitemap index (`<sitemapindex>`), which lists other sitemaps. Its URLs are returned in the `sitemap_urls`
      output, so you can fetch and extract them in a second step.

    Gzip-compressed sitemaps (`sitemap.xml.gz`) are decompressed automatically.
    The component doesn't make any HTTP requests itself.

    ### Usage example

    ```python
    from haystack.components.extractors import SitemapURLExtractor
    from haystack.dataclasses import ByteStream

    sitemap = ByteStream(
        data=b'''<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/docs/intro</loc></url>
      <url><loc>https://example.com/docs/install</loc></url>
      <url><loc>https://example.com/blog/hello</loc></url>
    </urlset>'''
    )

    extractor = SitemapURLExtractor(url_pattern=r"/docs/")
    result = extractor.run(sources=[sitemap])
    print(result["urls"])
    # ['https://example.com/docs/intro', 'https://example.com/docs/install']
    ```
    """

    def __init__(self, url_pattern: str | None = None) -> None:
        """
        Creates an instance of SitemapURLExtractor.

        :param url_pattern:
            An optional regular expression. If set, only page URLs that match it (anywhere in the URL) are returned
            in `urls`, for example `r"/docs/"`. URLs in `sitemap_urls` are never filtered, so you don't lose pages
            listed in a nested sitemap.
        :raises ValueError:
            If `url_pattern` is not a valid regular expression.
        """
        self.url_pattern = url_pattern
        try:
            self._compiled_pattern = re.compile(url_pattern) if url_pattern is not None else None
        except re.error as error:
            raise ValueError(f"Invalid url_pattern {url_pattern!r}: {error}") from error

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, url_pattern=self.url_pattern)

    @component.output_types(urls=list[str], sitemap_urls=list[str])
    def run(self, sources: list[str | Path | ByteStream]) -> dict[str, list[str]]:
        """
        Extracts URLs from the given sitemaps.

        Sources that can't be read, are empty, are not valid XML, or are not sitemaps are skipped with a warning.

        :param sources:
            List of sitemaps as file paths or ByteStream objects, for example the `streams` output of a
            `LinkContentFetcher`.

        :returns:
            A dictionary with the following keys:
            - `urls`: The page URLs found in URL set sitemaps, filtered by `url_pattern`, without duplicates,
              in the order they appear.
            - `sitemap_urls`: The sitemap URLs found in sitemap index files, without duplicates, in the order
              they appear.
        """
        urls: dict[str, None] = {}
        sitemap_urls: dict[str, None] = {}

        for source in sources:
            source_name = _get_source_name(source)
            root = self._parse_sitemap(source, source_name)
            if root is None:
                continue

            if _local_name(root.tag) == "sitemapindex":
                for url in self._extract_locs(root, entry_tag="sitemap", source_name=source_name):
                    sitemap_urls[url] = None
            else:
                for url in self._extract_locs(root, entry_tag="url", source_name=source_name):
                    if self._compiled_pattern is None or self._compiled_pattern.search(url):
                        urls[url] = None

        logger.debug(
            "Extracted {urls_count} page URLs and {sitemap_urls_count} sitemap URLs from {sources_count} sources.",
            urls_count=len(urls),
            sitemap_urls_count=len(sitemap_urls),
            sources_count=len(sources),
        )
        return {"urls": list(urls), "sitemap_urls": list(sitemap_urls)}

    @staticmethod
    def _parse_sitemap(source: str | Path | ByteStream, source_name: str) -> ElementTree.Element | None:
        """
        Reads a source and returns the root element of the sitemap, or `None` if the source must be skipped.
        """
        data = SitemapURLExtractor._read_data(source, source_name)
        if data is None:
            return None

        # Sitemaps never need a DTD. Refusing them blocks entity expansion attacks ("billion laughs").
        if _DOCTYPE_PATTERN.search(data):
            logger.warning("Sitemap {source} contains a DOCTYPE declaration. Skipping it.", source=source_name)
            return None

        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError as error:
            logger.warning(
                "Could not parse sitemap {source}. Skipping it. Error: {error}", source=source_name, error=error
            )
            return None

        if _local_name(root.tag) not in ("urlset", "sitemapindex"):
            logger.warning(
                "{source} is not a sitemap: expected a <urlset> or <sitemapindex> root element, got <{tag}>. "
                "Skipping it.",
                source=source_name,
                tag=_local_name(root.tag),
            )
            return None

        return root

    @staticmethod
    def _read_data(source: str | Path | ByteStream, source_name: str) -> bytes | None:
        """
        Returns the uncompressed bytes of a source, or `None` if it can't be read, is empty, or is too large.
        """
        try:
            data = get_bytestream_from_source(source).data
        except Exception as error:
            logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source_name, error=error)
            return None

        if not data:
            # LinkContentFetcher returns an empty ByteStream when a fetch fails and raise_on_failure is False
            logger.warning("Sitemap {source} is empty. Skipping it.", source=source_name)
            return None

        if data.startswith(_GZIP_MAGIC_NUMBER):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as gzip_file:
                    # read one byte more than the limit to detect files that are too large without unpacking all
                    data = gzip_file.read(MAX_SITEMAP_SIZE + 1)
            except (OSError, EOFError) as error:
                logger.warning(
                    "Could not decompress sitemap {source}. Skipping it. Error: {error}",
                    source=source_name,
                    error=error,
                )
                return None

        if len(data) > MAX_SITEMAP_SIZE:
            logger.warning(
                "Sitemap {source} is larger than {max_size} bytes, the limit of the sitemap protocol. Skipping it.",
                source=source_name,
                max_size=MAX_SITEMAP_SIZE,
            )
            return None

        return data

    @staticmethod
    def _extract_locs(root: ElementTree.Element, entry_tag: str, source_name: str) -> list[str]:
        """
        Returns the `<loc>` values of the direct `<url>` or `<sitemap>` entries of a sitemap.

        Only direct entries are read, so URLs from sitemap extensions such as `<image:loc>` are ignored.
        Values that are not absolute HTTP or HTTPS URLs are skipped.
        """
        locs = []
        invalid_count = 0
        # {*} matches any XML namespace, including no namespace
        for loc in root.iterfind(f"{{*}}{entry_tag}/{{*}}loc"):
            url = (loc.text or "").strip()
            if not url:
                continue
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                invalid_count += 1
                continue
            locs.append(url)

        if invalid_count:
            logger.warning(
                "Skipped {invalid_count} entries in sitemap {source} that are not absolute HTTP or HTTPS URLs.",
                invalid_count=invalid_count,
                source=source_name,
            )
        return locs


def _get_source_name(source: str | Path | ByteStream) -> str:
    """
    Returns a short name for a source to use in log messages: its URL or file path, never its content.
    """
    if isinstance(source, ByteStream):
        return str(source.meta.get("url") or source.meta.get("file_path") or "ByteStream without URL")
    return str(source)


def _local_name(tag: str) -> str:
    """
    Returns an XML tag name without its namespace, for example `urlset` for `{http://...}urlset`.
    """
    return tag.rsplit("}", 1)[-1]
