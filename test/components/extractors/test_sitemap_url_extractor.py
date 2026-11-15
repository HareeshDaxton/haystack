# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import logging
from unittest.mock import patch

import pytest

from haystack import Pipeline, component
from haystack.components.extractors import SitemapURLExtractor
from haystack.components.extractors import sitemap_url_extractor as sitemap_module
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream

URLSET = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/docs/intro</loc><lastmod>2026-10-01</lastmod></url>
  <url><loc>https://example.com/docs/install</loc></url>
  <url><loc>https://example.com/blog/hello</loc></url>
</urlset>"""

SITEMAP_INDEX = b"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-docs.xml</loc><lastmod>2026-10-01</lastmod></sitemap>
  <sitemap><loc>https://example.com/sitemap-blog.xml.gz</loc></sitemap>
</sitemapindex>"""


@component
class _FakeFetcher:
    """Returns fixed ByteStreams, like a LinkContentFetcher, without network access."""

    def __init__(self, streams: list[ByteStream]) -> None:
        self.streams = streams

    @component.output_types(streams=list[ByteStream])
    def run(self, urls: list[str]) -> dict[str, list[ByteStream]]:
        return {"streams": self.streams}


@component
class _UrlCollector:
    """Collects the URLs it receives, like a second LinkContentFetcher."""

    @component.output_types(received=list[str])
    def run(self, urls: list[str]) -> dict[str, list[str]]:
        return {"received": urls}


class TestSitemapURLExtractor:
    def test_init_default(self):
        extractor = SitemapURLExtractor()
        assert extractor.url_pattern is None

    def test_init_with_url_pattern(self):
        extractor = SitemapURLExtractor(url_pattern=r"/docs/")
        assert extractor.url_pattern == r"/docs/"

    def test_init_with_invalid_url_pattern_raises(self):
        with pytest.raises(ValueError, match="Invalid url_pattern"):
            SitemapURLExtractor(url_pattern="[unclosed")

    def test_to_dict(self):
        extractor = SitemapURLExtractor(url_pattern=r"/docs/")
        assert component_to_dict(extractor, "extractor") == {
            "type": "haystack.components.extractors.sitemap_url_extractor.SitemapURLExtractor",
            "init_parameters": {"url_pattern": r"/docs/"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.extractors.sitemap_url_extractor.SitemapURLExtractor",
            "init_parameters": {"url_pattern": r"/docs/"},
        }
        extractor = component_from_dict(SitemapURLExtractor, data, "extractor")
        assert extractor.url_pattern == r"/docs/"
        assert extractor.run(sources=[ByteStream(data=URLSET)])["urls"] == [
            "https://example.com/docs/intro",
            "https://example.com/docs/install",
        ]

    def test_run_urlset(self):
        result = SitemapURLExtractor().run(sources=[ByteStream(data=URLSET)])
        assert result == {
            "urls": [
                "https://example.com/docs/intro",
                "https://example.com/docs/install",
                "https://example.com/blog/hello",
            ],
            "sitemap_urls": [],
        }

    def test_run_sitemap_index(self):
        result = SitemapURLExtractor().run(sources=[ByteStream(data=SITEMAP_INDEX)])
        assert result == {
            "urls": [],
            "sitemap_urls": ["https://example.com/sitemap-docs.xml", "https://example.com/sitemap-blog.xml.gz"],
        }

    def test_run_mixed_sources(self):
        result = SitemapURLExtractor().run(sources=[ByteStream(data=SITEMAP_INDEX), ByteStream(data=URLSET)])
        assert len(result["urls"]) == 3
        assert len(result["sitemap_urls"]) == 2

    def test_run_with_url_pattern_filters_only_page_urls(self):
        extractor = SitemapURLExtractor(url_pattern=r"/docs/")
        result = extractor.run(sources=[ByteStream(data=URLSET), ByteStream(data=SITEMAP_INDEX)])
        assert result["urls"] == ["https://example.com/docs/intro", "https://example.com/docs/install"]
        # nested sitemaps are kept even if they don't match, so their pages are not lost
        assert result["sitemap_urls"] == [
            "https://example.com/sitemap-docs.xml",
            "https://example.com/sitemap-blog.xml.gz",
        ]

    def test_run_url_pattern_matches_anywhere(self):
        extractor = SitemapURLExtractor(url_pattern=r"install$")
        result = extractor.run(sources=[ByteStream(data=URLSET)])
        assert result["urls"] == ["https://example.com/docs/install"]

    def test_run_removes_duplicates_and_keeps_order(self):
        second = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/new</loc></url>
          <url><loc>https://example.com/docs/intro</loc></url>
          <url><loc>https://example.com/new</loc></url>
        </urlset>"""
        result = SitemapURLExtractor().run(sources=[ByteStream(data=URLSET), ByteStream(data=second)])
        assert result["urls"] == [
            "https://example.com/docs/intro",
            "https://example.com/docs/install",
            "https://example.com/blog/hello",
            "https://example.com/new",
        ]

    def test_run_strips_whitespace_and_skips_empty_loc(self):
        sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>
            https://example.com/a
          </loc></url>
          <url><loc></loc></url>
          <url><loc>   </loc></url>
          <url><lastmod>2026-10-01</lastmod></url>
        </urlset>"""
        result = SitemapURLExtractor().run(sources=[ByteStream(data=sitemap)])
        assert result["urls"] == ["https://example.com/a"]

    def test_run_decodes_xml_escaped_urls(self):
        sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/search?q=haystack&amp;page=2</loc></url>
        </urlset>"""
        result = SitemapURLExtractor().run(sources=[ByteStream(data=sitemap)])
        assert result["urls"] == ["https://example.com/search?q=haystack&page=2"]

    def test_run_without_namespace(self):
        sitemap = b"<urlset><url><loc>https://example.com/a</loc></url></urlset>"
        result = SitemapURLExtractor().run(sources=[ByteStream(data=sitemap)])
        assert result["urls"] == ["https://example.com/a"]

    def test_run_ignores_extension_locs(self):
        sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
                  xmlns:image="http://www.google.com/schemas/sitemap-image/1.1"
                  xmlns:video="http://www.google.com/schemas/sitemap-video/1.1">
          <url>
            <loc>https://example.com/page</loc>
            <image:image><image:loc>https://example.com/logo.png</image:loc></image:image>
            <video:video><video:content_loc>https://example.com/v.mp4</video:content_loc></video:video>
          </url>
        </urlset>"""
        result = SitemapURLExtractor().run(sources=[ByteStream(data=sitemap)])
        assert result["urls"] == ["https://example.com/page"]

    def test_run_skips_non_http_urls(self, caplog):
        sitemap = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/ok</loc></url>
          <url><loc>http://example.com/also-ok</loc></url>
          <url><loc>file:///etc/passwd</loc></url>
          <url><loc>/relative/path</loc></url>
          <url><loc>https://</loc></url>
        </urlset>"""
        stream = ByteStream(data=sitemap, meta={"url": "https://example.com/sitemap.xml"})
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[stream])
        assert result["urls"] == ["https://example.com/ok", "http://example.com/also-ok"]
        assert "Skipped 3 entries" in caplog.text
        assert "https://example.com/sitemap.xml" in caplog.text

    def test_run_gzip(self):
        result = SitemapURLExtractor().run(sources=[ByteStream(data=gzip.compress(URLSET))])
        assert len(result["urls"]) == 3

    def test_run_gzip_sitemap_index(self):
        result = SitemapURLExtractor().run(sources=[ByteStream(data=gzip.compress(SITEMAP_INDEX))])
        assert len(result["sitemap_urls"]) == 2

    def test_run_broken_gzip_is_skipped(self, caplog):
        broken = gzip.compress(URLSET)[:20]
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=broken), ByteStream(data=URLSET)])
        assert len(result["urls"]) == 3
        assert "Could not decompress sitemap" in caplog.text

    def test_run_too_large_sitemap_is_skipped(self, caplog):
        with patch.object(sitemap_module, "MAX_SITEMAP_SIZE", 100), caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=URLSET)])
        assert result == {"urls": [], "sitemap_urls": []}
        assert "larger than 100 bytes" in caplog.text

    def test_run_too_large_gzip_sitemap_is_skipped(self, caplog):
        # small compressed size, but unpacks to more than the limit
        with patch.object(sitemap_module, "MAX_SITEMAP_SIZE", 100), caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=gzip.compress(URLSET))])
        assert result == {"urls": [], "sitemap_urls": []}
        assert "larger than 100 bytes" in caplog.text

    def test_run_doctype_is_skipped(self, caplog):
        billion_laughs = b"""<?xml version="1.0"?>
        <!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>
        <urlset><url><loc>https://example.com/&lol2;</loc></url></urlset>"""
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=billion_laughs)])
        assert result == {"urls": [], "sitemap_urls": []}
        assert "DOCTYPE" in caplog.text

    def test_run_empty_source_is_skipped(self, caplog):
        # LinkContentFetcher returns an empty ByteStream when a fetch fails and raise_on_failure=False
        failed = ByteStream(data=b"", meta={"url": "https://example.com/missing.xml"})
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[failed, ByteStream(data=URLSET)])
        assert len(result["urls"]) == 3
        assert "https://example.com/missing.xml is empty" in caplog.text

    def test_run_invalid_xml_is_skipped(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=b"<html>not xml"), ByteStream(data=URLSET)])
        assert len(result["urls"]) == 3
        assert "Could not parse sitemap" in caplog.text

    def test_run_xml_that_is_not_a_sitemap_is_skipped(self, caplog):
        rss = b"<rss><channel><item><link>https://example.com/a</link></item></channel></rss>"
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[ByteStream(data=rss)])
        assert result == {"urls": [], "sitemap_urls": []}
        assert "got <rss>" in caplog.text

    def test_run_log_does_not_contain_content(self, caplog):
        with caplog.at_level(logging.WARNING):
            SitemapURLExtractor().run(sources=[ByteStream(data=b"<html>secret page content")])
        assert "ByteStream without URL" in caplog.text
        assert "secret page content" not in caplog.text

    def test_run_file_paths(self, tmp_path):
        xml_path = tmp_path / "sitemap.xml"
        xml_path.write_bytes(URLSET)
        gz_path = tmp_path / "sitemap-index.xml.gz"
        gz_path.write_bytes(gzip.compress(SITEMAP_INDEX))

        result = SitemapURLExtractor().run(sources=[xml_path, str(gz_path)])
        assert len(result["urls"]) == 3
        assert len(result["sitemap_urls"]) == 2

    def test_run_missing_file_is_skipped(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            result = SitemapURLExtractor().run(sources=[tmp_path / "missing.xml"])
        assert result == {"urls": [], "sitemap_urls": []}
        assert "Could not read" in caplog.text

    def test_run_no_sources(self):
        assert SitemapURLExtractor().run(sources=[]) == {"urls": [], "sitemap_urls": []}

    def test_run_does_not_change_input(self):
        stream = ByteStream(data=gzip.compress(URLSET), meta={"url": "https://example.com/sitemap.xml.gz"})
        original_data = stream.data
        SitemapURLExtractor().run(sources=[stream])
        assert stream.data == original_data
        assert stream.meta == {"url": "https://example.com/sitemap.xml.gz"}

    def test_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("sitemap_fetcher", _FakeFetcher(streams=[ByteStream(data=URLSET)]))
        pipeline.add_component("sitemap", SitemapURLExtractor(url_pattern=r"/docs/"))
        pipeline.add_component("page_fetcher", _UrlCollector())
        pipeline.connect("sitemap_fetcher.streams", "sitemap.sources")
        pipeline.connect("sitemap.urls", "page_fetcher.urls")

        result = pipeline.run({"sitemap_fetcher": {"urls": ["https://example.com/sitemap.xml"]}})
        assert result["page_fetcher"]["received"] == [
            "https://example.com/docs/intro",
            "https://example.com/docs/install",
        ]
        assert result["sitemap"]["sitemap_urls"] == []

    def test_pipeline_serialization_roundtrip(self):
        pipeline = Pipeline()
        pipeline.add_component("sitemap", SitemapURLExtractor(url_pattern=r"/docs/"))
        loaded = Pipeline.loads(pipeline.dumps())
        assert loaded.get_component("sitemap").url_pattern == r"/docs/"
