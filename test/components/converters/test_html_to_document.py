# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from haystack.components.converters import HTMLToDocument
from haystack.dataclasses import ByteStream


class TestHTMLToDocument:
    def test_serde(self):
        converter = HTMLToDocument(encoding="utf-8", extract_metadata=True)
        serde_data = converter.to_dict()
        assert serde_data["init_parameters"]["extract_metadata"] is True
        new_converter = HTMLToDocument.from_dict(serde_data)
        assert new_converter.extraction_kwargs == converter.extraction_kwargs
        assert new_converter.encoding == converter.encoding
        assert new_converter.extract_metadata is True

    def test_from_dict_without_extract_metadata(self):
        data = {
            "type": "haystack.components.converters.html.HTMLToDocument",
            "init_parameters": {"extraction_kwargs": {}, "store_full_path": False, "encoding": None},
        }
        converter = HTMLToDocument.from_dict(data)
        assert converter.extract_metadata is False

    @pytest.mark.parametrize("meta", [{"test": "TEST"}, [{"test": "TEST"}]], ids=["dict", "list"])
    def test_run(self, test_files_path, meta):
        """
        Test if the component runs correctly.
        """
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        converter = HTMLToDocument()
        results = converter.run(sources=sources, meta=meta)
        docs = results["documents"]
        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta["test"] == "TEST"

    def test_run_difficult_html(self, test_files_path):
        converter = HTMLToDocument()
        result = converter.run(sources=[Path(test_files_path / "html" / "paul_graham_superlinear.html")])

        assert len(result["documents"]) == 1
        assert "Superlinear" in result["documents"][0].content

    def test_run_detects_encoding(self):
        html = "<html><body><p>café</p></body></html>".encode("utf-16")
        converter = HTMLToDocument()

        result = converter.run(sources=[ByteStream(data=html)])

        assert [doc.content for doc in result["documents"]] == ["café"]

    def test_run_with_explicit_encoding(self):
        html = "<html><body><p>café</p></body></html>".encode("latin-1")
        converter = HTMLToDocument(encoding="latin-1")

        result = converter.run(sources=[ByteStream(data=html)])

        assert [doc.content for doc in result["documents"]] == ["café"]

    def test_mixed_sources_run(self, test_files_path):
        """
        Test if the component runs correctly if the input is a mix of paths and ByteStreams.
        """
        sources = [
            test_files_path / "html" / "what_is_haystack.html",
            str((test_files_path / "html" / "what_is_haystack.html").absolute()),
        ]
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as f:
            byte_stream = f.read()
            sources.append(ByteStream(byte_stream))

        converter = HTMLToDocument()
        results = converter.run(sources=sources)
        docs = results["documents"]
        assert len(docs) == 3
        for doc in docs:
            assert "Haystack" in doc.content

    @patch("haystack.components.converters.html.extract", return_value="test")
    def test_run_with_extraction_kwargs(self, mock_extract, test_files_path):
        sources = [test_files_path / "html" / "what_is_haystack.html"]

        converter = HTMLToDocument()
        converter.run(sources=sources)
        assert mock_extract.call_count == 1
        assert "favor_precision" not in mock_extract.call_args[1]

        precise_converter = HTMLToDocument(extraction_kwargs={"favor_precision": True})
        mock_extract.reset_mock()
        precise_converter.run(sources=sources)
        assert mock_extract.call_count == 1
        assert mock_extract.call_args[1]["favor_precision"] is True

    @pytest.mark.parametrize("store_full_path", [True, False])
    def test_run_with_store_full_path(self, test_files_path, store_full_path):
        """
        Test that file_path contains the full path or only the file name as configured.
        """
        converter = HTMLToDocument(store_full_path=store_full_path)
        sources = [test_files_path / "html" / "what_is_haystack.html"]

        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta["file_path"] == (str(sources[0]) if store_full_path else sources[0].name)

    def test_incorrect_meta(self, test_files_path):
        """
        Test if the component raises an error when incorrect metadata is supplied by the user.
        """
        converter = HTMLToDocument()
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        metadata = [{"file_name": "what_is_haystack.html"}, {"file_name": "haystack.html"}]
        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
            converter.run(sources=sources, meta=metadata)

    def test_run_bytestream_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url", "file_path": ""})

        results = converter.run(sources=[stream])
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {"content_type": "text/html", "url": "test_url", "file_path": ""}

    def test_run_bytestream_and_doc_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is no overlap between the metadata received.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url"})

        metadata = [{"file_name": "what_is_haystack.html"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {"file_name": "what_is_haystack.html", "content_type": "text/html", "url": "test_url"}

    def test_run_bytestream_doc_overlapping_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is an overlap between the metadata received.

        The component should use the supplied metadata to overwrite the values if there is an overlap between the keys.
        """
        converter = HTMLToDocument()
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            byte_stream = file.read()
            # ByteStream has "url" present in metadata
            stream = ByteStream(byte_stream, meta={"content_type": "text/html", "url": "test_url_correct"})

        # "url" supplied by the user overwrites value present in metadata
        metadata = [{"file_name": "what_is_haystack.html", "url": "test_url_new"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {
            "file_name": "what_is_haystack.html",
            "content_type": "text/html",
            "url": "test_url_new",
        }

    @pytest.mark.parametrize(
        ("encoding", "warning"), [(None, "No text could be extracted from"), ("utf-8", "Failed to extract text from")]
    )
    def test_run_skips_source_without_extractable_text(self, test_files_path, caplog, encoding, warning):
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = HTMLToDocument(encoding=encoding)
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)

        assert warning in caplog.text
        assert results["documents"] == []

    def test_run_error_handling(self, caplog):
        """
        Test if the component correctly handles errors.
        """
        sources: list[str | Path | ByteStream] = ["non_existing_file.html"]
        converter = HTMLToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.html" in caplog.text
            assert results["documents"] == []

    def test_run_empty_bytestream(self, caplog):
        """
        Test that an empty ByteStream is skipped without invoking extraction,
        so no noisy lxml parse errors are emitted.
        """
        empty_stream = ByteStream(data=b"", mime_type="text/html")
        converter = HTMLToDocument()

        with patch("haystack.components.converters.html.extract") as mock_extract:
            with caplog.at_level(logging.WARNING):
                results = converter.run(sources=[empty_stream])

        assert results["documents"] == []
        mock_extract.assert_not_called()
        assert "because it is empty" in caplog.text

    def test_run_with_extract_metadata(self, test_files_path):
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        default_doc = HTMLToDocument().run(sources=sources)["documents"][0]

        doc = HTMLToDocument(extract_metadata=True).run(sources=sources)["documents"][0]

        assert doc.content == default_doc.content
        assert doc.meta["file_path"] == "what_is_haystack.html"
        assert doc.meta["title"] == "What is Haystack? | Haystack"
        assert doc.meta["description"].startswith("Haystack is an open source Python framework")
        assert doc.meta["sitename"] == "Haystack"
        assert "author" not in doc.meta

    def test_run_with_extract_metadata_and_explicit_encoding(self):
        html = (
            "<html><head><title>Café guide</title><meta name='author' content='Zoë Müller'></head>"
            "<body><p>The best café in town.</p></body></html>"
        ).encode("latin-1")
        converter = HTMLToDocument(encoding="latin-1", extract_metadata=True)

        doc = converter.run(sources=[ByteStream(data=html)])["documents"][0]

        assert doc.content == "The best café in town."
        assert doc.meta["title"] == "Café guide"
        assert doc.meta["author"] == "Zoë Müller"

    def test_run_with_extract_metadata_user_meta_takes_precedence(self, test_files_path):
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            stream = ByteStream(file.read(), meta={"url": "test_url", "title": "Title from ByteStream"})
        converter = HTMLToDocument(extract_metadata=True)

        doc = converter.run(sources=[stream], meta={"title": "My own title"})["documents"][0]

        assert doc.meta["title"] == "My own title"
        assert doc.meta["url"] == "test_url"
        assert doc.meta["sitename"] == "Haystack"

    def test_run_with_extract_metadata_page_metadata_overrides_bytestream_meta(self, test_files_path):
        with open(test_files_path / "html" / "what_is_haystack.html", "rb") as file:
            stream = ByteStream(file.read(), meta={"title": "Title from ByteStream"})
        converter = HTMLToDocument(extract_metadata=True)

        doc = converter.run(sources=[stream])["documents"][0]

        assert doc.meta["title"] == "What is Haystack? | Haystack"

    @pytest.mark.parametrize(
        "mock_kwargs", [{"side_effect": ValueError("broken page")}, {"return_value": None}], ids=["error", "none"]
    )
    def test_run_with_extract_metadata_keeps_document_when_metadata_extraction_fails(
        self, test_files_path, caplog, mock_kwargs
    ):
        sources = [test_files_path / "html" / "what_is_haystack.html"]
        converter = HTMLToDocument(extract_metadata=True)

        with patch("haystack.components.converters.html.trafilatura_extract_metadata", **mock_kwargs):
            with caplog.at_level(logging.WARNING):
                docs = converter.run(sources=sources)["documents"]

        assert len(docs) == 1
        assert "Haystack" in docs[0].content
        assert docs[0].meta == {"file_path": "what_is_haystack.html"}
        if "side_effect" in mock_kwargs:
            assert "Could not extract page metadata from" in caplog.text

    def test_run_without_extract_metadata_does_not_call_trafilatura_extract_metadata(self, test_files_path):
        sources = [test_files_path / "html" / "what_is_haystack.html"]

        with patch("haystack.components.converters.html.trafilatura_extract_metadata") as mock_extract_metadata:
            docs = HTMLToDocument().run(sources=sources)["documents"]

        mock_extract_metadata.assert_not_called()
        assert docs[0].meta == {"file_path": "what_is_haystack.html"}
