# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

with LazyImport("Run 'pip install trafilatura'") as trafilatura_import:
    from trafilatura import extract
    from trafilatura import extract_metadata as trafilatura_extract_metadata

# Page metadata fields copied from Trafilatura into the Document meta when `extract_metadata=True`.
_PAGE_METADATA_FIELDS = ("title", "author", "date", "description", "sitename")


@component
class HTMLToDocument:
    """
    Converts an HTML file to a Document.

    Usage example:
    ```python
    from haystack.components.converters import HTMLToDocument

    converter = HTMLToDocument()
    results = converter.run(sources=["test/test_files/html/paul_graham_superlinear.html"])
    documents = results["documents"]

    print(documents[0].content)
    # >> 'This is a text from the HTML file.'
    ```

    To also store the page title, author, date, description and site name in the Document metadata:
    ```python
    converter = HTMLToDocument(extract_metadata=True)
    results = converter.run(sources=["test/test_files/html/paul_graham_superlinear.html"])

    print(results["documents"][0].meta)
    # >> {'file_path': 'paul_graham_superlinear.html', 'title': 'Superlinear Returns', 'date': '2023-01-01'}
    ```
    """

    def __init__(
        self,
        extraction_kwargs: dict[str, Any] | None = None,
        store_full_path: bool = False,
        encoding: str | None = None,
        *,
        extract_metadata: bool = False,
    ) -> None:
        """
        Create an HTMLToDocument component.

        :param extraction_kwargs: A dictionary containing keyword arguments to customize the extraction process. These
            are passed to the underlying Trafilatura `extract` function. For the full list of available arguments, see
            the [Trafilatura documentation](https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract).
        :param store_full_path:
        If True, the full path of the file is stored in the metadata of the document.
        If False, only the file name is stored.
        :param encoding:
            The encoding to use when decoding HTML files. If None (the default), Trafilatura detects the encoding.
            If detection produces incorrect characters, install `trafilatura[all]` for additional encoding detection
            support, or set this parameter if you know the encoding.
        :param extract_metadata:
            If True, the page metadata found in the HTML (title, author, date, description and site name) is added to
            the metadata of the document. Fields that can't be found are left out. The document content is not
            changed. Values passed in `meta` take precedence over the page metadata.
        """
        trafilatura_import.check()

        self.extraction_kwargs = extraction_kwargs or {}
        self.store_full_path = store_full_path
        self.encoding = encoding
        self.extract_metadata = extract_metadata

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            extraction_kwargs=self.extraction_kwargs,
            store_full_path=self.store_full_path,
            encoding=self.encoding,
            extract_metadata=self.extract_metadata,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HTMLToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
        extraction_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Converts a list of HTML files to Documents.

        :param sources:
            List of HTML file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.
        :param extraction_kwargs:
            Additional keyword arguments to customize the extraction process.

        :returns:
            A dictionary with the following keys:
            - `documents`: Created Documents
        """

        merged_extraction_kwargs = {**self.extraction_kwargs, **(extraction_kwargs or {})}

        documents = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            if not bytestream.data:
                logger.warning("Skipping {source} because it is empty.", source=source)
                continue

            try:
                html_bytes = bytestream.data
                html_content = html_bytes.decode(self.encoding) if self.encoding is not None else html_bytes
                text = extract(html_content, **merged_extraction_kwargs)
            except Exception as conversion_e:
                logger.warning(
                    "Failed to extract text from {source}. Skipping it. Error: {error}",
                    source=source,
                    error=conversion_e,
                )
                continue

            if text is None:
                logger.warning("No text could be extracted from {source}. Skipping it.", source=source)
                continue

            page_metadata = self._extract_page_metadata(html_content, source) if self.extract_metadata else {}
            merged_metadata = {**bytestream.meta, **page_metadata, **metadata}

            if not self.store_full_path and "file_path" in bytestream.meta:
                file_path = bytestream.meta.get("file_path")
                if file_path:  # Ensure the value is not None for mypy
                    merged_metadata["file_path"] = os.path.basename(file_path)

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}

    @staticmethod
    def _extract_page_metadata(html_content: str | bytes, source: str | Path | ByteStream) -> dict[str, Any]:
        """
        Extracts the page metadata (title, author, date, description and site name) from an HTML page.

        :param html_content: The HTML page, as a string or as bytes.
        :param source: The source of the HTML page, used in the warning if the extraction fails.
        :returns: A dictionary with the metadata fields that were found. Missing fields are left out.
        """
        try:
            extracted = trafilatura_extract_metadata(html_content)
        except Exception as error:
            logger.warning("Could not extract page metadata from {source}. Error: {error}", source=source, error=error)
            return {}

        if extracted is None:
            return {}

        page_metadata = extracted.as_dict()
        return {field: page_metadata[field] for field in _PAGE_METADATA_FIELDS if page_metadata.get(field)}
