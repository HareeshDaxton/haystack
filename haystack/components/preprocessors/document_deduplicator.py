# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from math import inf
from typing import Any

from haystack import Document, component, default_to_dict, logging
from haystack.utils.misc import _make_metadata_value_hashable

logger = logging.getLogger(__name__)


@component
class DocumentDeduplicator:
    """
    Removes duplicate documents by comparing their content or metadata fields, not their IDs.

    A document ID is built from both its content and its metadata. Two documents with the same text but different
    metadata, such as the same page crawled under two URLs, have different IDs.
    `DocumentWriter` and `DocumentJoiner` compare IDs, so they keep both copies.
    `DocumentDeduplicator` compares the fields you choose and keeps one document per unique value.

    When there are duplicates, it keeps the one with the highest `score`.
    If scores are equal or missing, it keeps the first one.
    The kept document takes the position of the first copy, so the order of the list doesn't change.

    Use it before a `DocumentWriter` in an indexing pipeline, or after a `DocumentJoiner` in a query pipeline.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import DocumentDeduplicator

    documents = [
        Document(content="Haystack is an open-source AI framework.", meta={"url": "https://example.com/a"}),
        Document(content="Haystack is an open-source AI framework.", meta={"url": "https://example.com/a?ref=1"}),
        Document(content="Pipelines connect components.", meta={"url": "https://example.com/b"}),
    ]

    deduplicator = DocumentDeduplicator()
    result = deduplicator.run(documents=documents)

    print([doc.meta["url"] for doc in result["documents"]])
    # ['https://example.com/a', 'https://example.com/b']
    print([doc.meta["url"] for doc in result["duplicates"]])
    # ['https://example.com/a?ref=1']
    ```
    """

    def __init__(self, keys: list[str] | None = None) -> None:
        """
        Creates a DocumentDeduplicator component.

        :param keys:
            The document fields that together decide if two documents are duplicates.
            Use `"content"` for the document text, or `"meta.<key>"` for a metadata field.
            Nested metadata fields are supported, for example `"meta.source.url"`.
            Two documents are duplicates only if all keys have equal values.
            A document where any key is missing or `None` is never treated as a duplicate.
            Defaults to `["content"]`.
        :raises ValueError:
            If `keys` is an empty list or contains an unsupported key.
        """
        keys = ["content"] if keys is None else keys
        if not keys:
            raise ValueError("'keys' must contain at least one key, for example 'content' or 'meta.url'.")
        for key in keys:
            if key != "content" and not (key.startswith("meta.") and len(key) > len("meta.")):
                raise ValueError(f"Unsupported key: '{key}'. Use 'content' or 'meta.<key>', for example 'meta.url'.")
        self.keys = keys

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, keys=self.keys)

    @staticmethod
    def _get_value(document: Document, key: str) -> Any:
        """
        Get the value of one key from a document, or `None` if the field is missing.
        """
        if key == "content":
            return document.content
        value: Any = document.meta
        for part in key[len("meta.") :].split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return _make_metadata_value_hashable(value)

    def _get_dedup_key(self, document: Document) -> tuple | None:
        """
        Build the value that identifies duplicates, or `None` if any key is missing.
        """
        values = tuple(self._get_value(document, key) for key in self.keys)
        if any(value is None for value in values):
            return None
        return values

    @component.output_types(documents=list[Document], duplicates=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Removes duplicate documents from the list.

        :param documents:
            The documents to deduplicate.
        :returns:
            A dictionary with the following keys:
            - `documents`: The unique documents, in the order of their first appearance.
            - `duplicates`: The documents that were removed, in the order they were found.
        :raises TypeError:
            If `documents` is not a list of Documents.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentDeduplicator expects a List of Documents as input.")

        kept: list[Document] = []
        position_by_key: dict[tuple, int] = {}
        duplicates: list[Document] = []

        for document in documents:
            dedup_key = self._get_dedup_key(document)
            if dedup_key is None:
                kept.append(document)
                continue

            position = position_by_key.get(dedup_key)
            if position is None:
                position_by_key[dedup_key] = len(kept)
                kept.append(document)
                continue

            current = kept[position]
            current_score = current.score if current.score is not None else -inf
            new_score = document.score if document.score is not None else -inf
            if new_score > current_score:
                kept[position] = document
                duplicates.append(current)
            else:
                duplicates.append(document)

        if duplicates:
            logger.debug(
                "Removed {removed} duplicate documents out of {total}.", removed=len(duplicates), total=len(documents)
            )

        return {"documents": kept, "duplicates": duplicates}
