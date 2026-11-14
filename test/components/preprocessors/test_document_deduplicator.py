# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from haystack import Document, Pipeline, component
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentDeduplicator
from haystack.components.writers import DocumentWriter
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


@component
class _FixedRetriever:
    """Returns a fixed list of documents, like a retriever with known results."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, list[Document]]:
        return {"documents": self.documents}


class TestDocumentDeduplicator:
    def test_init_default(self):
        deduplicator = DocumentDeduplicator()
        assert deduplicator.keys == ["content"]

    def test_init_with_keys(self):
        deduplicator = DocumentDeduplicator(keys=["content", "meta.url"])
        assert deduplicator.keys == ["content", "meta.url"]

    def test_init_with_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            DocumentDeduplicator(keys=[])

    @pytest.mark.parametrize("key", ["id", "score", "meta", "meta.", "url", "metadata.url", ""])
    def test_init_with_unsupported_key_raises(self, key):
        with pytest.raises(ValueError, match="Unsupported key"):
            DocumentDeduplicator(keys=[key])

    def test_to_dict(self):
        deduplicator = DocumentDeduplicator(keys=["meta.url"])
        assert component_to_dict(deduplicator, "deduplicator") == {
            "type": "haystack.components.preprocessors.document_deduplicator.DocumentDeduplicator",
            "init_parameters": {"keys": ["meta.url"]},
        }

    def test_to_dict_default(self):
        assert component_to_dict(DocumentDeduplicator(), "deduplicator") == {
            "type": "haystack.components.preprocessors.document_deduplicator.DocumentDeduplicator",
            "init_parameters": {"keys": ["content"]},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.preprocessors.document_deduplicator.DocumentDeduplicator",
            "init_parameters": {"keys": ["content", "meta.file_path"]},
        }
        deduplicator = component_from_dict(DocumentDeduplicator, data, "deduplicator")
        assert deduplicator.keys == ["content", "meta.file_path"]

    def test_from_dict_without_init_parameters(self):
        data = {"type": "haystack.components.preprocessors.document_deduplicator.DocumentDeduplicator"}
        deduplicator = component_from_dict(DocumentDeduplicator, data, "deduplicator")
        assert deduplicator.keys == ["content"]

    def test_run_empty_list(self):
        assert DocumentDeduplicator().run(documents=[]) == {"documents": [], "duplicates": []}

    def test_run_wrong_input_type(self):
        deduplicator = DocumentDeduplicator()
        with pytest.raises(TypeError, match="expects a List of Documents"):
            deduplicator.run(documents="not a list")
        with pytest.raises(TypeError, match="expects a List of Documents"):
            deduplicator.run(documents=["not a document"])

    def test_run_no_duplicates(self):
        docs = [Document(content="a"), Document(content="b"), Document(content="c")]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == docs
        assert result["duplicates"] == []

    def test_run_same_content_different_meta(self):
        docs = [
            Document(content="Haystack is a framework.", meta={"url": "https://example.com/a"}),
            Document(content="Haystack is a framework.", meta={"url": "https://example.com/a?ref=1"}),
            Document(content="Pipelines connect components.", meta={"url": "https://example.com/b"}),
        ]
        assert docs[0].id != docs[1].id

        result = DocumentDeduplicator().run(documents=docs)

        assert result["documents"] == [docs[0], docs[2]]
        assert result["duplicates"] == [docs[1]]

    def test_run_keeps_first_when_no_scores(self):
        docs = [Document(content="same", meta={"n": i}) for i in range(4)]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[0]]
        assert result["duplicates"] == docs[1:]

    def test_run_keeps_first_when_scores_are_equal(self):
        docs = [Document(content="same", meta={"n": 1}, score=0.5), Document(content="same", meta={"n": 2}, score=0.5)]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[0]]
        assert result["duplicates"] == [docs[1]]

    def test_run_keeps_highest_score_at_first_position(self):
        docs = [
            Document(content="same", meta={"source": "bm25"}, score=0.4),
            Document(content="other", score=0.7),
            Document(content="same", meta={"source": "embedding"}, score=0.9),
        ]
        result = DocumentDeduplicator().run(documents=docs)

        assert result["documents"] == [docs[2], docs[1]]
        assert result["duplicates"] == [docs[0]]

    def test_run_scored_document_beats_unscored(self):
        docs = [Document(content="same", meta={"n": 1}), Document(content="same", meta={"n": 2}, score=0.1)]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[1]]
        assert result["duplicates"] == [docs[0]]

    def test_run_unscored_document_does_not_beat_scored(self):
        docs = [Document(content="same", meta={"n": 1}, score=-5.0), Document(content="same", meta={"n": 2})]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[0]]
        assert result["duplicates"] == [docs[1]]

    def test_run_highest_score_across_many_copies(self):
        docs = [
            Document(content="same", meta={"n": 1}, score=0.2),
            Document(content="same", meta={"n": 2}, score=0.8),
            Document(content="same", meta={"n": 3}, score=0.5),
        ]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[1]]
        assert result["duplicates"] == [docs[0], docs[2]]

    def test_run_is_case_and_whitespace_sensitive(self):
        docs = [Document(content="Hello world"), Document(content="hello world"), Document(content="Hello  world")]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == docs
        assert result["duplicates"] == []

    def test_run_empty_content_is_a_value(self):
        docs = [Document(content="", meta={"n": 1}), Document(content="", meta={"n": 2})]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == [docs[0]]
        assert result["duplicates"] == [docs[1]]

    def test_run_none_content_is_never_a_duplicate(self):
        docs = [Document(content=None, meta={"n": 1}), Document(content=None, meta={"n": 2})]
        result = DocumentDeduplicator().run(documents=docs)
        assert result["documents"] == docs
        assert result["duplicates"] == []

    def test_run_by_meta_field(self):
        docs = [
            Document(content="Version 1 of the page", meta={"url": "https://example.com/a"}),
            Document(content="Version 2 of the page", meta={"url": "https://example.com/a"}),
            Document(content="Another page", meta={"url": "https://example.com/b"}),
        ]
        result = DocumentDeduplicator(keys=["meta.url"]).run(documents=docs)
        assert result["documents"] == [docs[0], docs[2]]
        assert result["duplicates"] == [docs[1]]

    def test_run_missing_meta_field_is_never_a_duplicate(self):
        docs = [
            Document(content="a"),
            Document(content="b"),
            Document(content="c", meta={"url": None}),
            Document(content="d", meta={"url": None}),
        ]
        result = DocumentDeduplicator(keys=["meta.url"]).run(documents=docs)
        assert result["documents"] == docs
        assert result["duplicates"] == []

    def test_run_by_nested_meta_field(self):
        docs = [
            Document(content="a", meta={"source": {"url": "https://example.com/a"}}),
            Document(content="b", meta={"source": {"url": "https://example.com/a"}}),
            Document(content="c", meta={"source": "not a dict"}),
            Document(content="d", meta={"source": {"title": "no url"}}),
        ]
        result = DocumentDeduplicator(keys=["meta.source.url"]).run(documents=docs)
        assert result["documents"] == [docs[0], docs[2], docs[3]]
        assert result["duplicates"] == [docs[1]]

    def test_run_by_meta_field_with_unhashable_values(self):
        docs = [
            Document(content="a", meta={"tags": ["x", "y"], "info": {"lang": "en"}}),
            Document(content="b", meta={"tags": ["x", "y"], "info": {"lang": "en"}}),
            Document(content="c", meta={"tags": ["y", "x"], "info": {"lang": "en"}}),
        ]
        result = DocumentDeduplicator(keys=["meta.tags", "meta.info"]).run(documents=docs)
        assert result["documents"] == [docs[0], docs[2]]
        assert result["duplicates"] == [docs[1]]

    def test_run_with_several_keys_needs_all_to_match(self):
        docs = [
            Document(content="same", meta={"file_path": "a.pdf"}),
            Document(content="same", meta={"file_path": "b.pdf"}),
            Document(content="same", meta={"file_path": "a.pdf", "page_number": 2}),
        ]
        result = DocumentDeduplicator(keys=["content", "meta.file_path"]).run(documents=docs)
        assert result["documents"] == [docs[0], docs[1]]
        assert result["duplicates"] == [docs[2]]

    def test_run_does_not_mix_values_of_different_keys(self):
        docs = [
            Document(content="x", meta={"a": "y"}),
            Document(content="y", meta={"a": "x"}),
            Document(content="x", meta={"b": "x", "a": "y"}),
        ]
        result = DocumentDeduplicator(keys=["content", "meta.a"]).run(documents=docs)
        assert result["documents"] == [docs[0], docs[1]]
        assert result["duplicates"] == [docs[2]]

    def test_run_does_not_change_documents(self):
        docs = [Document(content="same", meta={"n": 1}, score=0.1), Document(content="same", meta={"n": 2}, score=0.9)]
        before = [doc.to_dict() for doc in docs]
        DocumentDeduplicator().run(documents=docs)
        assert [doc.to_dict() for doc in docs] == before

    def test_run_is_stateless_between_runs(self):
        deduplicator = DocumentDeduplicator()
        first = deduplicator.run(documents=[Document(content="same", meta={"user": "A"})])
        second = deduplicator.run(documents=[Document(content="same", meta={"user": "B"})])
        assert [doc.meta["user"] for doc in first["documents"]] == ["A"]
        assert [doc.meta["user"] for doc in second["documents"]] == ["B"]
        assert first["duplicates"] == second["duplicates"] == []

    def test_run_logs_number_of_removed_documents(self, caplog):
        docs = [Document(content="same", meta={"n": 1}), Document(content="same", meta={"n": 2})]
        with caplog.at_level(logging.DEBUG):
            DocumentDeduplicator().run(documents=docs)
        assert "Removed 1 duplicate documents out of 2." in caplog.text

    def test_in_indexing_pipeline_before_document_writer(self):
        document_store = InMemoryDocumentStore()
        pipeline = Pipeline()
        pipeline.add_component("deduplicator", DocumentDeduplicator())
        pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
        pipeline.connect("deduplicator.documents", "writer.documents")

        docs = [
            Document(content="Haystack is a framework.", meta={"url": "https://example.com/a"}),
            Document(content="Haystack is a framework.", meta={"url": "https://example.com/a?ref=1"}),
            Document(content="Pipelines connect components.", meta={"url": "https://example.com/b"}),
        ]
        result = pipeline.run({"deduplicator": {"documents": docs}})

        assert result["writer"]["documents_written"] == 2
        assert result["deduplicator"]["duplicates"] == [docs[1]]
        assert {doc.meta["url"] for doc in document_store.filter_documents()} == {
            "https://example.com/a",
            "https://example.com/b",
        }

    def test_in_query_pipeline_after_document_joiner(self):
        bm25_hits = [Document(content="Paris is the capital of France.", meta={"file": "a.pdf"}, score=0.8)]
        embedding_hits = [
            Document(content="Paris is the capital of France.", meta={"file": "b.html"}, score=0.9),
            Document(content="Berlin is the capital of Germany.", meta={"file": "c.pdf"}, score=0.3),
        ]
        pipeline = Pipeline()
        pipeline.add_component("bm25", _FixedRetriever(bm25_hits))
        pipeline.add_component("embedding", _FixedRetriever(embedding_hits))
        pipeline.add_component("joiner", DocumentJoiner())
        pipeline.add_component("deduplicator", DocumentDeduplicator())
        pipeline.connect("bm25.documents", "joiner.documents")
        pipeline.connect("embedding.documents", "joiner.documents")
        pipeline.connect("joiner.documents", "deduplicator.documents")

        result = pipeline.run({"bm25": {"query": "capital"}, "embedding": {"query": "capital"}})

        assert [doc.meta["file"] for doc in result["deduplicator"]["documents"]] == ["b.html", "c.pdf"]
        assert [doc.meta["file"] for doc in result["deduplicator"]["duplicates"]] == ["a.pdf"]
