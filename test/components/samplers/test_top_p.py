# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import math
import random
import sys
import warnings
from typing import Any

import pytest

from haystack import Document
from haystack.components.samplers import top_p as top_p_module
from haystack.components.samplers.top_p import TopPSampler


@pytest.fixture
def documents_with_score_field() -> list[Document]:
    return [
        Document(content="Sarajevo", meta={"similarity_score": 0.7}),
        Document(content="Belgrade", meta={"similarity_score": 0.01}),
        Document(content="Berlin", meta={"similarity_score": 0.001}),
    ]


@pytest.fixture
def documents_with_score() -> list[Document]:
    return [
        Document(content="Sarajevo", score=0.7),
        Document(content="Belgrade", score=0.01),
        Document(content="Berlin", score=0.001),
    ]


class TestTopPSampler:
    @pytest.mark.parametrize("min_top_k", [-1, -10, 1.5, 2.0, "2", True, False])
    def test_init_invalid_min_top_k(self, min_top_k: Any) -> None:
        with pytest.raises(ValueError, match="min_top_k must be a non-negative integer or None"):
            TopPSampler(min_top_k=min_top_k)

    def test_init_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            TopPSampler(top_p=2.0)

    def test_run_raises_value_error(self, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(top_p=0.95)
        with pytest.raises(ValueError):
            sampler.run(documents=documents_with_score, top_p=2.0)

    def test_run_score_field(self, documents_with_score_field: list[Document]) -> None:
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = documents_with_score_field
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Sarajevo"
        assert docs[1].content == "Belgrade"

    def test_run_score_field_missing_scores(self, caplog: pytest.LogCaptureFixture) -> None:
        sampler = TopPSampler(top_p=1.0, score_field="similarity_score")
        docs = [
            Document(content="Sarajevo", meta={"similarity_score": 0.7}),
            Document(content="Belgrade", meta={"similarity_score": 0.01}),
            Document(content="Berlin", meta={"similarity_score": None}),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 2
        assert docs[0].content == "Sarajevo"
        assert docs[1].content == "Belgrade"
        assert "Score field" in caplog.text

    def test_run(self, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(top_p=0.99)
        docs = documents_with_score
        random.shuffle(docs)
        sorted_scores = sorted([doc.score for doc in docs if doc.score is not None], reverse=True)
        # The filter narrows `float | None` for mypy. This guard keeps the test sensitive to a
        # document losing its score, which the unfiltered `sorted()` used to catch via TypeError.
        assert len(sorted_scores) == len(docs)

        # top_p = 0.99 will get the top 1 document
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == 2
        assert docs_filtered[0].content == "Sarajevo"
        assert docs_filtered[1].content == "Belgrade"

        assert [doc.score for doc in docs_filtered] == sorted_scores[:2]

    def test_run_top_p_1(self, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(top_p=1.0)
        docs = documents_with_score
        random.shuffle(docs)
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) == len(docs)
        assert docs_filtered[0].content == "Sarajevo"
        assert [doc.score for doc in docs_filtered] == sorted(
            [doc.score for doc in docs if doc.score is not None], reverse=True
        )

    def test_run_top_p_0(self, caplog: pytest.LogCaptureFixture, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(top_p=0.0)
        docs = documents_with_score
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"
        assert "Top-p sampling with p=" in caplog.text

    def test_run_top_p_0_as_run_override(
        self, caplog: pytest.LogCaptureFixture, documents_with_score: list[Document]
    ) -> None:
        # top_p=0.0 passed to run() must not be silently replaced by the init value
        sampler = TopPSampler(top_p=1.0)
        docs = documents_with_score
        output = sampler.run(documents=docs, top_p=0.0)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"
        assert "Top-p sampling with p=" in caplog.text

    def test_run_with_integer_scores(self) -> None:
        # integer scores must be treated like float scores, not as missing
        sampler = TopPSampler(top_p=0.99)
        docs = [
            Document(content="Sarajevo", score=7),
            Document(content="Belgrade", score=1),
            Document(content="Berlin", score=-5),
        ]
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert len(docs_filtered) < len(docs)
        assert docs_filtered[0].content == "Sarajevo"

    def test_run_with_boolean_scores_treated_as_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        sampler = TopPSampler(top_p=0.95)
        docs = [Document(content="Sarajevo", score=True), Document(content="Belgrade", score=0.5)]
        output = sampler.run(documents=docs)
        docs_filtered = output["documents"]
        assert docs_filtered == [docs[1]]
        assert "Ensure all documents have a valid score value" in caplog.text

    def test_run_returns_empty_list_no_documents(self) -> None:
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output["documents"] == []

    def test_run_no_score_field(self, caplog: pytest.LogCaptureFixture, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(top_p=0.95, score_field="similarity_score")
        docs = documents_with_score
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content == "Sarajevo"
        assert "Score field 'similarity_score' not found" in caplog.text

    def test_run_missing_scores(self, caplog: pytest.LogCaptureFixture) -> None:
        sampler = TopPSampler(top_p=0.95)
        docs = [
            Document(content="Sarajevo", score=0.7),
            Document(content="Belgrade", score=0.01),
            Document(content="Berlin", score=None),
        ]
        output = sampler.run(documents=docs)
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "Sarajevo"
        assert "Ensure all documents have a valid score value" in caplog.text

    @pytest.mark.parametrize("min_top_k, expected_count", [(None, 1), (0, 1), (1, 1), (2, 2), (3, 3), (10, 3)])
    def test_run_min_top_k(
        self, documents_with_score: list[Document], min_top_k: int | None, expected_count: int
    ) -> None:
        sampler = TopPSampler(min_top_k=min_top_k, top_p=0.2)
        output = sampler.run(documents=list(reversed(documents_with_score)))
        assert output["documents"] == documents_with_score[:expected_count]

    def test_run_min_top_k_does_not_limit_selection(self, documents_with_score: list[Document]) -> None:
        sampler = TopPSampler(min_top_k=1, top_p=0.99)
        output = sampler.run(documents=documents_with_score)
        assert output["documents"] == documents_with_score[:2]

    def test_run_with_tied_scores_keeps_input_order(self) -> None:
        sampler = TopPSampler(top_p=0.5)
        docs = [Document(content=str(i), score=0.5) for i in range(4)]
        output = sampler.run(documents=docs)
        # each document gets 25% of the probability mass, so the first two reach top_p=0.5
        assert output["documents"] == docs[:2]

    @pytest.mark.parametrize("bad_score", [math.inf, -math.inf, math.nan])
    def test_run_with_non_finite_scores_does_not_raise_or_warn(self, bad_score: float) -> None:
        sampler = TopPSampler(top_p=0.9)
        docs = [Document(content="Sarajevo", score=0.7), Document(content="Berlin", score=bad_score)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            output = sampler.run(documents=docs)
        assert len(output["documents"]) >= 1

    def test_run_with_positive_infinite_score_returns_that_document(self) -> None:
        sampler = TopPSampler(top_p=0.9)
        docs = [Document(content="Sarajevo", score=0.7), Document(content="Berlin", score=math.inf)]
        output = sampler.run(documents=docs)
        assert output["documents"] == [docs[1]]

    def test_works_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Setting a module to None in sys.modules makes any `import torch` raise ImportError
        monkeypatch.setitem(sys.modules, "torch", None)
        module = importlib.reload(top_p_module)
        try:
            sampler = module.TopPSampler(top_p=0.95, score_field="similarity_score")
            docs = [
                Document(content="Berlin", meta={"similarity_score": -10.6}),
                Document(content="Belgrade", meta={"similarity_score": -8.9}),
                Document(content="Sarajevo", meta={"similarity_score": -4.6}),
            ]
            output = sampler.run(documents=docs)
            assert [doc.content for doc in output["documents"]] == ["Sarajevo"]
        finally:
            monkeypatch.undo()
            importlib.reload(top_p_module)

    def test_run_matches_torch_softmax(self) -> None:
        # The sampler used torch before. This test checks that the numpy implementation selects
        # exactly the same documents as the original torch code on many random score lists.
        torch = pytest.importorskip("torch")

        def torch_selection(sorted_scores: list[float], top_p: float) -> list[int]:
            tensor_scores = torch.tensor(sorted_scores, dtype=torch.float32)
            probs = torch.nn.functional.softmax(tensor_scores, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            close_to_top_p = torch.isclose(cumulative_probs, torch.tensor(top_p), atol=1e-6)
            condition = (cumulative_probs <= top_p) | close_to_top_p
            return [int(i) for i in torch.where(condition)[0]]

        rng = random.Random(42)
        for _ in range(1000):
            scale = rng.choice([0.01, 1.0, 10.0, 100.0])
            scores = sorted((rng.uniform(-scale, scale) for _ in range(rng.randint(1, 20))), reverse=True)
            top_p = rng.choice([0.0, 0.5, 0.9, 0.95, 1.0, round(rng.random(), 3)])
            docs = [Document(content=str(i), score=score) for i, score in enumerate(scores)]

            expected = [docs[i] for i in torch_selection(scores, top_p)] or [docs[0]]
            assert TopPSampler(top_p=top_p).run(documents=docs)["documents"] == expected
