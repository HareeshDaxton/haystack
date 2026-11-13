# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import pytest

from haystack import Pipeline, default_from_dict
from haystack.components.evaluators.document_precision import DocumentPrecisionEvaluator
from haystack.dataclasses import Document


@pytest.fixture
def evaluator():
    return DocumentPrecisionEvaluator()


def test_init_default_comparison_field():
    evaluator = DocumentPrecisionEvaluator()
    assert evaluator.document_comparison_field == "content"


def test_to_dict(evaluator):
    assert evaluator.to_dict() == {
        "type": "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator",
        "init_parameters": {"document_comparison_field": "content"},
    }


def test_to_dict_with_custom_comparison_field():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.file_id")
    assert evaluator.to_dict() == {
        "type": "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator",
        "init_parameters": {"document_comparison_field": "meta.file_id"},
    }


def test_from_dict():
    data = {
        "type": "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator",
        "init_parameters": {"document_comparison_field": "id"},
    }
    evaluator = default_from_dict(DocumentPrecisionEvaluator, data)
    assert evaluator.document_comparison_field == "id"


def test_from_dict_without_init_parameters():
    data = {"type": "haystack.components.evaluators.document_precision.DocumentPrecisionEvaluator"}
    evaluator = default_from_dict(DocumentPrecisionEvaluator, data)
    assert evaluator.document_comparison_field == "content"


def test_run_with_all_matching(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
    )
    assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_run_with_no_matching(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[[Document(content="Paris")], [Document(content="London")]],
    )
    assert all(isinstance(individual_score, float) for individual_score in result["individual_scores"])
    assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}


def test_run_with_partial_matching(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
        retrieved_documents=[
            [Document(content="Berlin"), Document(content="Munich")],
            [Document(content="Paris"), Document(content="Lyon"), Document(content="Nice"), Document(content="Rome")],
        ],
    )
    assert result == {"individual_scores": [0.5, 0.25], "score": 0.375}


def test_precision_drops_when_more_noise_is_retrieved(evaluator):
    # Both retrievers find the only relevant document, so recall is the same.
    # Precision shows that the second retriever returns much more noise.
    ground_truth = [[Document(content="Paris is the capital of France.")]]
    focused = [[Document(content="Paris is the capital of France."), Document(content="Lyon is in France.")]]
    noisy = [[Document(content="Paris is the capital of France.")] + [Document(content=f"noise {i}") for i in range(9)]]

    assert evaluator.run(ground_truth_documents=ground_truth, retrieved_documents=focused)["score"] == 0.5
    assert evaluator.run(ground_truth_documents=ground_truth, retrieved_documents=noisy)["score"] == 0.1


def test_run_with_more_ground_truth_than_retrieved(evaluator):
    # Missing ground truth documents lower recall, not precision.
    result = evaluator.run(
        ground_truth_documents=[[Document(content="a"), Document(content="b"), Document(content="c")]],
        retrieved_documents=[[Document(content="a")]],
    )
    assert result == {"individual_scores": [1.0], "score": 1.0}


def test_run_with_duplicate_retrieved_documents_counted_once(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin")]],
        retrieved_documents=[[Document(content="Berlin"), Document(content="Berlin"), Document(content="London")]],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_duplicate_ground_truth_documents(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="Berlin"), Document(content="Berlin")]],
        retrieved_documents=[[Document(content="Berlin"), Document(content="London")]],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_complex_data(evaluator):
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
            [Document(content="classical music"), Document(content="classical")],
            [Document(content="11th century"), Document(content="the 11th")],
            [Document(content="Denmark"), Document(content="Iceland"), Document(content="Norway")],
            [Document(content="10th century"), Document(content="10th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
            [Document(content="classical"), Document(content="rock music"), Document(content="dubstep")],
            [Document(content="11th"), Document(content="the 11th"), Document(content="11th century")],
            [Document(content="Denmark"), Document(content="Norway"), Document(content="Sweden")],
            [Document(content="the first half of the 10th century"), Document(content="10th")],
        ],
    )
    assert result["individual_scores"] == pytest.approx([1.0, 2 / 3, 1 / 3, 2 / 3, 2 / 3, 0.5])
    assert result["score"] == pytest.approx((1.0 + 2 / 3 + 1 / 3 + 2 / 3 + 2 / 3 + 0.5) / 6)


def test_run_with_id_comparison():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="id")
    result = evaluator.run(
        ground_truth_documents=[[Document(id="doc1", content="foo")], [Document(id="doc2", content="bar")]],
        retrieved_documents=[
            [Document(id="doc1", content="different"), Document(id="doc3", content="other")],
            [Document(id="wrong", content="bar")],
        ],
    )
    assert result == {"individual_scores": [0.5, 0.0], "score": 0.25}


def test_run_with_meta_comparison():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="x", meta={"file_id": "a"})]],
        retrieved_documents=[
            [
                Document(content="z", meta={"file_id": "a"}),
                Document(content="w", meta={"file_id": "b"}),
                Document(content="v", meta={"file_id": "c"}),
                Document(content="u", meta={"file_id": "d"}),
            ]
        ],
    )
    assert result == {"individual_scores": [0.25], "score": 0.25}


def test_run_with_nested_meta_comparison():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.source.url")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="x", meta={"source": {"url": "https://a.com"}})]],
        retrieved_documents=[
            [
                Document(content="z", meta={"source": {"url": "https://a.com"}}),
                Document(content="w", meta={"source": {"url": "https://c.com"}}),
            ]
        ],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_nested_meta_comparison_when_parent_is_not_a_dict():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.source.url")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="x", meta={"source": {"url": "https://a.com"}})]],
        retrieved_documents=[[Document(content="z", meta={"source": "https://a.com"})]],
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


@pytest.mark.parametrize(
    "truth, matching, different",
    [
        (["a", "b"], ["a", "b"], ["b", "a"]),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2, "b": 1}),
        ({"ids": ["a", "b"]}, {"ids": ["a", "b"]}, {"ids": ["b", "a"]}),
        ("a", "a", "b"),
    ],
)
def test_container_metadata_comparison(truth: Any, matching: Any, different: Any) -> None:
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.source_ids")
    result = evaluator.run(
        ground_truth_documents=[[Document(meta={"source_ids": truth})]],
        retrieved_documents=[[Document(meta={"source_ids": matching}), Document(meta={"source_ids": different})]],
    )
    assert result == {"individual_scores": [0.5], "score": 0.5}


def test_run_with_unsupported_comparison_field():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="title")
    with pytest.raises(ValueError, match="Unsupported document_comparison_field"):
        evaluator.run(ground_truth_documents=[[Document(content="a")]], retrieved_documents=[[Document(content="a")]])


def test_run_with_empty_inputs(evaluator):
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_documents=[], retrieved_documents=[])


def test_run_with_different_lengths(evaluator):
    with pytest.raises(ValueError, match="must be the same"):
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")]],
            retrieved_documents=[[Document(content="Berlin")], [Document(content="London")]],
        )

    with pytest.raises(ValueError, match="must be the same"):
        evaluator.run(
            ground_truth_documents=[[Document(content="Berlin")], [Document(content="Paris")]],
            retrieved_documents=[[Document(content="Berlin")]],
        )


def test_empty_ground_truth_documents(evaluator, caplog):
    with caplog.at_level(logging.WARNING):
        result = evaluator.run(ground_truth_documents=[[]], retrieved_documents=[[Document(content="test")]])
    assert result == {"individual_scores": [0.0], "score": 0.0}
    assert "no ground truth documents" in caplog.text


def test_empty_retrieved_documents(evaluator, caplog):
    with caplog.at_level(logging.WARNING):
        result = evaluator.run(ground_truth_documents=[[Document(content="test")]], retrieved_documents=[[]])
    assert result == {"individual_scores": [0.0], "score": 0.0}
    assert "no retrieved documents" in caplog.text


def test_empty_string_ground_truth_documents(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="")]], retrieved_documents=[[Document(content="test")]]
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


def test_empty_string_retrieved_documents(evaluator):
    result = evaluator.run(
        ground_truth_documents=[[Document(content="test")]], retrieved_documents=[[Document(content="")]]
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


def test_retrieved_documents_without_comparison_value_are_ignored():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[[Document(meta={"file_id": "f1"})]],
        retrieved_documents=[[Document(meta={"file_id": "f1"}), Document(content="no file_id")]],
    )
    assert result == {"individual_scores": [1.0], "score": 1.0}


def test_no_match_when_comparison_field_missing_on_all_documents():
    evaluator = DocumentPrecisionEvaluator(document_comparison_field="meta.file_id")
    result = evaluator.run(
        ground_truth_documents=[[Document(content="real ground truth")]],
        retrieved_documents=[[Document(content="totally unrelated doc")]],
    )
    assert result == {"individual_scores": [0.0], "score": 0.0}


def test_in_pipeline(evaluator):
    pipeline = Pipeline()
    pipeline.add_component("precision", evaluator)
    result = pipeline.run(
        {
            "precision": {
                "ground_truth_documents": [[Document(content="Berlin")]],
                "retrieved_documents": [[Document(content="Berlin"), Document(content="Paris")]],
            }
        }
    )
    assert result == {"precision": {"score": 0.5, "individual_scores": [0.5]}}
