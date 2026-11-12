# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.evaluators import AnswerF1Evaluator
from haystack.core.serialization import component_from_dict, component_to_dict


def test_init_default():
    evaluator = AnswerF1Evaluator()
    assert evaluator.normalize is True


def test_to_dict():
    evaluator = AnswerF1Evaluator(normalize=False)
    assert component_to_dict(evaluator, "evaluator") == {
        "type": "haystack.components.evaluators.answer_f1.AnswerF1Evaluator",
        "init_parameters": {"normalize": False},
    }


def test_from_dict():
    data = {
        "type": "haystack.components.evaluators.answer_f1.AnswerF1Evaluator",
        "init_parameters": {"normalize": False},
    }
    evaluator = component_from_dict(AnswerF1Evaluator, data, "evaluator")
    assert evaluator.normalize is False


def test_pipeline_serialization_round_trip():
    pipeline = Pipeline()
    pipeline.add_component("evaluator", AnswerF1Evaluator(normalize=False))
    new_pipeline = Pipeline.loads(pipeline.dumps())
    assert new_pipeline.get_component("evaluator").normalize is False


def test_run_with_all_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Berlin", "Paris"])
    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_run_with_no_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Paris", "London"])
    assert result == {"individual_scores": [0.0, 0.0], "score": 0.0}


def test_run_with_partial_matching():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        ground_truth_answers=["Eiffel Tower", "A question answering library in Python"],
        predicted_answers=["the Eiffel Tower in Paris", "A Python question answering library"],
    )
    # "eiffel tower" vs "eiffel tower in paris": precision 2/4, recall 2/2, F1 2/3
    # "question answering library in python" vs "python question answering library": precision 4/4, recall 4/5
    assert result["individual_scores"] == pytest.approx([2 / 3, 8 / 9])
    assert result["score"] == pytest.approx((2 / 3 + 8 / 9) / 2)


def test_run_counts_repeated_tokens_only_as_often_as_they_appear_in_both_answers():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(ground_truth_answers=["New York"], predicted_answers=["New New York"])
    # 2 common tokens: precision 2/3, recall 2/2
    assert result["individual_scores"] == pytest.approx([0.8])


def test_run_normalizes_case_punctuation_articles_and_whitespace():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        ground_truth_answers=["Paris", "The Eiffel Tower"], predicted_answers=["paris.", "  eiffel,   tower! "]
    )
    assert result == {"individual_scores": [1.0, 1.0], "score": 1.0}


def test_run_does_not_remove_articles_inside_words():
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(ground_truth_answers=["theme park"], predicted_answers=["theme park"])
    assert result["individual_scores"] == [1.0]
    assert evaluator._tokenize("theme an apple") == ["theme", "apple"]


def test_run_without_normalization():
    evaluator = AnswerF1Evaluator(normalize=False)
    result = evaluator.run(
        ground_truth_answers=["Paris", "The Eiffel Tower"], predicted_answers=["paris.", "The Eiffel Tower"]
    )
    assert result == {"individual_scores": [0.0, 1.0], "score": 0.5}


@pytest.mark.parametrize(
    "ground_truth, predicted, expected",
    [("", "", 1.0), ("The", "a", 1.0), ("", "Paris", 0.0), ("Paris", "", 0.0), ("Paris", "...", 0.0)],
)
def test_run_with_empty_answers(ground_truth, predicted, expected):
    evaluator = AnswerF1Evaluator()
    result = evaluator.run(ground_truth_answers=[ground_truth], predicted_answers=[predicted])
    assert result == {"individual_scores": [expected], "score": expected}


def test_run_with_no_answers():
    evaluator = AnswerF1Evaluator()
    with pytest.raises(ValueError, match="must be provided"):
        evaluator.run(ground_truth_answers=[], predicted_answers=[])


def test_run_with_different_lengths():
    evaluator = AnswerF1Evaluator()
    with pytest.raises(ValueError, match="must be the same"):
        evaluator.run(ground_truth_answers=["Berlin"], predicted_answers=["Berlin", "Paris"])


def test_run_in_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("evaluator", AnswerF1Evaluator())
    result = pipeline.run(
        {"evaluator": {"ground_truth_answers": ["Berlin", "Paris"], "predicted_answers": ["Berlin", "Lyon"]}}
    )
    assert result == {"evaluator": {"individual_scores": [1.0, 0.0], "score": 0.5}}
