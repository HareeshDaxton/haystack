# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
import string
from collections import Counter
from typing import Any

from haystack import component, default_to_dict

_ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


@component
class AnswerF1Evaluator:
    """
    An answer F1 evaluator class.

    The evaluator computes the token-level F1 score between each predicted answer and its ground truth answer,
    as defined in the SQuAD benchmark. Unlike exact match, it gives partial credit when the two answers share
    some but not all of their words.
    The result is a number from 0.0 to 1.0: the average F1 score over all answer pairs.

    By default, both answers are normalized before comparison: they are lowercased, and punctuation, the English
    articles "a", "an", and "the", and extra whitespace are removed.

    Usage example:
    ```python
    from haystack.components.evaluators import AnswerF1Evaluator

    evaluator = AnswerF1Evaluator()
    result = evaluator.run(
        ground_truth_answers=["Eiffel Tower", "Paris"],
        predicted_answers=["the Eiffel Tower in Paris", "Paris."],
    )

    print(result["individual_scores"])
    # [0.6666666666666666, 1.0]
    print(result["score"])
    # 0.8333333333333333
    ```
    """

    def __init__(self, normalize: bool = True) -> None:
        """
        Creates an AnswerF1Evaluator component.

        :param normalize:
            If `True`, lowercases both answers and removes punctuation, the articles "a", "an", and "the",
            and extra whitespace before splitting them into tokens.
            If `False`, the answers are only split on whitespace.
        """
        self.normalize = normalize

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, normalize=self.normalize)

    @component.output_types(individual_scores=list[float], score=float)
    def run(self, ground_truth_answers: list[str], predicted_answers: list[str]) -> dict[str, Any]:
        """
        Run the AnswerF1Evaluator on the given inputs.

        The `ground_truth_answers` and `predicted_answers` must have the same length.

        :param ground_truth_answers:
            A list of expected answers.
        :param predicted_answers:
            A list of predicted answers.
        :returns:
            A dictionary with the following outputs:
            - `individual_scores` - A list of F1 scores from 0.0 to 1.0, one for each pair of ground truth and
                predicted answer.
            - `score` - The average of the individual scores.
        """
        if len(ground_truth_answers) == 0 or len(predicted_answers) == 0:
            raise ValueError("ground_truth_answers and predicted_answers must be provided.")

        if len(ground_truth_answers) != len(predicted_answers):
            raise ValueError("The length of ground_truth_answers and predicted_answers must be the same.")

        individual_scores = [
            self._f1_score(truth, predicted)
            for truth, predicted in zip(ground_truth_answers, predicted_answers, strict=True)
        ]
        score = sum(individual_scores) / len(individual_scores)

        return {"individual_scores": individual_scores, "score": score}

    def _tokenize(self, text: str) -> list[str]:
        """
        Splits an answer into tokens, normalizing it first if `normalize` is enabled.

        :param text:
            The answer to tokenize.
        :returns:
            The list of tokens.
        """
        if self.normalize:
            text = text.lower().translate(_PUNCTUATION_TABLE)
            text = _ARTICLES_REGEX.sub(" ", text)
        return text.split()

    def _f1_score(self, ground_truth_answer: str, predicted_answer: str) -> float:
        """
        Computes the token-level F1 score between one ground truth answer and one predicted answer.

        :param ground_truth_answer:
            The expected answer.
        :param predicted_answer:
            The predicted answer.
        :returns:
            The F1 score. If both answers have no tokens, the score is 1.0. If only one of them has no tokens,
            the score is 0.0.
        """
        truth_tokens = self._tokenize(ground_truth_answer)
        predicted_tokens = self._tokenize(predicted_answer)

        if not truth_tokens or not predicted_tokens:
            return float(truth_tokens == predicted_tokens)

        common_tokens = sum((Counter(truth_tokens) & Counter(predicted_tokens)).values())
        if common_tokens == 0:
            return 0.0

        precision = common_tokens / len(predicted_tokens)
        recall = common_tokens / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)
