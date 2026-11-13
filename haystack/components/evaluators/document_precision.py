# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.misc import _make_metadata_value_hashable

logger = logging.getLogger(__name__)


@component
class DocumentPrecisionEvaluator:
    """
    Evaluator that calculates the Precision score for a list of documents.

    Precision is the share of retrieved documents that are relevant, that is, that match one of the ground truth
    documents. It shows how much noise a retriever returns: a retriever that finds the right document but also returns
    many irrelevant ones gets a high recall but a low precision.

    Returns both a list of scores for each question and the average.
    There can be multiple ground truth documents and multiple retrieved documents as input.
    Documents are compared by their unique values of `document_comparison_field`, so a document that is retrieved
    twice counts once.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentPrecisionEvaluator

    evaluator = DocumentPrecisionEvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(content="France")],
            [Document(content="9th century"), Document(content="9th")],
        ],
        retrieved_documents=[
            [Document(content="France")],
            [
                Document(content="9th century"),
                Document(content="10th century"),
                Document(content="9th"),
                Document(content="the 9th century"),
            ],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 0.5]
    print(result["score"])
    # 0.75
    ```
    """

    def __init__(self, document_comparison_field: str = "content") -> None:
        """
        Create a DocumentPrecisionEvaluator component.

        :param document_comparison_field:
            The Document field to use for comparison. Possible options:
            - `"content"`: uses `doc.content`
            - `"id"`: uses `doc.id`
            - A `meta.` prefix followed by a key name: uses `doc.meta["<key>"]`
              (e.g. `"meta.file_id"`, `"meta.page_number"`)
              Nested keys are supported (e.g. `"meta.source.url"`).
              Values may also be lists or dictionaries.
        """
        self.document_comparison_field = document_comparison_field

    def _get_comparison_value(self, doc: Document) -> Any:
        """
        Extract the comparison value from a document based on the configured field.
        """
        if self.document_comparison_field == "content":
            return doc.content
        if self.document_comparison_field == "id":
            return doc.id
        if self.document_comparison_field.startswith("meta."):
            parts = self.document_comparison_field[5:].split(".")
            value = doc.meta
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return None
                value = value[part]
            return _make_metadata_value_hashable(value)
        msg = (
            f"Unsupported document_comparison_field: '{self.document_comparison_field}'. "
            "Use 'content', 'id', or 'meta.<key>'."
        )
        raise ValueError(msg)

    def _unique_comparison_values(self, documents: list[Document]) -> set:
        """
        Collect the unique comparison values of the documents, ignoring missing or empty ones.
        """
        return {value for doc in documents if (value := self._get_comparison_value(doc)) not in ("", None)}

    def _precision(self, ground_truth_documents: list[Document], retrieved_documents: list[Document]) -> float:
        """
        Calculate the precision for the retrieved documents of a single question.
        """
        unique_truths = self._unique_comparison_values(ground_truth_documents)
        if not unique_truths:
            logger.warning(
                "There are no ground truth documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )
            return 0.0

        unique_retrievals = self._unique_comparison_values(retrieved_documents)
        if not unique_retrievals:
            logger.warning(
                "There are no retrieved documents or none of them contain a valid comparison value. "
                "Score will be set to 0."
            )
            return 0.0

        return len(unique_truths.intersection(unique_retrievals)) / len(unique_retrievals)

    @component.output_types(score=float, individual_scores=list[float])
    def run(
        self, ground_truth_documents: list[list[Document]], retrieved_documents: list[list[Document]]
    ) -> dict[str, Any]:
        """
        Run the DocumentPrecisionEvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the proportion of retrieved
                documents that match a ground truth document.
        :raises ValueError:
            If the inputs are empty or don't have the same length.
        """
        if len(ground_truth_documents) == 0 or len(retrieved_documents) == 0:
            msg = "ground_truth_documents and retrieved_documents must be provided."
            raise ValueError(msg)

        if len(ground_truth_documents) != len(retrieved_documents):
            msg = "The length of ground_truth_documents and retrieved_documents must be the same."
            raise ValueError(msg)

        scores = [self._precision(gt, ret) for gt, ret in zip(ground_truth_documents, retrieved_documents, strict=True)]

        return {"score": sum(scores) / len(retrieved_documents), "individual_scores": scores}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, document_comparison_field=self.document_comparison_field)
