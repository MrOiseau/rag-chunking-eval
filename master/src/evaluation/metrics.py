"""
Metrics for evaluating RAG system performance.
"""

import difflib
from typing import List, Dict, Tuple, Any, Optional

from langchain.schema import Document
from .models import Question


def longest_common_substring_length(str1: str, str2: str) -> int:
    """
    Find the length of the longest common substring between two strings.

    Args:
        str1 (str): First string
        str2 (str): Second string

    Returns:
        int: Length of the longest common substring
    """
    # Use difflib to find matching blocks
    matcher = difflib.SequenceMatcher(None, str1, str2)
    match = matcher.find_longest_match(0, len(str1), 0, len(str2))

    return match.size


def is_document_relevant(
    retrieved_text: str,
    evidence_list: List[str],
    similarity_threshold_pct: float = 0.3,
) -> bool:
    """
    Determine if a retrieved document is relevant to any evidence.

    Args:
        retrieved_text (str): The text of the retrieved document
        evidence_list (List[str]): List of evidence texts
        similarity_threshold_pct (float): Minimum percentage of evidence length that must match
            to be considered relevant (default: 0.3 or 30%)

    Returns:
        bool: True if the document is relevant, False otherwise
    """
    for evidence in evidence_list:
        if not evidence:  # Skip empty evidence
            continue

        # Calculate the minimum required match length (threshold % of evidence length)
        min_match_length = int(len(evidence) * similarity_threshold_pct)

        # Find the longest common substring
        lcs_length = longest_common_substring_length(retrieved_text, evidence)

        # Check if the match exceeds the threshold
        if lcs_length >= min_match_length:
            return True

    return False


def calculate_mrr_at_k(
    question: Question,
    retrieved_paragraphs: List[Tuple[Document, float]],
    k: int = 5,
    similarity_threshold_pct: float = 0.3,
) -> float:
    """
    Calculate Mean Reciprocal Rank at k (MRR@k).

    MRR@k measures the rank of the first relevant document within the top k results.
    If the first relevant document is at position 1, RR = 1
    If the first relevant document is at position 2, RR = 1/2 = 0.5
    If no relevant document is found in the top k, RR = 0

    Args:
        question (Question): The question object with evidence
        retrieved_paragraphs (List[Tuple[Document, float]]): Retrieved paragraphs with scores
        k (int): Number of paragraphs to consider (default: 5)
        similarity_threshold_pct (float): Minimum percentage of evidence length that must match
            to be considered relevant (default: 0.3 or 30%)

    Returns:
        float: MRR@k value between 0 and 1
    """
    # Extract just the documents from the (document, score) tuples
    retrieved_docs = [doc for doc, _ in retrieved_paragraphs[:k]]
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Get non-empty evidence items
    non_empty_evidence = [e for e in question.highlighted_evidence if e]

    # If there's no evidence, return 0
    if not non_empty_evidence:
        return 0.0

    # Find the first relevant document
    for i, retrieved_text in enumerate(retrieved_texts):
        if is_document_relevant(
            retrieved_text, non_empty_evidence, similarity_threshold_pct
        ):
            # Return reciprocal rank (1-based indexing)
            return 1.0 / (i + 1)

    # No relevant document found in the top k
    return 0.0


def calculate_recall_at_k(
    question: Question,
    retrieved_paragraphs: List[Tuple[Document, float]],
    k: int,
    similarity_threshold_pct: float = 0.3,
) -> float:
    """
    Calculate Recall at k (R@k).

    R@k measures the proportion of relevant documents retrieved in the top k results.
    R@k = (number of relevant documents in top k) / (total number of relevant documents)

    Args:
        question (Question): The question object with evidence
        retrieved_paragraphs (List[Tuple[Document, float]]): Retrieved paragraphs with scores
        k (int): Number of paragraphs to consider
        similarity_threshold_pct (float): Minimum percentage of evidence length that must match
            to be considered relevant (default: 0.3 or 30%)

    Returns:
        float: R@k value between 0 and 1
    """
    # Extract just the documents from the (document, score) tuples
    retrieved_docs = [doc for doc, _ in retrieved_paragraphs[:k]]
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Get non-empty evidence items
    non_empty_evidence = [e for e in question.highlighted_evidence if e]

    # If there's no evidence, return 0
    if not non_empty_evidence:
        return 0.0

    # Count relevant documents in top k
    relevant_count = 0
    for retrieved_text in retrieved_texts:
        if is_document_relevant(
            retrieved_text, non_empty_evidence, similarity_threshold_pct
        ):
            relevant_count += 1

    # Calculate recall
    return min(relevant_count / len(non_empty_evidence), 1)


def calculate_retrieval_metrics(
    question: Question,
    retrieved_paragraphs: List[Tuple[Document, float]],
    k: int,
    similarity_threshold_pct: float = 0.3,  # 30% threshold
) -> Dict[str, float]:
    """
    Calculate comprehensive retrieval metrics based on longest common substring.

    Args:
        question (Question): The question object with evidence
        retrieved_paragraphs (List[Tuple[Document, float]]): Retrieved paragraphs with scores
        k (int): Number of paragraphs to consider
        similarity_threshold_pct (float): Minimum percentage of evidence length that must match
            to be considered relevant (default: 0.3 or 30%)

    Returns:
        Dict[str, float]: Dictionary containing precision, recall, F1, MRR@k (for k=3,5,7,9),
        and R@k (for k=3,5,7,9) metrics
    """
    # Extract just the documents from the (document, score) tuples
    retrieved_docs = [doc for doc, _ in retrieved_paragraphs[:k]]
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Get non-empty evidence items
    non_empty_evidence = [e for e in question.highlighted_evidence if e]

    # Calculate similarity between each retrieved paragraph and each evidence
    relevant_count = 0

    for retrieved_text in retrieved_texts:
        # Check if this paragraph is relevant to any evidence
        if is_document_relevant(
            retrieved_text, non_empty_evidence, similarity_threshold_pct
        ):
            relevant_count += 1

    # Calculate precision and recall
    precision = relevant_count / k if k > 0 else 0
    recall = min(
        relevant_count / len(non_empty_evidence) if non_empty_evidence else 0, 1
    )

    # Calculate F1 score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate MRR@k for different k values
    mrr_at_3 = (
        calculate_mrr_at_k(
            question,
            retrieved_paragraphs,
            k=3,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 3
        else 0.0
    )

    mrr_at_5 = (
        calculate_mrr_at_k(
            question,
            retrieved_paragraphs,
            k=5,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 5
        else 0.0
    )

    mrr_at_7 = (
        calculate_mrr_at_k(
            question,
            retrieved_paragraphs,
            k=7,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 7
        else 0.0
    )

    mrr_at_9 = (
        calculate_mrr_at_k(
            question,
            retrieved_paragraphs,
            k=9,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 9
        else 0.0
    )

    # Calculate R@k for different k values
    r_at_3 = (
        calculate_recall_at_k(
            question,
            retrieved_paragraphs,
            k=3,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 3
        else 0.0
    )

    r_at_5 = (
        calculate_recall_at_k(
            question,
            retrieved_paragraphs,
            k=5,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 5
        else 0.0
    )

    r_at_7 = (
        calculate_recall_at_k(
            question,
            retrieved_paragraphs,
            k=7,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 7
        else 0.0
    )

    r_at_9 = (
        calculate_recall_at_k(
            question,
            retrieved_paragraphs,
            k=9,
            similarity_threshold_pct=similarity_threshold_pct,
        )
        if len(retrieved_paragraphs) >= 9
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr@3": mrr_at_3,
        "mrr@5": mrr_at_5,
        "mrr@7": mrr_at_7,
        "mrr@9": mrr_at_9,
        "r@3": r_at_3,
        "r@5": r_at_5,
        "r@7": r_at_7,
        "r@9": r_at_9,
    }
