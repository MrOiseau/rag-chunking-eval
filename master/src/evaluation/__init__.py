"""
Evaluation module for Retrieval-Augmented Generation (RAG) systems.
Provides tools for assessing RAG performance using precision, recall, and other metrics.
"""
from .models import Question
from .rag_evaluator import RagEvaluator
from .metrics import longest_common_substring_length, calculate_retrieval_metrics

__all__ = ["Question", "RagEvaluator", "longest_common_substring_length", "calculate_retrieval_metrics"]