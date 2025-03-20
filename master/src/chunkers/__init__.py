"""
Chunkers package for the RAG system.

This package contains various document chunking strategies for processing
text documents into smaller, manageable chunks for embedding and retrieval.
"""

from chunkers.base import BaseChunker
from chunkers.recursive_character import RecursiveCharacterChunker
from chunkers.sentence_chunker import SentenceChunker
from chunkers.semantic_clustering import SemanticClusteringChunker
from chunkers.sentence_transformers_splitter import SentenceTransformersSplitter
from chunkers.hierarchical_chunker import HierarchicalChunker

__all__ = [
    "BaseChunker",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "SemanticClusteringChunker",
    "SentenceTransformersSplitter",
    "HierarchicalChunker"
]