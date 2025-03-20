"""
BaseChunker module for the RAG system.

This module defines the BaseChunker abstract base class that all chunker implementations
should inherit from. It provides a common interface for document chunking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.

    All chunker implementations should inherit from this class and implement
    the required methods. This ensures a consistent interface across different
    chunking strategies.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ):
        """
        Initialize the BaseChunker.

        Args:
            max_chunk_size (int): The target size of each chunk in characters.
            chunk_overlap (int): The number of characters to overlap between chunks.
            **kwargs: Additional arguments for specific chunker implementations.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents according to the specific chunking strategy.

        This method must be implemented by all subclasses.

        Args:
            docs (List[Document]): List of documents to be chunked.

        Returns:
            List[Document]: A list of chunked documents.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chunker.

        Returns:
            Dict[str, Any]: A dictionary containing the chunker's configuration.
        """
        return {
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunker_type": self.__class__.__name__,
        }
