"""
RecursiveCharacterChunker module for the RAG system.

This module implements the RecursiveCharacterChunker class, which uses
LangChain's RecursiveCharacterTextSplitter to chunk documents.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chunkers.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """
    A chunker that uses RecursiveCharacterTextSplitter to chunk documents.

    This chunker splits documents based on character count with specified
    chunk size and overlap, using a list of separators to guide the splitting.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the RecursiveCharacterChunker.

        Args:
            max_chunk_size (int): The target size of each chunk in characters.
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (Optional[List[str]]): List of separators to use for splitting.
                If None, a default list will be used.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs,
        )

        # Default separators if none provided
        if separators is None:
            separators = [
                ".\n\n",
                ".\n",
                "!\n\n",
                "!\n",
                "! ",
                "? ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,  # LangChain uses chunk_size parameter
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents using RecursiveCharacterTextSplitter.

        Args:
            docs (List[Document]): List of document dictionaries to be chunked.

        Returns:
            List[Document]: A list of chunked document dictionaries.
        """
        try:
            chunks = self.text_splitter.split_documents(docs)
            return chunks
        except Exception as e:
            print(f"Error during document chunking: {e}")
            return []
