from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter
)

class ChunkingStrategy:
    """Base class for document chunking strategies"""
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Split text into chunks
        
        Args:
            text (str): Text to split
            metadata (Dict): Optional metadata for the document
            
        Returns:
            List[Document]: List of document chunks
        """
        raise NotImplementedError("Subclasses must implement this method")


class RecursiveCharacterChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy using RecursiveCharacterTextSplitter.
    This is good for most text types and tries to keep paragraphs, sentences, etc. together.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunking strategy
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            separators (List[str]): List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks"""
        return self.text_splitter.create_documents([text], metadatas=[metadata] if metadata else None)


class MarkdownChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy specifically for Markdown documents.
    This preserves Markdown structure like headers, lists, etc.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the chunking strategy
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks"""
        return self.text_splitter.create_documents([text], metadatas=[metadata] if metadata else None)


class TokenChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy using TokenTextSplitter.
    This is good for controlling the exact number of tokens per chunk.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base"  # Default for GPT-4
    ):
        """
        Initialize the chunking strategy
        
        Args:
            chunk_size (int): Size of each chunk in tokens
            chunk_overlap (int): Overlap between chunks in tokens
            encoding_name (str): Name of the encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks"""
        return self.text_splitter.create_documents([text], metadatas=[metadata] if metadata else None)


class SentenceChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy that tries to keep sentences together.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the chunking strategy
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use separators that respect sentence boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ",", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Split text into chunks"""
        return self.text_splitter.create_documents([text], metadatas=[metadata] if metadata else None)


# Factory function to get a chunking strategy
def get_chunking_strategy(
    strategy_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> ChunkingStrategy:
    """
    Get a chunking strategy based on the strategy type
    
    Args:
        strategy_type (str): Type of chunking strategy
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        ChunkingStrategy: A chunking strategy instance
    """
    if strategy_type == "recursive":
        return RecursiveCharacterChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators")
        )
    elif strategy_type == "markdown":
        return MarkdownChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy_type == "token":
        return TokenChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=kwargs.get("encoding_name", "cl100k_base")
        )
    elif strategy_type == "sentence":
        return SentenceChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_type}")