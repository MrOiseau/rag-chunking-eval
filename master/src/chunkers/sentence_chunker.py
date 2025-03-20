"""
SentenceChunker module for the RAG system.

This module implements the SentenceChunker class, which chunks documents
by sentences using NLTK's sentence tokenizer.
"""

from typing import List, Dict, Any, Optional
import nltk
from langchain.schema import Document
from chunkers.base import BaseChunker


class SentenceChunker(BaseChunker):
    """
    A chunker that splits documents by sentences.
    
    This chunker uses NLTK's sentence tokenizer to split documents into
    sentences, and then groups sentences into chunks based on the maximum
    chunk size.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        language: str = "english",
        **kwargs,
    ):
        """
        Initialize the SentenceChunker.
        
        Args:
            max_chunk_size (int): The target size of each chunk in characters.
            chunk_overlap (int): The number of characters to overlap between chunks.
            language (str): The language for NLTK's sentence tokenizer.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
        
        self.language = language
        
        # Download NLTK resources if needed
        try:
            nltk.data.find(f'tokenizers/punkt/{self.language}.pickle')
        except LookupError:
            # Use trusted_host to avoid SSL certificate issues
            nltk.download('punkt', quiet=True, raise_on_error=False)
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents by sentences.
        
        Args:
            docs (List[Document]): List of documents to be chunked.
            
        Returns:
            List[Document]: A list of chunked documents.
        """
        chunked_docs = []
        
        try:
            for doc in docs:
                # Get the document text and metadata
                text = doc.page_content
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                
                # Split the text into sentences
                sentences = nltk.sent_tokenize(text, language=self.language)
                
                # Group sentences into chunks
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    # If adding this sentence would exceed the max chunk size,
                    # and we already have some content, create a new chunk
                    if current_size + sentence_size > self.max_chunk_size and current_chunk:
                        # Join the current chunk sentences and create a document
                        chunk_text = " ".join(current_chunk)
                        chunk_doc = Document(page_content=chunk_text, metadata=metadata.copy())
                        chunked_docs.append(chunk_doc)
                        
                        # Start a new chunk with overlap
                        overlap_size = 0
                        overlap_chunk = []
                        
                        # Add sentences from the end of the previous chunk for overlap
                        for s in reversed(current_chunk):
                            if overlap_size + len(s) <= self.chunk_overlap:
                                overlap_chunk.insert(0, s)
                                overlap_size += len(s)
                            else:
                                break
                        
                        current_chunk = overlap_chunk
                        current_size = overlap_size
                    
                    # Add the current sentence to the chunk
                    current_chunk.append(sentence)
                    current_size += sentence_size
                
                # Add the last chunk if it's not empty
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk_doc = Document(page_content=chunk_text, metadata=metadata.copy())
                    chunked_docs.append(chunk_doc)
            
            return chunked_docs
        
        except Exception as e:
            print(f"Error during document chunking: {e}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chunker.
        
        Returns:
            Dict[str, Any]: A dictionary containing the chunker's configuration.
        """
        config = super().get_config()
        config.update({
            "language": self.language
        })
        return config