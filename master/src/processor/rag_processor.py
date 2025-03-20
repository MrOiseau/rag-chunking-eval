from typing import List, Optional, Any
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from chunkers.recursive_character import RecursiveCharacterChunker
from database.chromadb import ChromaDBHandler
from utils.markdown_utils import read_markdown_files


class RAGProcessor:
    """
    A class to handle the RAG (Retrieval-Augmented Generation) processing pipeline.
    This includes reading documents, chunking them, and storing them in a vector database.
    """

    def __init__(
        self,
        chunker: Optional[Any] = None,
        database: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize the RAG processor.

        Args:
            chunker: Document chunker (defaults to RecursiveCharacterChunker)
            database: Vector database (defaults to ChromaDBHandler)
            embedding_model: Embedding model (defaults to HuggingFaceEmbeddings)
        """
        # Initialize embedding model if not provided
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            import torch
            
            # Check for available GPU backends
            if torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA GPU")
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                print("Using Apple Metal GPU (MPS)")
            else:
                device = "cpu"
                print("No GPU available, using CPU")
                
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": device},  # Use best available device
            )

        # Initialize chunker if not provided
        self.chunker = chunker or RecursiveCharacterChunker(
            max_chunk_size=1000,
            chunk_overlap=200,
        )

        # Initialize database if not provided
        self.database = database or ChromaDBHandler(
            collection_name="default_collection",
            embedding_model=self.embedding_model,
        )

    def read_documents(self, directory: str) -> List[Document]:
        """
        Read documents from a directory.

        Args:
            directory: Directory containing documents

        Returns:
            List of Document objects
        """
        return read_markdown_files(directory)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the configured chunker.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        return self.chunker.chunk_documents(documents)

    def add_chunks_to_database(self, chunks: List[Document]) -> List[str]:
        """
        Add chunks to the vector database.

        Args:
            chunks: List of Document chunks

        Returns:
            List of document IDs
        """
        return self.database.add_chunks(chunks)

    def process_directory(self, directory: str) -> List[str]:
        """
        Process all documents in a directory: read, chunk, and add to database.

        Args:
            directory: Directory containing documents

        Returns:
            List of document IDs
        """
        # Read documents
        documents = self.read_documents(directory)
        print(f"Read {len(documents)} documents from {directory}")

        # Chunk documents
        chunks = self.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        # Add chunks to database
        doc_ids = self.add_chunks_to_database(chunks)
        print(f"Added {len(doc_ids)} chunks to database")

        return doc_ids
