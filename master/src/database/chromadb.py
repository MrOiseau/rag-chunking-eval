import os
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata


class ChromaDBHandler:
    """
    A handler class for ChromaDB operations using LangChain.
    This class provides methods to store and retrieve document chunks.
    """

    def __init__(
        self,
        collection_name: str = "default_collection",
        persist_directory: str = "/app/data/chroma_db",
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize the ChromaDB handler.

        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist the ChromaDB data
            embedding_model: Embedding model to use (defaults to OpenAIEmbeddings)
        """
        # Create the persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the embedding model
        self.embedding_model = embedding_model or OpenAIEmbeddings()

        # Initialize the ChromaDB client
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def add_chunks(
        self, chunks: List[Document], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add pre-chunked documents directly to the ChromaDB collection.

        Args:
            chunks (List[Document]): List of pre-chunked LangChain Document objects
            metadatas (List[Dict]): Optional metadata for each chunk

        Returns:
            List[str]: List of document IDs
        """
        # Add the chunks directly to the ChromaDB collection
        # Don't pass metadatas parameter as Document objects already contain metadata
        ids = self.db.add_documents(chunks)

        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search on the ChromaDB collection.

        Args:
            query (str): Query string
            k (int): Number of results to return

        Returns:
            List[Document]: List of similar documents
        """
        return self.db.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4, paper_id: Optional[str] = None
    ) -> List[tuple]:
        """
        Perform a similarity search on the ChromaDB collection with scores.

        Args:
            query (str): Query string
            k (int): Number of results to return
            paper_id (Optional[str]): Filter results by paper_id if provided

        Returns:
            List[tuple]: List of (document, score) tuples
        """
        # Create filter if paper_id is provided
        filter_dict = {"paper_id": paper_id} if paper_id else None

        return self.db.similarity_search_with_score(query, k=k, filter=filter_dict)

    def delete_collection(self) -> None:
        """
        Delete the ChromaDB collection.
        """
        self.db.delete_collection()

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.

        Returns:
            Dict: Collection statistics
        """
        return {
            "collection_name": self.collection_name,
            "document_count": len(self.db.get()["documents"]) if self.db.get() else 0,
            "persist_directory": self.persist_directory,
        }