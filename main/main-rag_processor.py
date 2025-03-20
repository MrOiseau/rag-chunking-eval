import sys
import os

# Add the master/src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'master', 'src'))

from langchain_huggingface import HuggingFaceEmbeddings

from database.chromadb import ChromaDBHandler
from processor.rag_processor import RAGProcessor
from chunkers.recursive_character import RecursiveCharacterChunker
from chunkers.semantic_clustering import SemanticClusteringChunker
from chunkers.sentence_transformers_splitter import SentenceTransformersSplitter
from chunkers.hierarchical_chunker import HierarchicalChunker


if __name__ == "__main__":

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
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": device},  # Use best available device
    )

    # Uncomment the chunker you want to use
    
    # Option 1: RecursiveCharacterChunker
    # collection_name = "qasper-recursive_character-test"
    # chunker = RecursiveCharacterChunker(
    #     max_chunk_size=1000,
    #     chunk_overlap=200
    # )

    # Option 2: SemanticClusteringChunker
    # collection_name = "qasper-semantic_clustering-test"
    # chunker = SemanticClusteringChunker(
    #     embedding_model_name="BAAI/bge-small-en-v1.5",
    #     max_chunk_size=200,
    #     min_clusters=2,
    #     max_clusters=10,
    #     position_weight=0.2,
    #     preserve_order=True,
    # )
    
    # Option 3: SentenceTransformersSplitter
    # collection_name = "qasper-sentence_transformers-test"
    # chunker = SentenceTransformersSplitter(
    #     embedding_model_name="BAAI/bge-small-en-v1.5",
    #     max_chunk_size=200,
    #     similarity_threshold=0.75,
    #     window_size=3,
    #     adaptive_threshold=True,
    # )
    
    # Option 4: HierarchicalChunker
    collection_name = "qasper-hierarchical-test"
    chunker = HierarchicalChunker(
        max_chunk_size=200,
        include_headings=True,
    )

    # Use a local directory for ChromaDB
    chroma_db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
    os.makedirs(chroma_db_dir, exist_ok=True)
    print(f"Using ChromaDB directory: {chroma_db_dir}")
    
    database = ChromaDBHandler(
        collection_name=collection_name,
        embedding_model=embedding_model,
        persist_directory=chroma_db_dir
    )

    # Create Rag Processor: {ChromaDB, Embedding Model, Chunker}
    processor = RAGProcessor(database=database, chunker=chunker)

    # Process Qasper dataset
    qasper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qasper")
    os.makedirs(qasper_dir, exist_ok=True)
    print(f"Processing Qasper dataset from: {qasper_dir}")
    processor.process_directory(qasper_dir)

    # Print database stats
    print("\nDatabase stats:")
    print(processor.database.get_collection_stats())
