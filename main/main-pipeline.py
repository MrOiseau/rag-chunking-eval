import os
import json
import sys

# Add the master/src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'master', 'src'))

from langchain_huggingface import HuggingFaceEmbeddings

from database.chromadb import ChromaDBHandler
from chunkers.base import BaseChunker
from chunkers.recursive_character import RecursiveCharacterChunker
from chunkers.semantic_clustering import SemanticClusteringChunker
from chunkers.sentence_transformers_splitter import SentenceTransformersSplitter
from chunkers.hierarchical_chunker import HierarchicalChunker
from evaluation.rag_evaluator import RagEvaluator
from processor.rag_processor import RAGProcessor


def evaluate(
    database: ChromaDBHandler,
    chunker: BaseChunker,
    qasper_dir: str,
    k: int = 5,
    similarity_threshold_pct: float = 0.3,
    use_llm: bool = True,
    temperature: float = 0.0,
    model: str = "gpt-4o-mini",
):
    """
    Evaluate the RAG system and save metrics to output directory.

    Args:
        database (ChromaDBHandler): Database handler
        chunker (BaseChunker): Chunker to use
        k (int): Number of paragraphs to retrieve for each question (default: 5)
        similarity_threshold_pct (float): Threshold for similarity matching
        use_llm (bool): Whether to use LLM for answer generation
        temperature (float): Temperature for LLM generation
        model (str): Model to use for LLM generation
    """
    # Process documents

    if database.get_collection_stats()["document_count"] == 0:
        processor = RAGProcessor(database=database, chunker=chunker)
        processor.process_directory(qasper_dir)

    print(database.get_collection_stats())

    # Initialize evaluator
    evaluator = RagEvaluator(
        qasper_dir=qasper_dir,
        database=database,
        similarity_threshold_pct=similarity_threshold_pct,
        use_llm=use_llm,
        temperature=temperature,
        model=model,
    )

    # Create output directory
    base_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "evaluation_results")
    output_dir = os.path.join(base_output_dir, database.collection_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving evaluation results to: {output_dir}")

    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        k=k, save_metrics=True, output_dir=os.path.join(output_dir, "metrics")
    )

    # Save overall metrics to metrics.json
    metrics_file = os.path.join(output_dir, "metrics.json")
    overall_metrics = {
        "average_metrics": results["average_metrics"],
        "total_questions": results["total_questions"],
        "paper_count": len(results.get("paper_metrics", {})),
        "collection_name": database.collection_name,
        "k": k,  # Add k to metrics
        "similarity_threshold_pct": similarity_threshold_pct,
        "llm_config": {
            "use_llm": use_llm,
            "temperature": temperature,
            "model": model,
        },
        "chunker_type": chunker.__class__.__name__,
        "chunker_config": chunker.get_config(),
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)

    print(f"Overall metrics saved to: {metrics_file}")
    print(f"Average Precision: {results['average_metrics']['precision']:.4f}")
    print(f"Average Recall: {results['average_metrics']['recall']:.4f}")
    print(f"Average F1 Score: {results['average_metrics']['f1']:.4f}")


if __name__ == "__main__":

    # Process Qasper dataset
    qasper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qasper")
    os.makedirs(qasper_dir, exist_ok=True)
    print(f"Using Qasper dataset from: {qasper_dir}")

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

    for collection_name, chunker in [
        # (
        #     "qasper-recursive_character",
        #     RecursiveCharacterChunker(max_chunk_size=1000, chunk_overlap=200),
        # ),
        (
            "qasper-semantic_clustering",
            SemanticClusteringChunker(
                embedding_model_name="BAAI/bge-small-en-v1.5",
                max_chunk_size=200,
                min_clusters=2,
                max_clusters=10,
                position_weight=0.2,
                preserve_order=True,
            ),
        ),
        # (
        #     "qasper-sentence_transformers",
        #     SentenceTransformersSplitter(
        #         embedding_model_name="BAAI/bge-small-en-v1.5",
        #         max_chunk_size=200,
        #         similarity_threshold=0.75,
        #         window_size=3,
        #         adaptive_threshold=True,
        #     ),
        # ),
        # (
        #     "qasper-hierarchical",
        #     HierarchicalChunker(
        #         max_chunk_size=200,
        #         include_headings=True,
        #     ),
        # ),
    ]:

        print(collection_name)
        # Use a local directory for ChromaDB
        chroma_db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
        os.makedirs(chroma_db_dir, exist_ok=True)
        print(f"Using ChromaDB directory: {chroma_db_dir}")
        
        database = ChromaDBHandler(
            collection_name=collection_name,
            embedding_model=embedding_model,
            persist_directory=chroma_db_dir
        )
        evaluate(
            database=database,
            chunker=chunker,
            qasper_dir=qasper_dir,
            k=5,
            similarity_threshold_pct=0.3,
            use_llm=True,
            temperature=0.0,
            model="gpt-4o-mini",
        )
