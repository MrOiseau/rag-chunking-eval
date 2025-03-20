from database.chromadb import ChromaDBHandler
from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda"},  # Set GPU usage
    )

    collection_names = [
        "qasper-recursive_character",
        "qasper-semantic_clustering",
        "qasper-sentence_transformers",
        "qasper-hierarchical",
    ]

    for collection_name in collection_names:
        database = ChromaDBHandler(
            collection_name=collection_name, embedding_model=embedding_model
        )

        print(collection_name)
        print(database.get_collection_stats())
