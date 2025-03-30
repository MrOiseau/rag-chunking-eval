import chromadb
import os

# --- IMPORTANT: Set this to your ChromaDB persist directory ---
# Use the path from your Streamlit app logs if unsure
persist_directory = "/Users/bijanic/Documents/master/master_code_3/rag_chunking_eval/data/chroma_db"
# Or get it from environment variable like in your app:
# persist_directory = os.getenv("DB_DIR", "data/chroma_db")
# persist_directory = os.path.abspath(persist_directory) # Ensure absolute path

print(f"Looking for ChromaDB collections in: {persist_directory}")

try:
    # Initialize the persistent client
    client = chromadb.PersistentClient(path=persist_directory)

    # List all collection names (returns List[str] in newer versions)
    collection_names = client.list_collections()

    if collection_names:
        print("\nFound the following collections and chunk counts:")
        # Iterate directly over the list of names (strings)
        for name in collection_names:
            print(f"- {name}")
            try:
                # Get the specific collection object by name
                collection = client.get_collection(name)
                # Get the count of items (chunks/documents) in the collection
                item_count = collection.count()
                print(f"  Chunks: {item_count}")
            except Exception as e:
                # Print an error if fetching count fails for a specific collection
                print(f"  Error retrieving count: {e}")
    else:
        print("\nNo collections found in this directory.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure the path is correct and ChromaDB is installed.")