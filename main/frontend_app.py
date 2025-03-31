import sys
import os

import streamlit as st
import uuid
import concurrent.futures
import re
from typing import List, Dict, Any, Optional, Tuple

# Adding the master/src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'master', 'src'))


# Now we can import directly from master/src
from database.chromadb import ChromaDBHandler
from chunkers.recursive_character import RecursiveCharacterChunker
from chunkers.semantic_clustering import SemanticClusteringChunker
from chunkers.sentence_transformers_splitter import SentenceTransformersSplitter
from chunkers.hierarchical_chunker import HierarchicalChunker

# --- LangChain / Standard Imports ---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled

# Set Streamlit page configuration at the very beginning
st.set_page_config(page_title="Утицај метода парчања текста на квалитет добављања информација у RAG системима", layout="wide")

# --- Configuration (Replaces backend.config) ---
# Get ChromaDB path from environment variable or use a default relative path
DB_DIR = os.getenv("DB_DIR", os.path.abspath("data/chroma_db"))
# Default collection name (will be overridden by selection)
DB_COLLECTION_DEFAULT = os.getenv("DB_COLLECTION", "default_rag_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") # or "text-embedding-ada-002" for OpenAI
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
SEARCH_RESULTS_NUM = int(os.getenv("SEARCH_RESULTS_NUM", "5")) # Number of results to display
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-chunking-comparison")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# --- Validate Required Environment Variables ---
required_vars = ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
if "openai" in EMBEDDING_MODEL_NAME.lower() or "gpt" in CHAT_MODEL.lower():
    required_vars.append("OPENAI_API_KEY")

missing_vars = [var for var in required_vars if not globals().get(var)]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    st.stop()

# --- LangSmith Client Initialization ---
try:
    langsmith_client = Client(api_key=LANGSMITH_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize LangSmith client: {e}")
    st.stop()

# --- Available Collections (UPDATE THESE) ---
# IMPORTANT: Replace 'name' with your ACTUAL ChromaDB collection names
# The description should match the chunker used to create that collection
available_collections = [
    {"name": "qasper-recursive_character", "description": "Recursive Character"},
    {"name": "qasper-hierarchical", "description": "Hierarchical Chunking"},
    {"name": "qasper-sentence_transformers", "description": "Sentence Transformers Splitter"},
    {"name": "qasper-semantic_clustering", "description": "Semantic Clustering"},
]
if not available_collections:
    st.error("No collections defined in `available_collections`. Please update the Streamlit script.")
    st.stop()

# No longer adding default entry

collection_descriptions = {c["name"]: c["description"] for c in available_collections}

# --- Embedding Model Initialization ---
@st.cache_resource
def get_embedding_model():
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    if "openai" in EMBEDDING_MODEL_NAME.lower() or "ada" in EMBEDDING_MODEL_NAME.lower():
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found for OpenAI embedding model.")
            st.stop()
        return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=OPENAI_API_KEY)
    else:
        # Use HuggingFace BGE model
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA GPU for embeddings")
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Metal GPU (MPS) for embeddings")
        else:
            device = "cpu"
            print("No GPU available, using CPU for embeddings")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}, # Crucial for BGE models
        )

embedding_function = get_embedding_model()

# --- LLM Initialization ---
@st.cache_resource
def get_llm():
    print(f"Initializing LLM: {CHAT_MODEL}")
    if "gpt" in CHAT_MODEL.lower():
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found for OpenAI chat model.")
            st.stop()
        return ChatOpenAI(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE, api_key=OPENAI_API_KEY)
    else:
        # Add logic for other LLMs if needed (e.g., HuggingFace Hub, Anthropic)
        st.error(f"Chat model '{CHAT_MODEL}' is not currently supported in this setup. Please use an OpenAI model.")
        st.stop()

llm = get_llm()

# --- Helper Class to Bundle DB Handler and Models ---
# Simplifies passing around components
class QueryComponents:
    def __init__(self, collection_name: str, db_dir: str, embed_model, chat_llm):
        print(f"Initializing QueryComponents for collection: {collection_name}")
        self.collection_name = collection_name
        self.db_handler = ChromaDBHandler(
            collection_name=collection_name,
            persist_directory=db_dir,
            embedding_model=embed_model # Pass the initialized model
        )
        self.llm = chat_llm
        self.embedding_model = embed_model

    def get_db_stats(self):
        try:
            return self.db_handler.get_collection_stats()
        except Exception as e:
            # Handle case where collection might not exist yet or other DB errors
            print(f"WARN: Could not get stats for {self.collection_name}: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": "Error",
                "persist_directory": self.db_handler.persist_directory,
            }

    def get_unique_titles(self) -> List[str]:
        """
        Retrieves unique titles from the collection's metadata.
        Assumes 'title' is stored in the metadata.
        """
        print(f"Fetching unique titles for {self.collection_name}...")
        try:
            # Use the underlying Chroma client's get method if handler doesn't expose it directly
            # This assumes the handler's 'db' attribute is the Chroma client instance
            data = self.db_handler.db.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            if not metadatas:
                print(f"No metadata found for collection {self.collection_name}")
                return []

            titles = set()
            for meta in metadatas:
                if meta and "title" in meta:
                    titles.add(meta["title"])
                elif meta and "source" in meta: # Fallback to source if title is missing
                     titles.add(meta["source"])


            print(f"Found {len(titles)} unique titles.")
            return sorted(list(titles))
        except Exception as e:
            print(f"ERROR: Failed to fetch unique titles for {self.collection_name}: {e}")
            st.warning(f"Could not fetch titles for {collection_descriptions.get(self.collection_name, self.collection_name)}: {e}")
            return []

# --- Initialize Query Components ---
@st.cache_resource(hash_funcs={QueryComponents: lambda qc: qc.collection_name})
def initialize_query_components(collection_name: str) -> QueryComponents:
    """
    Initialize the QueryComponents (DB handler, LLM, Embeddings) for a specific collection.
    """
    try:
        components = QueryComponents(
            collection_name=collection_name,
            db_dir=DB_DIR,
            embed_model=embedding_function, # Use the globally initialized model
            chat_llm=llm # Use the globally initialized LLM
        )
        # Try getting stats to confirm connection
        stats = components.get_db_stats()
        print(f"Initialized components for {collection_name}. Stats: {stats}")
        if stats.get("document_count", 0) == 0 and stats.get("document_count") != "Error":
             print(f"WARN: Collection '{collection_name}' appears to be empty.")
             st.warning(f"Database collection '{collection_descriptions.get(collection_name, collection_name)}' seems to be empty.")
        elif stats.get("document_count") == "Error":
            st.error(f"Failed to connect to or get stats for collection '{collection_descriptions.get(collection_name, collection_name)}'. Check DB path and collection name.")

        return components
    except Exception as e:
        print(f"ERROR: Failed to initialize QueryComponents for {collection_name}: {e}")
        st.error(f"Initialization error for {collection_descriptions.get(collection_name, collection_name)}: {e}")
        st.stop()

# --- Streamlit UI Setup ---
st.title("📊 Поређење изабраних стратегија парчања текста у RAG системима")
st.markdown(f"Поређење резултата из различитих векторских база")  # у **{DB_DIR}**")

# Sidebar for Filters and Settings
st.sidebar.title("Подешавања и филтери")

# Collection Selection for two databases
st.sidebar.subheader("Изаберите базе за поређење")
st.sidebar.markdown("**База 1**")
selected_collection_1 = st.sidebar.selectbox(
    "Изаберите стратегију парчања текста 1:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{collection_descriptions.get(x, x)}",
    help="Изаберите прву колекцију векторске базе (која представља стратегију парчања текста).",
    key="db1"
)

st.sidebar.markdown("**База 2**")
# Set default for second box to be different if possible
default_index_2 = 1 if len(available_collections) > 1 and available_collections[1]["name"] != selected_collection_1 else 0
if selected_collection_1 == available_collections[default_index_2]["name"] and len(available_collections) > 2:
    default_index_2 = 2 # Try the third one

selected_collection_2 = st.sidebar.selectbox(
    "Изаберите стратегију парчања текста 2:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{collection_descriptions.get(x, x)}",
    help="Изаберите другу колекцију векторске базе за поређење.",
    key="db2",
    index=default_index_2
)

# Clear stale session state if available collection names have changed
for key in ["current_collection_1", "current_collection_2"]:
    if key in st.session_state:
        current_coll = st.session_state[key]
        available_names = [c["name"] for c in available_collections]
        if current_coll not in available_names:
            st.session_state.pop(key, None)
            st.session_state.pop("query_components_" + key.split("_")[-1], None)

# Initialize Query Pipelines (Components) for both selected collections
if "query_components_1" not in st.session_state or st.session_state.get("current_collection_1") != selected_collection_1:
    with st.spinner(f"Учитавање базе 1: {collection_descriptions.get(selected_collection_1, selected_collection_1)}..."):
        st.session_state["query_components_1"] = initialize_query_components(selected_collection_1)
        st.session_state["current_collection_1"] = selected_collection_1
        # st.success(f"Loaded DB 1: {collection_descriptions.get(selected_collection_1, 'Unknown')}")

if "query_components_2" not in st.session_state or st.session_state.get("current_collection_2") != selected_collection_2:
    with st.spinner(f"Учитавање базе 2: {collection_descriptions.get(selected_collection_2, selected_collection_2)}..."):
        st.session_state["query_components_2"] = initialize_query_components(selected_collection_2)
        st.session_state["current_collection_2"] = selected_collection_2
        # st.success(f"Loaded DB 2: {collection_descriptions.get(selected_collection_2, 'Unknown')}")

# Get both query component instances and monkey-patch the similarity_search_with_score method
query_components_1 = st.session_state["query_components_1"]
query_components_2 = st.session_state["query_components_2"]

def patch_similarity_search(qc):
    """
    Patch the similarity_search_with_score method to properly handle paper_id filtering.
    This ensures that filtering works as defined in the ChromaDBHandler class.
    """
    original_qs = qc.db_handler.db.similarity_search_with_score
    
    def new_similarity_search(query, k=SEARCH_RESULTS_NUM, paper_id=None):
        if paper_id is None:
            # No filter, use the original method
            return original_qs(query, k=k)
        else:
            # First, let's examine what the metadata looks like in the database
            # This will help us understand how to construct the filter
            try:
                # Get a sample of the metadata to see its structure
                data = qc.db_handler.db.get(include=["metadatas"], limit=5)
                metadatas = data.get("metadatas", [])
                if metadatas:
                    print(f"Sample metadata structure: {metadatas[0]}")
                    
                # Extract various forms of the document identifier
                filter_basename = os.path.basename(paper_id)  # Just the filename: 1605.04278.md
                filter_dirname = os.path.basename(os.path.dirname(paper_id))  # Directory name: 1605.04278
                filter_id = filter_dirname  # The paper ID without extension: 1605.04278
                
                # Extract paper ID using regex (more reliable)
                paper_id_match = re.search(r'(\d{4}\.\d{5})', paper_id)
                regex_id = paper_id_match.group(1) if paper_id_match else None
                
                # Create a filter using only supported operators ($eq, $in)
                # Note: ChromaDB doesn't support $contains operator
                filter_values = []
                
                # Add all possible variations of the paper ID
                if paper_id:
                    filter_values.append(paper_id)
                if filter_basename:
                    filter_values.append(filter_basename)
                if filter_dirname:
                    filter_values.append(filter_dirname)
                if filter_id:
                    filter_values.append(filter_id)
                if regex_id:
                    filter_values.append(regex_id)
                
                # Remove duplicates
                filter_values = list(set(filter_values))
                
                # Create a filter that matches paper_id exactly
                if regex_id:
                    # If we have a regex ID (e.g., 1605.04278), use it as the primary filter
                    filter_dict = {"paper_id": {"$eq": regex_id}}
                    print(f"Using exact paper_id filter: {filter_dict}")
                else:
                    # Otherwise, use an $in filter with all possible values
                    filter_dict = {"$or": [
                        {"paper_id": {"$in": filter_values}},
                        {"source": {"$in": filter_values}},
                        {"title": {"$in": filter_values}}
                    ]}
                    print(f"Using $in filter with values: {filter_values}")
                
                print(f"Applying comprehensive filter: {filter_dict}")
                return original_qs(query, k=k, filter=filter_dict)
            except Exception as e:
                print(f"Error creating filter: {e}")
                # Fallback to no filter if there's an error
                return original_qs(query, k=k)
    
    # Replace the method with our patched version
    qc.db_handler.similarity_search_with_score = new_similarity_search

patch_similarity_search(query_components_1)
patch_similarity_search(query_components_2)

# Fetch unique titles for filtering (try from DB1, fallback to DB2)
unique_titles = []
if query_components_1:
    unique_titles = query_components_1.get_unique_titles()
if not unique_titles and query_components_2: # Fallback if DB1 failed or had no titles
     unique_titles = query_components_2.get_unique_titles()

# Title Filter - Changed to single select instead of multiselect
st.sidebar.subheader("Филтер за докумената")
st.sidebar.info("Изаберите један документ на којем ће се извршити упит.")

# Extract just the document ID from the full title for cleaner display
def format_title(title):
    if title == "":
        return "Сви документи"
    # Try to extract just the document ID (e.g., "1605.04278" from longer paths)
    match = re.search(r'(\d{4}\.\d{5})', title)
    if match:
        return f"Рад {match.group(1)}"
    return title

selected_title = st.sidebar.selectbox(
    "Филтрирај по документу:",
    options=[""] + unique_titles,  # Add empty option for no filter
    format_func=format_title,
    help="Изаберите један документ да ограничите претрагу само на тај документ.",
)

# Display original document if one is selected
if selected_title:
    st.subheader("📄 Оригинални документ")
    try:
        # Extract paper ID using regex
        paper_id_match = re.search(r'(\d{4}\.\d{5})', selected_title)
        if paper_id_match:
            paper_id = paper_id_match.group(1)
            # Try to find and read the original document
            doc_path = f"data/qasper/{paper_id}/{paper_id}.md"
            try:
                with open(doc_path, "r") as f:
                    original_text = f.read()
                with st.expander(f"Оригинални текст - Рад {paper_id}", expanded=True):
                    st.markdown(original_text)
            except FileNotFoundError:
                st.info(f"Оригинални документ није пронађен на путањи {doc_path}")
    except Exception as e:
        st.error(f"Грешка при учитавању оригиналног документа: {e}")

# --- Query Input Form ---
with st.form(key="query_form"):
    user_input = st.text_input("Унесите питање на енглеском:", placeholder="нпр. What word level and character level model baselines are used?")
    submit_button = st.form_submit_button(label="Претражи и упореди")

# Initialize state fields for results
for field in ["run_id_1", "run_id_2", "last_result_1", "last_result_2", "last_user_input", "trace_urls"]:
    if field not in st.session_state:
        st.session_state[field] = None

# --- Query Execution Logic ---
def format_docs_for_llm(docs: List[Tuple[Document, float]]) -> str:
    """Formats retrieved documents for the LLM context."""
    formatted = []
    for i, (doc, score) in enumerate(docs):
        metadata_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
        formatted.append(f"Document {i+1} (Score: {score:.4f}):\nMetadata: {metadata_str}\nContent:\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def format_docs_for_display(docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    """Formats retrieved documents for display in Streamlit, preserving structure."""
    display_list = []
    for i, (doc, score) in enumerate(docs):
        display_list.append({
            "number": i + 1,
            "score": score,
            "metadata": doc.metadata,
            "content": doc.page_content,
            "title": doc.metadata.get("title", doc.metadata.get("source", "Unknown Title")) # Extract title
        })
    return display_list


def run_full_query(query: str, components: QueryComponents, paper_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs retrieval and LLM generation for a single database.
    """
    # 1. Retrieval
    print(f"[{components.collection_name}] Retrieving documents for query: '{query}'")
    
    # Log the paper_id filter if it's being used
    if paper_id:
        print(f"[{components.collection_name}] Filtering by paper_id: {paper_id}")
    
    # Use the paper_id parameter directly with the ChromaDBHandler
    try:
        retrieved_docs_with_scores = components.db_handler.similarity_search_with_score(
            query=query,
            k=SEARCH_RESULTS_NUM,
            paper_id=paper_id
        )
        
        # Check if we got any results with the filter
        if not retrieved_docs_with_scores and paper_id:
            print(f"[{components.collection_name}] No documents retrieved with filter. Trying direct query...")
            
            # Try a direct query to the database with a more flexible approach
            try:
                # Extract paper ID using regex
                paper_id_match = re.search(r'(\d{4}\.\d{5})', paper_id)
                regex_id = paper_id_match.group(1) if paper_id_match else None
                
                if regex_id:
                    print(f"[{components.collection_name}] Trying direct query with ID: {regex_id}")
                    
                    # Try a more direct approach - get all documents and filter manually
                    all_docs = components.db_handler.db.similarity_search(
                        query=query,
                        k=SEARCH_RESULTS_NUM * 5  # Get enough documents to filter from
                    )
                    
                    # Manually filter the documents - STRICT MATCHING
                    filtered_docs = []
                    for doc in all_docs:
                        # Check all metadata fields for an exact match with the regex_id
                        paper_id_value = doc.metadata.get('paper_id', '')
                        source_value = doc.metadata.get('source', '')
                        title_value = doc.metadata.get('title', '')
                        
                        # Only include documents that have an EXACT match with the regex_id
                        if (paper_id_value == regex_id or
                            regex_id in source_value or  # Allow partial match in source since it might be a path
                            f"/{regex_id}/" in source_value):  # Match directory pattern
                            
                            print(f"[{components.collection_name}] Found exact matching document: {source_value}")
                            # Assign a dummy score since we don't have the actual similarity score
                            filtered_docs.append((doc, 0.95))
                    
                    if filtered_docs:
                        print(f"[{components.collection_name}] Manual filtering found {len(filtered_docs)} documents with EXACT match")
                        retrieved_docs_with_scores = filtered_docs
            except Exception as e:
                print(f"[{components.collection_name}] Error in fallback query: {e}")
        
        # Log the number of documents retrieved and their titles for debugging
        doc_titles = [doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                     for doc, _ in retrieved_docs_with_scores]
        print(f"[{components.collection_name}] Retrieved {len(retrieved_docs_with_scores)} documents:")
        for i, title in enumerate(doc_titles[:5]):  # Log first 5 titles
            print(f"  - Doc {i+1}: {title}")
        if len(doc_titles) > 5:
            print(f"  - ... and {len(doc_titles) - 5} more")
            
    except Exception as e:
        print(f"[{components.collection_name}] Error in similarity search: {e}")
        retrieved_docs_with_scores = []
    print(f"[{components.collection_name}] Retrieved {len(retrieved_docs_with_scores)} documents initially.")

    # We already limited to top N in the similarity search
    final_docs_with_scores = retrieved_docs_with_scores
    print(f"[{components.collection_name}] Retrieved {len(final_docs_with_scores)} documents.")


    # 3. Prepare for Display and LLM
    display_docs = format_docs_for_display(final_docs_with_scores)
    context_for_llm = format_docs_for_llm(final_docs_with_scores) # Pass docs with scores for context

    # 4. LLM Generation (Summary/Answer)
    summary = "No documents retrieved or LLM generation failed."
    if final_docs_with_scores:
        print(f"[{components.collection_name}] Generating summary/answer...")
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question:
{question}

Answer the question based only on the provided context. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Answer:"""
        )
        prompt = prompt_template.format(question=query, context=context_for_llm)
        try:
            response = components.llm.invoke(prompt)
            summary = response.content
            print(f"[{components.collection_name}] Summary generated successfully.")
        except Exception as e:
            summary = f"Error generating summary: {e}"
            print(f"ERROR: [{components.collection_name}] LLM generation failed: {e}")

    return {
        "summary": summary,
        "documents_for_display": display_docs, # Use this structured list for display
        "raw_context": context_for_llm, # Raw context passed to LLM (for debugging maybe)
        # "run_id": Needs to be captured via LangSmith callback context
    }


# --- Run Query on Submit ---
if submit_button and user_input.strip():
    with st.spinner("Обрада упита и поређење резултата..."):
        try:
            metadata_filter = {}
            paper_id = None
            if selected_title:
                # Use the selected title as the paper_id for filtering
                paper_id = selected_title
                print(f"Applying filter for document: {paper_id}")

            results = {}
            trace_urls = {}
            run_ids = {}

            # Define functions to run each query within its own tracing context
            def run_with_tracing(query: str, components: QueryComponents, paper_id: Optional[str], db_key: int):
                 try:
                     with tracing_v2_enabled(project_name=LANGSMITH_PROJECT) as cb:
                         result_data = run_full_query(query, components, paper_id)
                         run_id = cb.latest_run.id if cb.latest_run else str(uuid.uuid4())  # Get run ID from callback
                         try:
                             trace_url = cb.get_run_url()
                         except Exception as e:
                             print(f"WARNING: No traced run found: {e}")
                             trace_url = "N/A"
                         print(f"[{components.collection_name}] LangSmith Run ID: {run_id}, Trace URL: {trace_url}")
                         return result_data, run_id, trace_url
                 except Exception as e:
                     print(f"WARNING: Tracing failed: {e}")
                     result_data = run_full_query(query, components, paper_id)
                     return result_data, "N/A", "N/A"

            # Execute both queries in parallel with individual error handling
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    1: executor.submit(run_with_tracing, user_input, query_components_1, paper_id, 1),
                    2: executor.submit(run_with_tracing, user_input, query_components_2, paper_id, 2)
                }
                for key, future in futures.items():
                    try:
                        res, rid, turl = future.result()
                    except Exception as e:
                        print(f"WARNING: Query for DB {key} failed: {e}")
                        res, rid, turl = {}, "N/A", "N/A"
                    results[key] = res
                    run_ids[key] = rid
                    trace_urls[key] = turl

            # Store results in session state
            st.session_state["run_id_1"] = run_ids[1]
            st.session_state["run_id_2"] = run_ids[2]
            st.session_state["last_result_1"] = results[1]
            st.session_state["last_result_2"] = results[2]
            st.session_state["last_user_input"] = user_input
            st.session_state["trace_urls"] = trace_urls
            st.session_state["results_available"] = True # Flag to indicate results are ready

        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"ERROR: Query processing failed: {e}")
            st.session_state["results_available"] = False


# --- Display Results ---
if st.session_state.get("results_available"):
    col1, col2 = st.columns(2)
    results = {1: st.session_state["last_result_1"], 2: st.session_state["last_result_2"]}
    trace_urls = st.session_state["trace_urls"]
    selected_collections = {1: st.session_state["current_collection_1"], 2: st.session_state["current_collection_2"]}
    query_components = {1: st.session_state["query_components_1"], 2: st.session_state["query_components_2"]}

    # --- Display Loop for Both Columns ---
    for db_key, col, result_data in [(1, col1, results[1]), (2, col2, results[2])]:
        with col:
            collection_name = selected_collections[db_key]
            st.header(f"База {db_key}: {collection_descriptions.get(collection_name, collection_name)}")

            summary_tab, documents_tab, system_tab = st.tabs(["Генерисани одговор", "Пронађени чанкови", "Системске информације"])

            with summary_tab:
                st.subheader("💬 LLM генерисани одговор (на енг.)")
                st.write(result_data.get("summary", "Генерисани одговор није доступан."))
                # Feedback requires run_id associated with the summary generation step
                # This needs adjustment if summary generation isn't part of the main traced run.
                # run_id = st.session_state.get(f"run_id_{db_key}")
                # if run_id:
                #    # Add feedback logic here if needed, similar to original app
                #    pass

            with documents_tab:
                st.subheader(f"📄 Пронађени чанкови (Најбољих {len(result_data.get('documents_for_display', []))})")
                retrieved_docs = result_data.get('documents_for_display', [])
                if retrieved_docs:
                    for doc_info in retrieved_docs:
                        doc_num = doc_info['number']
                        doc_title = doc_info['title']
                        doc_score = doc_info['score']
                        doc_content = doc_info['content']
                        doc_metadata = doc_info['metadata']

                        # Prepare metadata display string (excluding potentially long fields like 'content' or 'text')
                        meta_display = {k: v for k, v in doc_metadata.items() if k not in ['content', 'text', 'summary', 'title', 'source']} # Exclude common large fields
                        meta_str = ", ".join([f"`{k}`: {v}" for k, v in meta_display.items() if v is not None])


                        expander_title = f"Документ {doc_num}: {doc_title} (Оцена: {doc_score:.4f})"
                        with st.expander(expander_title):
                             st.markdown(f"**Метаподаци:** {meta_str}", unsafe_allow_html=True)
                             st.markdown("**Садржај:**")
                             st.markdown(doc_content)

                else:
                    st.write("Нема пронађених чанкова.")

            with system_tab:
                st.subheader("⚙️ Конфигурација система")
                components = query_components[db_key]
                embed_model_info = components.embedding_model.__class__.__name__
                if hasattr(components.embedding_model, 'model_name'):
                     embed_model_info += f" ({components.embedding_model.model_name})"

                system_info = {
                    "Колекција": collection_name,
                    "Метод парчања текста": collection_descriptions.get(collection_name, "Непознато"),
                    "Модел ембединга": embed_model_info,
                    "Модел за генерисање одговора": components.llm.model_name if hasattr(components.llm, 'model_name') else components.llm.__class__.__name__,
                    "Температура модела за генерисање одговора": components.llm.temperature if hasattr(components.llm, 'temperature') else "Н/А",
                    "Број резултата": SEARCH_RESULTS_NUM,
                    "Директоријум базе": DB_DIR,
                }
                st.json(system_info)

                # Display LangSmith trace link
                trace_url = trace_urls.get(db_key)
                if trace_url:
                    st.markdown(f"[Погледај траг у LangSmith-у]({trace_url})")
                else:
                     st.markdown("LangSmith URL траг није доступан.")

    # --- Chunking Methods Comparison Table ---
    st.markdown("---")
    with st.expander("📊 Преглед метода парчања текста"):
         # Build markdown table from available_collections with descriptions
        table_md = "| Назив колекције | Назив методе парчања текста | Опис методе парчања текста |\n"
        table_md += "|-----------------|----------------------------|-----------------------------|\n"
        
        # Define detailed descriptions for each method
        descriptions = {
            "qasper-recursive_character": "Користи рекурзивно парчање текста на основу фиксних сепаратора са дефинисаном величином и преклапањем, али не узима у обзир семантичку структуру.",
            "qasper-hierarchical": "Парчи текст тако да сваки сегмент почиње са насловом или поднасловом, задржавајући оригиналну хијерархијску структуру документа.",
            "qasper-semantic_clustering": "Групише реченице у кластерима користећи ембединг моделе и K-means алгоритам, уз динамичко одређивање броја кластера и узимање у обзир позиције реченица.",
            "qasper-sentence_transformers": "Динамички спаја реченице у кохерентне chunk-ове користећи косинусну сличност ембединг вектора и технику клизећег прозора."
        }
        
        # Add each collection to the table with its description
        for coll in available_collections:
            description = descriptions.get(coll["name"], "Нема доступног описа.")
            table_md += f"| `{coll['name']}` | {coll['description']} | {description} |\n"
        
        st.markdown(table_md)


elif submit_button:
    # Handle case where submit was pressed but results are not available (e.g., empty query)
    if not user_input.strip():
       st.warning("Молимо вас да унесете питање.")
    # Error handled during query execution

# --- Initial Info Message ---
if not st.session_state.get("results_available"): # Show if no results yet
    st.info("""
    **Добродошли у алат за поређење RAG стратегија парчања текста!**

    1.  Изаберите две различите **стратегије парчања текста** (векторске базе) из бочне падајуће листе.
    2.  Филтрирајте по одређеном **наслову документа**.
    3.  Унесите **питање** у поље за текст изнад.
    4.  Кликните на **Претражи и упореди**.

    Резултати ће приказати LLM генерисани одговор и пронађене чанкове један испод другог за обе изабране стратегије парчања текста.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Развијено за академско истраживање метода парчања текста у RAG системима. Користи Streamlit, LangChain, ChromaDB и изабране LLM/Embedding моделе.")
