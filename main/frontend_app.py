import sys
import os

import streamlit as st
import uuid
import concurrent.futures
import re
import hashlib
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
st.set_page_config(page_title="RAG Chunking Comparison", layout="wide")

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


# --- Highlighting Helper Functions (Copied from original, assumed correct) ---
def preprocess_text(text: str) -> list:
    text = re.sub(r'[\W_]+', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    words = text.split()
    return words

def find_longest_common_substring(text1, text2, min_words=5):
    text1_norm = re.sub(r'\s+', ' ', text1.lower()).strip()
    text2_norm = re.sub(r'\s+', ' ', text2.lower()).strip()
    words1 = text1_norm.split()
    words2 = text2_norm.split()
    if not words1 or not words2: return ""
    dp = [[0 for _ in range(len(words2) + 1)] for _ in range(len(words1) + 1)]
    max_length = 0
    end_pos = 0
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    if max_length >= min_words:
        start_pos = end_pos - max_length
        # Find corresponding original text position more robustly
        orig_words1 = text1.split() # Split preserving original casing/spacing somewhat
        if start_pos < len(orig_words1) and end_pos <= len(orig_words1):
             # Simple join, might not be perfect with complex spacing/punctuation
             return ' '.join(orig_words1[start_pos:end_pos])
    return ""


def find_common_sequences(text1, text2, min_words=5):
    if not text1 or not text2: return []
    # Prioritize longest common substring first
    longest = find_longest_common_substring(text1, text2, min_words)
    if longest:
        # print(f"DEBUG: Found longest common: '{longest[:50]}...'")
        return [longest]

    # Fallback (less efficient, find any common sequence) - consider removing if LCS is enough
    text1_norm = re.sub(r'\s+', ' ', text1.lower()).strip()
    text2_norm = re.sub(r'\s+', ' ', text2.lower()).strip()
    words1 = text1_norm.split()
    if len(words1) < min_words: return []

    common_sequences = []
    text2_word_set = set(text2_norm.split()) # Quick check optimization

    for i in range(len(words1) - min_words + 1):
        current_sequence_words = words1[i : i + min_words]
        # Optimization: check if first word exists in text2
        if current_sequence_words[0] not in text2_word_set:
            continue

        sequence_str_norm = ' '.join(current_sequence_words)

        if sequence_str_norm in text2_norm:
             # Attempt to extend match (simple greedy extension)
            j = i + min_words
            extended_sequence_norm = sequence_str_norm
            while j < len(words1):
                next_word = words1[j]
                potential_extended_norm = extended_sequence_norm + ' ' + next_word
                if potential_extended_norm in text2_norm:
                    extended_sequence_norm = potential_extended_norm
                    j += 1
                else:
                    break
            # Find original text span for the final extended sequence
            # This part is tricky to get exactly right with original formatting
            # Using the normalized version for now
            # TODO: Improve original text extraction for this fallback case
            common_sequences.append(extended_sequence_norm)


    unique_sequences = list(set(common_sequences))
    unique_sequences.sort(key=len, reverse=True)
    # print(f"DEBUG: Fallback found {len(unique_sequences)} common sequences.")
    return unique_sequences

def generate_consistent_color(text):
    hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    # Make colors lighter pastel-like
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    # Ensure minimum brightness
    r = max(180, r)
    g = max(180, g)
    b = max(180, b)
    return f"#{r:02x}{g:02x}{b:02x}"


def highlight_common_sequences(text, common_sequences, color_map):
    if not common_sequences or not text:
        return text

    intervals = []
    lower_text = text.lower()

    # Sort sequences by length DESC to find longest matches first
    for seq in sorted(common_sequences, key=lambda s: -len(s)):
        seq_lower = seq.lower().strip()
        if not seq_lower: continue

        start = 0
        while True:
            idx = lower_text.find(seq_lower, start)
            if idx == -1: break
            end = idx + len(seq_lower)

            # Check for overlap with already added intervals
            is_overlapping = False
            for existing_start, existing_end, _ in intervals:
                # Check if the new interval is completely within an existing one
                if idx >= existing_start and end <= existing_end:
                    is_overlapping = True
                    break
                # Check for partial overlaps (more complex, simplified here)
                if not (end <= existing_start or idx >= existing_end):
                     # Basic overlap check: if they touch or intersect
                     # We prioritize the longer sequence already added (due to sorting)
                     is_overlapping = True
                     break # Don't add this shorter/later sequence if it overlaps

            if not is_overlapping:
                intervals.append((idx, end, color_map[seq]))
                # Move start significantly to avoid re-matching subsets within the found match
                start = end # Start next search after the current match ends
            else:
                 # If overlapping, still advance start to avoid infinite loops on partial overlaps
                 start = idx + 1


    intervals.sort(key=lambda x: x[0])

    # Filter out intervals fully contained within others (can happen with adjusted logic)
    filtered_intervals = []
    for i, (start1, end1, color1) in enumerate(intervals):
        is_contained = False
        for j, (start2, end2, color2) in enumerate(intervals):
            if i != j and start1 >= start2 and end1 <= end2:
                is_contained = True
                break
        if not is_contained:
            filtered_intervals.append((start1, end1, color1))

    # Rebuild text with highlights
    highlighted_text = ""
    last_index = 0
    for start, end, color in filtered_intervals: # Use filtered intervals
        # Ensure intervals are sequential and valid
        if start >= last_index:
             highlighted_text += text[last_index:start]
             highlighted_text += f'<span style="background-color: {color}; display: inline; border-radius: 3px; padding: 1px 2px;">{text[start:end]}</span>'
             last_index = end
        # else: print(f"WARN: Skipping overlapping or out-of-order interval: ({start}-{end}) vs last_index {last_index}")


    highlighted_text += text[last_index:]
    return highlighted_text
# --- End Highlighting Helpers ---

# --- Streamlit UI Setup ---
st.title("📊 RAG Chunking Strategy Comparison")
st.markdown(f"Comparing results from different vector stores in **{DB_DIR}**")

# Sidebar for Filters and Settings
st.sidebar.title("Settings & Filters")

# Collection Selection for two databases
st.sidebar.subheader("Select Databases to Compare")
st.sidebar.markdown("**Database 1**")
selected_collection_1 = st.sidebar.selectbox(
    "Choose Chunking Strategy 1:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{collection_descriptions.get(x, x)}",
    help="Select the first vector database collection (representing a chunking strategy).",
    key="db1"
)

st.sidebar.markdown("**Database 2**")
# Set default for second box to be different if possible
default_index_2 = 1 if len(available_collections) > 1 and available_collections[1]["name"] != selected_collection_1 else 0
if selected_collection_1 == available_collections[default_index_2]["name"] and len(available_collections) > 2:
    default_index_2 = 2 # Try the third one

selected_collection_2 = st.sidebar.selectbox(
    "Choose Chunking Strategy 2:",
    options=[c["name"] for c in available_collections],
    format_func=lambda x: f"{collection_descriptions.get(x, x)}",
    help="Select the second vector database collection for comparison.",
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
    with st.spinner(f"Loading DB 1: {collection_descriptions.get(selected_collection_1, selected_collection_1)}..."):
        st.session_state["query_components_1"] = initialize_query_components(selected_collection_1)
        st.session_state["current_collection_1"] = selected_collection_1
        # st.success(f"Loaded DB 1: {collection_descriptions.get(selected_collection_1, 'Unknown')}")

if "query_components_2" not in st.session_state or st.session_state.get("current_collection_2") != selected_collection_2:
    with st.spinner(f"Loading DB 2: {collection_descriptions.get(selected_collection_2, selected_collection_2)}..."):
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
st.sidebar.subheader("Document Filters")
st.sidebar.info("Select a single document below to focus your query on just that paper.")

# Extract just the document ID from the full title for cleaner display
def format_title(title):
    if title == "":
        return "All Documents"
    # Try to extract just the document ID (e.g., "1605.04278" from longer paths)
    match = re.search(r'(\d{4}\.\d{5})', title)
    if match:
        return f"Paper {match.group(1)}"
    return title

selected_title = st.sidebar.selectbox(
    "Filter by Document:",
    options=[""] + unique_titles,  # Add empty option for no filter
    format_func=format_title,
    help="Select a single document to restrict the search to only that document.",
)

# --- Query Input Form ---
with st.form(key="query_form"):
    user_input = st.text_input("Enter your query:", placeholder="e.g., What challenges were faced during the COVID-19 response?")
    submit_button = st.form_submit_button(label="Search & Compare")

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
    with st.spinner("Processing query and comparing results..."):
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


    # No highlighting processing needed when filtering by a single document
    all_common_sequences = []
    color_map = {}

    # --- Display Loop for Both Columns ---
    for db_key, col, result_data in [(1, col1, results[1]), (2, col2, results[2])]:
        with col:
            collection_name = selected_collections[db_key]
            st.header(f"DB {db_key}: {collection_descriptions.get(collection_name, collection_name)}")

            summary_tab, documents_tab, system_tab = st.tabs(["Summary", "Retrieved Docs", "System Info"])

            with summary_tab:
                st.subheader("💬 LLM Answer/Summary")
                st.write(result_data.get("summary", "No summary available."))
                # Feedback requires run_id associated with the summary generation step
                # This needs adjustment if summary generation isn't part of the main traced run.
                # run_id = st.session_state.get(f"run_id_{db_key}")
                # if run_id:
                #    # Add feedback logic here if needed, similar to original app
                #    pass

            with documents_tab:
                st.subheader(f"📄 Retrieved Documents (Top {len(result_data.get('documents_for_display', []))})")
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


                        expander_title = f"Doc {doc_num}: {doc_title} (Score: {doc_score:.4f})"
                        with st.expander(expander_title):
                             st.markdown(f"**Metadata:** {meta_str}", unsafe_allow_html=True)
                             st.markdown("**Content:**")
                             # Display content without highlighting
                             st.markdown(doc_content)

                else:
                    st.write("No documents retrieved.")

            with system_tab:
                st.subheader("⚙️ System Configuration")
                components = query_components[db_key]
                embed_model_info = components.embedding_model.__class__.__name__
                if hasattr(components.embedding_model, 'model_name'):
                     embed_model_info += f" ({components.embedding_model.model_name})"

                system_info = {
                    "Collection": collection_name,
                    "Chunking Method": collection_descriptions.get(collection_name, "Unknown"),
                    "Embedding Model": embed_model_info,
                    "Chat Model": components.llm.model_name if hasattr(components.llm, 'model_name') else components.llm.__class__.__name__,
                    "Chat Temperature": components.llm.temperature if hasattr(components.llm, 'temperature') else "N/A",
                    "Results K": SEARCH_RESULTS_NUM,
                    "DB Directory": DB_DIR,
                }
                st.json(system_info)

                # Display LangSmith trace link
                trace_url = trace_urls.get(db_key)
                if trace_url:
                    st.markdown(f"[View LangSmith Trace]({trace_url})")
                else:
                     st.markdown("LangSmith trace URL not available.")


    # No highlighting explanation needed

    # --- Chunking Methods Comparison Table ---
    st.markdown("---")
    with st.expander("📊 Chunking Methods Overview"):
         # Build markdown table from available_collections
        table_md = "| Collection Name | Chunking Method Description |\n"
        table_md += "|-----------------|-----------------------------|\n"
        for coll in available_collections:
            table_md += f"| `{coll['name']}` | {coll['description']} |\n"
        st.markdown(table_md)
        # Add more detailed explanations if desired


elif submit_button:
    # Handle case where submit was pressed but results are not available (e.g., empty query)
    if not user_input.strip():
       st.warning("Please enter a query.")
    # Error handled during query execution

# --- Initial Info Message ---
if not st.session_state.get("results_available"): # Show if no results yet
    st.info("""
    **Welcome to the RAG Chunking Comparison Tool!**

    1.  Select two different **Chunking Strategies** (Vector Databases) from the sidebar.
    2.  Optionally, filter by specific **Source Document Titles**.
    3.  Enter your **Query** in the text box above.
    4.  Click **Search & Compare**.

    The results will show summaries and retrieved documents side-by-side for both chunking strategies.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Developed for academic research on chunking methods in RAG systems. Uses Streamlit, LangChain, ChromaDB, and selected LLMs/Embedding models.")
