"""
SentenceTransformersSplitter module for the RAG system.

This module implements the SentenceTransformersSplitter class, which splits
documents by sentences and uses embeddings to merge similar sentences,
with an improved sliding window approach for better context awareness.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from langchain.schema import Document
from chunkers.base import BaseChunker

# Download the punkt tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # Use trusted_host to avoid SSL certificate issues
    nltk.download('punkt', quiet=True, raise_on_error=False)


class SentenceTransformersSplitter(BaseChunker):
    """
    A chunker that splits documents by sentences and uses embeddings to merge similar sentences.
    
    This chunker first splits documents into sentences, then computes embeddings for each sentence,
    and finally merges sentences based on their semantic similarity until a maximum chunk size
    is reached. It uses a sliding window approach to maintain better context awareness.
    """
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        max_chunk_size: int = 200,
        similarity_threshold: float = 0.75,
        window_size: int = 3,  # Size of sliding window for context
        adaptive_threshold: bool = True,  # Whether to use adaptive thresholds
        **kwargs
    ):
        """
        Initialize the SentenceTransformersSplitter.
        
        Args:
            embedding_model_name (str): The sentence-transformers model name.
            max_chunk_size (int): Maximum number of words in a single chunk.
            similarity_threshold (float): Threshold for merging sentences (0.0 to 1.0).
            window_size (int): Number of recent sentences to consider for context.
            adaptive_threshold (bool): Whether to adapt threshold based on content.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            **kwargs
        )
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.adaptive_threshold = adaptive_threshold
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents by splitting into sentences and merging similar ones.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects where each document
            represents a chunk with semantically similar sentences.
        """
        try:
            all_chunks = []
            
            print(f"SentenceTransformersSplitter: Processing {len(docs)} documents")
            
            for doc_idx, doc in enumerate(docs):
                try:
                    # Split document into sentences
                    text = doc.page_content
                    print(f"  Document {doc_idx+1}: Length {len(text)} characters")
                    
                    try:
                        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
                        print(f"  Extracted {len(sentences)} sentences")
                    except Exception as e:
                        print(f"  Error tokenizing sentences: {e}")
                        # Fallback to simple splitting by periods
                        sentences = [s.strip() for s in text.split(".") if s.strip()]
                        print(f"  Fallback: Extracted {len(sentences)} sentences by splitting on periods")
                    
                    if not sentences:
                        print(f"  No sentences found in document {doc_idx+1}, skipping")
                        continue
                        
                    if len(sentences) == 1:
                        # Only one sentence, just make it a chunk
                        all_chunks.append(Document(page_content=sentences[0], metadata=doc.metadata))
                        print(f"  Document {doc_idx+1} has only 1 sentence, added as a chunk")
                        continue
                    
                    # Compute embeddings for all sentences
                    print(f"  Computing embeddings for {len(sentences)} sentences")
                    try:
                        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
                        print(f"  Generated embeddings with shape {embeddings.shape}")
                    except Exception as e:
                        print(f"  Error computing embeddings: {e}")
                        # If embeddings fail, fall back to simple chunking
                        for sentence in sentences:
                            all_chunks.append(Document(page_content=sentence, metadata=doc.metadata))
                        print(f"  Fallback: Created {len(sentences)} chunks (one per sentence)")
                        continue
                    
                    # Create chunks by merging similar sentences
                    doc_chunks = self._merge_similar_sentences(sentences, embeddings, doc.metadata)
                    print(f"  Created {len(doc_chunks)} chunks from document {doc_idx+1}")
                    all_chunks.extend(doc_chunks)
                    
                except Exception as doc_error:
                    print(f"  Error processing document {doc_idx+1}: {doc_error}")
                    continue
            
            print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
            if not all_chunks:
                print("WARNING: No chunks were created! Falling back to simple chunking")
                # Fallback to simple chunking if no chunks were created
                for doc in docs:
                    # Split into paragraphs and create a chunk for each paragraph
                    paragraphs = [p.strip() for p in doc.page_content.split('\n\n') if p.strip()]
                    if not paragraphs:
                        # If no paragraphs, just use the whole document
                        all_chunks.append(Document(page_content=doc.page_content, metadata=doc.metadata))
                    else:
                        for para in paragraphs:
                            all_chunks.append(Document(page_content=para, metadata=doc.metadata))
                print(f"Fallback created {len(all_chunks)} chunks")
            
            return all_chunks
        except Exception as e:
            print(f"Error during sentence transformers chunking: {e}")
            # Fallback to returning the original documents
            print("Critical error, returning original documents")
            return docs
    
    def _merge_similar_sentences(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray,
        metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Merge similar sentences into chunks based on semantic similarity,
        using a sliding window approach for better context awareness.
        
        Args:
            sentences (List[str]): List of sentences to merge.
            embeddings (np.ndarray): Array of sentence embeddings.
            metadata (Dict[str, Any]): Metadata to preserve in the chunks.
            
        Returns:
            List[Document]: List of Document objects representing chunks.
        """
        chunks = []
        current_chunk = []
        current_chunk_words = 0
        
        # Compute pairwise similarities between sentences
        similarities = cosine_similarity(embeddings)
        
        # Start with the first sentence
        current_chunk.append(sentences[0])
        current_chunk_words += len(sentences[0].split())
        processed = [0]
        
        # Calculate adaptive threshold if enabled
        if self.adaptive_threshold:
            # Calculate average similarity between adjacent sentences
            adjacent_similarities = [similarities[i, i+1] for i in range(len(sentences)-1)]
            avg_similarity = np.mean(adjacent_similarities) if adjacent_similarities else 0.5
            # Adjust threshold based on document characteristics
            # Lower threshold for documents with lower average similarity
            adjusted_threshold = max(0.4, min(0.8, avg_similarity * 0.9))
            threshold = adjusted_threshold
            print(f"  Using adaptive threshold: {threshold:.2f} (avg similarity: {avg_similarity:.2f})")
        else:
            threshold = self.similarity_threshold
            print(f"  Using fixed threshold: {threshold:.2f}")
        
        while len(processed) < len(sentences):
            # Find the most similar unprocessed sentence to any sentence in the current chunk
            # using a sliding window of the most recent sentences for better context
            max_similarity = -1
            next_sentence_idx = -1
            
            # Consider only the most recent window_size sentences in the current chunk
            window_start = max(0, len(processed) - self.window_size)
            
            for i in processed[window_start:]:
                for j in range(len(sentences)):
                    if j not in processed and similarities[i, j] > max_similarity:
                        max_similarity = similarities[i, j]
                        next_sentence_idx = j
            
            # If we found a similar sentence and it's above the threshold
            if max_similarity >= threshold:
                next_sentence = sentences[next_sentence_idx]
                next_sentence_words = len(next_sentence.split())
                
                # Check if adding this sentence would exceed the max chunk size
                if current_chunk_words + next_sentence_words <= self.max_chunk_size:
                    current_chunk.append(next_sentence)
                    current_chunk_words += next_sentence_words
                    processed.append(next_sentence_idx)
                else:
                    # Create a new chunk with the current sentences
                    chunk_text = " ".join(current_chunk).strip()
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
                    
                    # Start a new chunk with the next sentence
                    current_chunk = [next_sentence]
                    current_chunk_words = next_sentence_words
                    processed.append(next_sentence_idx)
            else:
                # No similar sentences found, create a chunk with what we have
                chunk_text = " ".join(current_chunk).strip()
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
                
                # Find the first unprocessed sentence to start a new chunk
                for i in range(len(sentences)):
                    if i not in processed:
                        current_chunk = [sentences[i]]
                        current_chunk_words = len(sentences[i].split())
                        processed.append(i)
                        break
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
        return chunks
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chunker.
        
        Returns:
            Dict[str, Any]: A dictionary containing the chunker's configuration.
        """
        config = super().get_config()
        config.update({
            "embedding_model_name": self.embedding_model_name,
            "similarity_threshold": self.similarity_threshold,
            "window_size": self.window_size,
            "adaptive_threshold": self.adaptive_threshold,
        })
        return config