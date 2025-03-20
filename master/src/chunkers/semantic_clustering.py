"""
SemanticClusteringChunker module for the RAG system.

This module implements the SemanticClusteringChunker class, which uses
sentence embeddings and K-means clustering to group semantically similar
sentences into chunks, with added time-sensitivity to maintain chronological order.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from langchain.schema import Document
from chunkers.base import BaseChunker

# Download the punkt tokenizer if not already downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    # Use trusted_host to avoid SSL certificate issues
    nltk.download("punkt", quiet=True, raise_on_error=False)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    # Use trusted_host to avoid SSL certificate issues
    nltk.download("punkt_tab", quiet=True, raise_on_error=False)


class SemanticClusteringChunker(BaseChunker):
    """
    A chunker that uses sentence-transformers embeddings + k-means
    to group semantically similar sentences into chunk(s), with
    time-sensitivity to maintain chronological order.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        max_chunk_size: int = 200,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
        position_weight: float = 0.2,  # Weight for position information
        preserve_order: bool = True,  # Whether to preserve chronological order
        **kwargs,
    ):
        """
        Initialize the SemanticClusteringChunker.

        Args:
            embedding_model_name (str): The sentence-transformers model name.
            max_chunk_size (int): Maximum number of words in a single chunk (heuristic).
            min_clusters (int): Lower bound for searching the optimal number of clusters.
            max_clusters (int): Upper bound for searching the optimal number of clusters.
            random_state (int): For reproducible clustering.
            position_weight (float): Weight for position information (0-1).
                Higher values give more importance to chronological order.
            preserve_order (bool): Whether to preserve chronological order within clusters.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(max_chunk_size=max_chunk_size, **kwargs)

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.position_weight = position_weight
        self.preserve_order = preserve_order

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Main method: for each Document, splits into sentences, embeds them,
        finds the best K (clusters), groups by cluster, forms final chunk(s).

        Uses batch processing for embeddings to improve performance.
        Incorporates time-sensitivity to maintain chronological order.

        Args:
            docs (List[Document]): List of Document objects to be chunked.

        Returns:
            List[Document]: A list of Document objects where each document
            represents a semantically coherent chunk with the original metadata preserved.
        """
        try:
            all_chunks = []

            print(f"SemanticClusteringChunker: Processing {len(docs)} documents")

            # Process each document individually to reduce memory usage
            for doc_idx, doc in enumerate(docs):
                try:
                    # Check if document is very large (more than 1000 sentences)
                    text = doc.page_content
                    print(f"  Document {doc_idx+1}: Length {len(text)} characters")

                    # Make sure NLTK punkt is available
                    try:
                        sentences_text = [
                            s.strip() for s in sent_tokenize(text) if s.strip()
                        ]
                        print(f"  Extracted {len(sentences_text)} sentences")
                    except Exception as e:
                        print(f"  Error tokenizing sentences: {e}")
                        # Fallback to simple splitting by periods
                        sentences_text = [
                            s.strip() for s in text.split(".") if s.strip()
                        ]
                        print(
                            f"  Fallback: Extracted {len(sentences_text)} sentences by splitting on periods"
                        )

                    if not sentences_text:
                        print(f"  No sentences found in document {doc_idx+1}, skipping")
                        continue

                    if len(sentences_text) == 1:
                        # Only 1 sentence, just make it a chunk
                        all_chunks.append(
                            Document(
                                page_content=sentences_text[0], metadata=doc.metadata
                            )
                        )
                        print(
                            f"  Document {doc_idx+1} has only 1 sentence, added as a chunk"
                        )
                        continue
                except Exception as doc_error:
                    print(f"  Error processing document {doc_idx+1}: {doc_error}")
                    continue

                # For very large documents, process in chunks to avoid memory issues
                if len(sentences_text) > 1000:
                    print(
                        f"  Document {doc_idx+1} is very large ({len(sentences_text)} sentences), processing in sections"
                    )
                    doc_chunks = self._process_large_document(
                        sentences_text, doc.metadata
                    )
                    print(f"  Created {len(doc_chunks)} chunks from large document")
                    all_chunks.extend(doc_chunks)
                    continue

                # For regular-sized documents, process normally with batch embedding
                print(f"  Generating embeddings for {len(sentences_text)} sentences")
                batch_size = 32
                doc_embeddings = []

                # Process embeddings in batches
                for i in range(0, len(sentences_text), batch_size):
                    batch = sentences_text[i : min(i + batch_size, len(sentences_text))]
                    try:
                        batch_embeddings = self.embedding_model.encode(
                            batch, show_progress_bar=False
                        )
                        doc_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        print(f"  Error encoding batch {i//batch_size + 1}: {e}")
                        # Skip this batch
                        continue

                if not doc_embeddings:
                    print(
                        f"  No embeddings generated for document {doc_idx+1}, skipping"
                    )
                    continue

                # Convert to numpy array
                doc_embeddings = np.array(doc_embeddings)
                print(f"  Generated embeddings with shape {doc_embeddings.shape}")

                # Add time-sensitivity to embeddings
                time_weighted_embeddings = self._add_position_information(
                    doc_embeddings
                )
                print(f"  Added position information to embeddings")

                # Find optimal K in [min_clusters, max_clusters] via silhouette
                best_k = self._select_optimal_k(time_weighted_embeddings)
                print(f"  Selected optimal k={best_k} clusters")

                # K-means clustering with time-weighted embeddings
                kmeans = KMeans(n_clusters=best_k, random_state=self.random_state)
                labels = kmeans.fit_predict(time_weighted_embeddings)
                print(f"  Performed K-means clustering")

                # Group sentences per cluster, preserving original indices
                cluster_dict = {}
                for i, (label, sentence) in enumerate(
                    zip(labels, sentences_text[: len(labels)])
                ):
                    if label not in cluster_dict:
                        cluster_dict[label] = []
                    # Store sentence along with its original position
                    cluster_dict[label].append((i, sentence))

                print(f"  Grouped sentences into {len(cluster_dict)} clusters")

                # Form final chunks with max_chunk_size limit
                doc_chunks_count = 0
                for cl_label, sentence_items in cluster_dict.items():
                    # Sort by original position if preserve_order is True
                    if self.preserve_order:
                        sentence_items.sort(key=lambda x: x[0])

                    # Extract just the sentences after sorting
                    sentence_list = [item[1] for item in sentence_items]

                    chunk_words = 0
                    current_chunk = []
                    for sent in sentence_list:
                        sent_len = len(sent.split())
                        if (chunk_words + sent_len) <= self.max_chunk_size:
                            current_chunk.append(sent)
                            chunk_words += sent_len
                        else:
                            # close current chunk
                            chunk_text = " ".join(current_chunk).strip()
                            if chunk_text:
                                all_chunks.append(
                                    Document(
                                        page_content=chunk_text, metadata=doc.metadata
                                    )
                                )
                                doc_chunks_count += 1
                            # start a new chunk
                            current_chunk = [sent]
                            chunk_words = sent_len

                    # any leftover
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            all_chunks.append(
                                Document(page_content=chunk_text, metadata=doc.metadata)
                            )
                            doc_chunks_count += 1

                print(f"  Created {doc_chunks_count} chunks from document {doc_idx+1}")

            return all_chunks
        except Exception as e:
            print(f"Error during semantic clustering chunking: {e}")
            return []

    def _process_large_document(
        self, sentences: List[str], metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Process a very large document by splitting it into manageable sections.

        This method divides a large document into sections of 500 sentences each,
        processes each section separately, and then combines the results.

        Args:
            sentences (List[str]): List of all sentences in the document.
            metadata (Dict[str, Any]): Metadata from the original document.

        Returns:
            List[Document]: List of Document objects representing chunks.
        """
        section_chunks = []
        section_size = 500  # Process 500 sentences at a time

        # Add a note to metadata that document was processed in sections
        section_metadata = metadata.copy()
        section_metadata["large_document_processing"] = "sectioned"

        # Process document in sections
        for i in range(0, len(sentences), section_size):
            section = sentences[i : min(i + section_size, len(sentences))]

            # Update metadata with section information
            current_section_metadata = section_metadata.copy()
            current_section_metadata["section_index"] = i // section_size
            current_section_metadata["total_sections"] = (
                len(sentences) + section_size - 1
            ) // section_size

            # Process this section
            batch_size = 32
            section_embeddings = []

            # Process embeddings in batches
            for j in range(0, len(section), batch_size):
                batch = section[j : min(j + batch_size, len(section))]
                batch_embeddings = self.embedding_model.encode(
                    batch, show_progress_bar=False
                )
                section_embeddings.extend(batch_embeddings)

            # Convert to numpy array
            section_embeddings = np.array(section_embeddings)

            # Add time-sensitivity to embeddings
            time_weighted_embeddings = self._add_position_information(
                section_embeddings
            )

            # Find optimal K
            best_k = self._select_optimal_k(time_weighted_embeddings)

            # K-means clustering
            kmeans = KMeans(n_clusters=best_k, random_state=self.random_state)
            labels = kmeans.fit_predict(time_weighted_embeddings)

            # Group sentences per cluster
            cluster_dict = {}
            for j, (label, sentence) in enumerate(zip(labels, section)):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append((j, sentence))

            # Form chunks
            for cl_label, sentence_items in cluster_dict.items():
                if self.preserve_order:
                    sentence_items.sort(key=lambda x: x[0])

                sentence_list = [item[1] for item in sentence_items]

                chunk_words = 0
                current_chunk = []
                for sent in sentence_list:
                    sent_len = len(sent.split())
                    if (chunk_words + sent_len) <= self.max_chunk_size:
                        current_chunk.append(sent)
                        chunk_words += sent_len
                    else:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            section_chunks.append(
                                Document(
                                    page_content=chunk_text,
                                    metadata=current_section_metadata,
                                )
                            )
                        current_chunk = [sent]
                        chunk_words = sent_len

                if current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        section_chunks.append(
                            Document(
                                page_content=chunk_text,
                                metadata=current_section_metadata,
                            )
                        )

        return section_chunks

    def _add_position_information(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Add position information to embeddings to maintain chronological order.

        Args:
            embeddings (np.ndarray): Original sentence embeddings.

        Returns:
            np.ndarray: Embeddings with added position information.
        """
        n_samples = embeddings.shape[0]
        if n_samples <= 1:
            return embeddings

        # Create time-weighted embeddings
        time_weighted_embeddings = []

        for i, embedding in enumerate(embeddings):
            # Create a normalized position vector (0 to 1)
            position = i / (n_samples - 1)

            # Normalize the original embedding to have a consistent scale
            norm_embedding = (
                embedding / np.linalg.norm(embedding)
                if np.linalg.norm(embedding) > 0
                else embedding
            )

            # Scale the embeddings based on position weight
            scaled_embedding = norm_embedding * (1 - self.position_weight)

            # Create position features (using a simple approach with position as a feature)
            position_feature = np.array([position])

            # Concatenate the scaled embedding with the position feature
            # We need to ensure the position feature has appropriate weight
            position_feature_scaled = (
                position_feature * self.position_weight * np.linalg.norm(norm_embedding)
            )

            # Create the final embedding by appending the position feature
            final_embedding = np.concatenate(
                [scaled_embedding, position_feature_scaled]
            )

            time_weighted_embeddings.append(final_embedding)

        return np.array(time_weighted_embeddings)

    def _select_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Select the optimal number of clusters (k) using silhouette scoring.

        Args:
            embeddings (np.ndarray): Array of sentence embeddings.

        Returns:
            int: The optimal number of clusters.
        """
        n_samples = embeddings.shape[0]
        # If we have 2 or fewer sentences, clustering doesn't make sense
        if n_samples <= 2:
            return 1

        max_k = min(self.max_clusters, n_samples - 1)
        if max_k < 2:
            return 1

        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(embeddings)
            # Check if it makes sense to calculate silhouette_score
            # (there must be at least 2 unique labels)
            if len(np.unique(labels)) >= 2:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chunker.

        Returns:
            Dict[str, Any]: A dictionary containing the chunker's configuration.
        """
        config = super().get_config()
        config.update(
            {
                "embedding_model_name": self.embedding_model_name,
                "min_clusters": self.min_clusters,
                "max_clusters": self.max_clusters,
                "random_state": self.random_state,
                "position_weight": self.position_weight,
                "preserve_order": self.preserve_order,
            }
        )
        return config
