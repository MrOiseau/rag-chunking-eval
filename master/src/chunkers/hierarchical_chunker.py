"""
HierarchicalChunker module for the RAG system.

This module implements the HierarchicalChunker class, which creates a hierarchical
structure of chunks (parent-child relationships) based on document structure.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
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


class HierarchicalChunker(BaseChunker):
    """
    A chunker that creates a hierarchical structure of chunks.
    
    This chunker identifies document structure (headings, sections, paragraphs)
    and creates a hierarchical representation where each chunk knows its parent
    and children, allowing for more contextual retrieval.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 200,
        heading_patterns: Optional[List[str]] = None,
        include_headings: bool = True,
        **kwargs
    ):
        """
        Initialize the HierarchicalChunker.
        
        Args:
            max_chunk_size (int): Maximum number of words in a single chunk.
            heading_patterns (Optional[List[str]]): Regex patterns to identify headings.
            include_headings (bool): Whether to include headings in the chunk content.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(
            max_chunk_size=max_chunk_size,
            **kwargs
        )
        
        self.max_chunk_size = max_chunk_size
        self.include_headings = include_headings
        
        # Default heading patterns if none provided
        if heading_patterns is None:
            self.heading_patterns = [
                r'^#+\s+(.+)$',  # Markdown headings
                r'^(.+)\n[=]+\s*$',  # Underlined headings with =
                r'^(.+)\n[-]+\s*$',  # Underlined headings with -
                r'^(\d+\.?\d*\.?\s+.+)$',  # Numbered headings
            ]
        else:
            self.heading_patterns = heading_patterns
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Chunk documents by creating a hierarchical structure.
        
        Args:
            docs (List[Document]): List of Document objects to be chunked.
            
        Returns:
            List[Document]: A list of Document objects where each document
            represents a chunk with hierarchical metadata.
        """
        try:
            all_chunks = []
            
            print(f"HierarchicalChunker: Processing {len(docs)} documents")
            
            for doc_idx, doc in enumerate(docs):
                try:
                    # Extract the document structure
                    print(f"  Document {doc_idx+1}: Extracting structure")
                    sections = self._extract_document_structure(doc.page_content)
                    print(f"  Found {len(sections)} sections")
                    
                    # Create chunks from the sections
                    doc_chunks = self._create_chunks_from_sections(sections, doc.metadata)
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
            print(f"Error during hierarchical chunking: {e}")
            # Fallback to returning the original documents
            print("Critical error, returning original documents")
            return docs
    
    def _extract_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract the hierarchical structure of a document.
        
        Args:
            text (str): The document text.
            
        Returns:
            List[Dict[str, Any]]: A list of sections, each with a heading and content.
        """
        # Split the text into lines
        lines = text.split('\n')
        
        # Identify headings and their levels
        sections = []
        current_section = {"heading": "", "level": 0, "content": []}
        
        for line in lines:
            is_heading, heading_text, level = self._is_heading(line)
            
            if is_heading:
                # Save the previous section if it has content
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start a new section
                current_section = {
                    "heading": heading_text,
                    "level": level,
                    "content": []
                }
            else:
                # Add the line to the current section's content
                if line.strip():
                    current_section["content"].append(line)
        
        # Add the last section
        if current_section["content"]:
            sections.append(current_section)
        
        # If no sections were found, treat the whole document as one section
        if not sections:
            sections = [{
                "heading": "",
                "level": 0,
                "content": text.split('\n')
            }]
        
        return sections
    
    def _is_heading(self, line: str) -> Tuple[bool, str, int]:
        """
        Check if a line is a heading and determine its level.
        
        Args:
            line (str): The line to check.
            
        Returns:
            Tuple[bool, str, int]: A tuple containing:
                - Whether the line is a heading
                - The heading text (if it's a heading)
                - The heading level (if it's a heading)
        """
        for i, pattern in enumerate(self.heading_patterns):
            match = re.match(pattern, line)
            if match:
                heading_text = match.group(1).strip()
                # Determine the level based on the pattern index and any specific markers
                if pattern.startswith(r'^#+'):
                    # Count the number of # characters for Markdown headings
                    level = len(line) - len(line.lstrip('#'))
                else:
                    # For other patterns, use the pattern index as a proxy for level
                    level = i + 1
                
                return True, heading_text, level
        
        return False, "", 0
    
    def _create_chunks_from_sections(
        self, 
        sections: List[Dict[str, Any]], 
        base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Create chunks from the document sections.
        
        Args:
            sections (List[Dict[str, Any]]): List of document sections.
            base_metadata (Dict[str, Any]): Base metadata from the original document.
            
        Returns:
            List[Document]: List of Document objects representing chunks.
        """
        chunks = []
        
        # Keep track of the heading hierarchy
        heading_hierarchy = [""] * 10  # Assuming max 10 levels of headings
        
        for section in sections:
            heading = section["heading"]
            level = section["level"]
            content_lines = section["content"]
            
            # Update the heading hierarchy at this level
            heading_hierarchy[level] = heading
            # Clear lower levels
            for i in range(level + 1, len(heading_hierarchy)):
                heading_hierarchy[i] = ""
            
            # Create the full heading path
            heading_path = [h for h in heading_hierarchy[:level+1] if h]
            
            # Join the content lines into a single string
            content_text = "\n".join(content_lines)
            
            # If the content is too long, split it into paragraphs
            if len(content_text.split()) > self.max_chunk_size:
                # Split by paragraphs (double newlines)
                paragraphs = re.split(r'\n\s*\n', content_text)
                
                for i, para in enumerate(paragraphs):
                    if not para.strip():
                        continue
                    
                    # If a paragraph is still too long, split it into sentences
                    if len(para.split()) > self.max_chunk_size:
                        sentences = sent_tokenize(para)
                        current_chunk = []
                        current_chunk_words = 0
                        
                        for sentence in sentences:
                            sentence_words = len(sentence.split())
                            
                            if current_chunk_words + sentence_words <= self.max_chunk_size:
                                current_chunk.append(sentence)
                                current_chunk_words += sentence_words
                            else:
                                # Create a chunk with the current sentences
                                if current_chunk:
                                    chunk_text = " ".join(current_chunk)
                                    
                                    # Include the heading if specified
                                    if self.include_headings and heading:
                                        chunk_text = f"{heading}\n\n{chunk_text}"
                                    
                                    # Create metadata with hierarchical information
                                    chunk_metadata = self._create_hierarchical_metadata(
                                        base_metadata, heading_path, i, len(paragraphs)
                                    )
                                    
                                    chunks.append(Document(
                                        page_content=chunk_text,
                                        metadata=chunk_metadata
                                    ))
                                
                                # Start a new chunk with this sentence
                                current_chunk = [sentence]
                                current_chunk_words = sentence_words
                        
                        # Add the last chunk if there's anything left
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            
                            # Include the heading if specified
                            if self.include_headings and heading:
                                chunk_text = f"{heading}\n\n{chunk_text}"
                            
                            # Create metadata with hierarchical information
                            chunk_metadata = self._create_hierarchical_metadata(
                                base_metadata, heading_path, i, len(paragraphs)
                            )
                            
                            chunks.append(Document(
                                page_content=chunk_text,
                                metadata=chunk_metadata
                            ))
                    else:
                        # The paragraph fits within the max chunk size
                        chunk_text = para
                        
                        # Include the heading if specified
                        if self.include_headings and heading:
                            chunk_text = f"{heading}\n\n{chunk_text}"
                        
                        # Create metadata with hierarchical information
                        chunk_metadata = self._create_hierarchical_metadata(
                            base_metadata, heading_path, i, len(paragraphs)
                        )
                        
                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=chunk_metadata
                        ))
            else:
                # The content fits within the max chunk size
                chunk_text = content_text
                
                # Include the heading if specified
                if self.include_headings and heading:
                    chunk_text = f"{heading}\n\n{chunk_text}"
                
                # Create metadata with hierarchical information
                chunk_metadata = self._create_hierarchical_metadata(
                    base_metadata, heading_path, 0, 1
                )
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
        
        return chunks
    
    def _create_hierarchical_metadata(
        self,
        base_metadata: Dict[str, Any],
        heading_path: List[str],
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """
        Create metadata with hierarchical information.
        
        Args:
            base_metadata (Dict[str, Any]): Base metadata from the original document.
            heading_path (List[str]): List of headings in the hierarchy.
            chunk_index (int): Index of this chunk within its section.
            total_chunks (int): Total number of chunks in the section.
            
        Returns:
            Dict[str, Any]: Metadata with hierarchical information.
        """
        # Create a copy of the base metadata
        metadata = base_metadata.copy()
        
        # Add hierarchical information
        # Convert list to string to avoid ChromaDB metadata issues
        metadata["heading_path_str"] = " > ".join(heading_path)
        metadata["heading_level"] = len(heading_path)
        metadata["chunk_index"] = chunk_index
        metadata["total_chunks"] = total_chunks
        
        return metadata
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the chunker.
        
        Returns:
            Dict[str, Any]: A dictionary containing the chunker's configuration.
        """
        config = super().get_config()
        config.update({
            "include_headings": self.include_headings,
            "heading_patterns": self.heading_patterns,
        })
        return config