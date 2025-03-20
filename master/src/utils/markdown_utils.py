import os
import re
from typing import List, Dict, Any
from langchain.schema import Document


def read_markdown_files(directory: str) -> List[Document]:
    """
    Read all markdown files from a directory and its subdirectories and convert them to Document objects.
    Handles the structure where each paper is in its own subdirectory.

    Args:
        directory (str): Directory containing paper subdirectories

    Returns:
        List[Document]: List of Document objects with content and metadata
    """
    documents = []

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return documents

    # Get all subdirectories (each representing a paper)
    subdirs = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]

    if not subdirs:
        # Fallback to old behavior if no subdirectories found
        print(
            f"No paper subdirectories found in {directory}. Checking for direct markdown files."
        )
        files = [f for f in os.listdir(directory) if f.endswith(".md")]

        for file in files:
            file_path = os.path.join(directory, file)
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract paper ID from filename
                paper_id = file.replace(".md", "")

                # Create Document object with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "paper_id": paper_id,
                        "file_name": file,
                    },
                )
                documents.append(doc)
                print(f"Processed: {file}")

            except Exception as e:
                print(f"Error reading file {file}: {e}")
    else:
        # Process each paper subdirectory
        for paper_dir in subdirs:
            paper_id = paper_dir  # The directory name is the paper ID
            paper_path = os.path.join(directory, paper_dir)

            # Look for the markdown file in the paper directory
            md_files = [f for f in os.listdir(paper_path) if f.endswith(".md")]

            if not md_files:
                print(f"No markdown files found in {paper_path}")
                continue

            for file in md_files:
                file_path = os.path.join(paper_path, file)
                try:
                    # Read file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create Document object with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "paper_id": paper_id,
                            "file_name": file,
                        },
                    )
                    documents.append(doc)
                    print(f"Processed: {file}")

                except Exception as e:
                    print(f"Error reading file {file}: {e}")

    return documents
