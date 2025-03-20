from datasets import load_dataset
import json
import re
import os
from typing import List, Dict, Any


def process_qasper_dataset(
    output_dir: str = "/app/data/dataset/qasper", num_samples: int = 5
) -> List[str]:
    """
    Process the Qasper dataset and save papers as markdown files.

    Args:
        output_dir (str): Directory to save the markdown files
        num_samples (int): Number of samples to process

    Returns:
        list: List of saved file paths
    """
    try:
        # Load dataset
        dataset = load_dataset("allenai/qasper", split="train+validation+test")

        # Create data directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        # Process only the specified number of samples
        # Using a try-except block to handle potential indexing issues
        try:
            # Get a subset of the dataset
            samples = list(dataset)[:num_samples]

            for paper in samples:
                try:
                    # Extract paper details
                    text = f"# {paper['title']}\n\n"
                    text += f"## Abstract\n{paper['abstract']}\n\n"

                    # Process sections
                    for section_name, paragraphs in zip(
                        paper["full_text"]["section_name"],
                        paper["full_text"]["paragraphs"],
                    ):
                        if len(paragraphs) == 0:
                            continue
                        text += f"## {section_name}\n"
                        for paragraph in paragraphs:
                            text += f"{paragraph}\n"
                        text += "\n"

                    # Use paper ID as directory and filename
                    paper_id = paper["id"]
                    
                    # Create a directory for this paper
                    paper_dir = os.path.join(output_dir, paper_id)
                    os.makedirs(paper_dir, exist_ok=True)
                    
                    # Write markdown content to file in paper-specific directory
                    file_path = os.path.join(paper_dir, f"{paper_id}.md")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    # Extract questions from paper['qas'] and save to JSON file
                    if 'qas' in paper and paper['qas']:
                        questions_data: Dict[str, Any] = {
                            "paper_id": paper_id,
                            "questions": []
                        }
                        
                        # Extract all questions
                        if 'question' in paper['qas']:
                            questions_data["questions"] = paper['qas']['question']
                        
                        # Save questions to JSON file
                        questions_file_path = os.path.join(paper_dir, f"{paper_id}.json")
                        with open(questions_file_path, "w", encoding="utf-8") as f:
                            json.dump(questions_data, f, indent=2)
                        
                        print(f"Saved questions to: {questions_file_path}")

                    saved_files.append(file_path)
                    print(f"Saved: {file_path}")

                except Exception as e:
                    print(f"Error processing paper: {e}")
                    continue

        except Exception as e:
            print(f"Error accessing dataset: {e}")

    except Exception as e:
        print(f"Error loading dataset: {e}")

    return saved_files
