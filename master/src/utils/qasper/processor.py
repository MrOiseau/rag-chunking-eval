import json
import os
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset


def extract_answerable_questions(
    output_dir: str = "/app/data/dataset/qasper",
    num_samples: Optional[int] = None,
    shuffle: bool = False
) -> List[str]:
    """
    Extract questions where 'unanswerable' is False from the Qasper dataset
    and save their free_form_answer, evidence, and highlighted_evidence.
    Creates markdown files and JSON files for each paper.

    Args:
        output_dir (str): Directory to save the output files
        num_samples (int, optional): Number of samples to process. If None, process all.
        shuffle (bool): Whether to shuffle the dataset before selecting samples.

    Returns:
        list: List of saved file paths
    """
    try:
        # Load dataset
        print("Loading Qasper dataset...")
        dataset = load_dataset("allenai/qasper", split="train+validation+test")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Process samples
        samples = list(dataset)
        
        # Shuffle the samples if requested
        if shuffle:
            print("Shuffling dataset...")
            random.shuffle(samples)
        
        # Select the specified number of papers
        if num_samples is not None:
            samples = samples[:num_samples]
            
        print(f"Processing {len(samples)} papers...")
        
        # Keep track of how many valid questions we've found
        valid_question_count = 0
        
        # Process the selected papers
        for paper in samples:
            try:
                # Convert the dataset item to a dictionary for easier access
                paper_dict = dict(paper)
                paper_id = paper_dict.get("id", "unknown_paper")
                
                # Extract answerable questions first to check if we need to create files
                qas = paper_dict.get('qas', {})
                answerable_questions = []
                
                if qas:
                    questions = qas.get('question', [])
                    answers_list = qas.get('answers', [])
                    
                    for q_idx, question_text in enumerate(questions):
                        # Get answers for this question
                        if q_idx < len(answers_list):
                            question_answers = answers_list[q_idx]
                            
                            # Get the 'answer' field which is a list of answer dictionaries
                            answer_dicts = question_answers.get('answer', [])
                            
                            # Process each answer dictionary
                            for answer_dict in answer_dicts:
                                # Check if answer is answerable and free_form_answer is not empty and not "None"
                                free_form_answer = answer_dict.get('free_form_answer', '')
                                if (not answer_dict.get('unanswerable', True) and
                                    len(free_form_answer) > 0 and
                                    free_form_answer != "None"):
                                    # Check if any highlighted evidence contains "Table"
                                    highlighted_evidence = answer_dict.get('highlighted_evidence', [])
                                    has_table = any("Table" in evidence for evidence in highlighted_evidence)
                                    
                                    # Skip this question if it has "Table" in highlighted evidence
                                    if not has_table:
                                        # Extract required fields
                                        question_data = {
                                            'question': question_text,
                                            'free_form_answer': free_form_answer,
                                            'evidence': answer_dict.get('evidence', []),
                                            'highlighted_evidence': highlighted_evidence
                                        }
                                        
                                        answerable_questions.append(question_data)
                                        valid_question_count += 1
                
                # Only create files if there are answerable questions
                if answerable_questions:
                    # Create a directory for this paper
                    paper_dir = os.path.join(output_dir, paper_id)
                    os.makedirs(paper_dir, exist_ok=True)
                    
                    # Extract paper details for markdown
                    title = paper_dict.get('title', 'Untitled')
                    abstract = paper_dict.get('abstract', '')
                    
                    # Create markdown content
                    markdown_content = f"# {title}\n\n"
                    markdown_content += f"## Abstract\n{abstract}\n\n"
                    
                    # Process sections
                    full_text = paper_dict.get('full_text', {})
                    section_names = full_text.get('section_name', [])
                    paragraphs_list = full_text.get('paragraphs', [])
                    
                    for section_name, paragraphs in zip(section_names, paragraphs_list):
                        if len(paragraphs) == 0:
                            continue
                        markdown_content += f"## {section_name}\n"
                        for paragraph in paragraphs:
                            markdown_content += f"{paragraph}\n"
                        markdown_content += "\n"
                    
                    # Save markdown content
                    md_file_path = os.path.join(paper_dir, f"{paper_id}.md")
                    with open(md_file_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    
                    saved_files.append(md_file_path)
                    print(f"Saved markdown: {md_file_path}")
                    
                    # Save questions data to JSON
                    questions_data = {
                        "paper_id": paper_id,
                        "title": title,
                        "questions": answerable_questions
                    }
                    
                    json_file_path = os.path.join(paper_dir, f"{paper_id}.json")
                    with open(json_file_path, "w", encoding="utf-8") as f:
                        json.dump(questions_data, f, indent=2)
                    
                    saved_files.append(json_file_path)
                    print(f"Saved JSON with {len(answerable_questions)} answerable questions: {json_file_path}")
                else:
                    print(f"No answerable questions found for paper {paper_id}, skipping file creation")
            
            except Exception as e:
                print(f"Error processing paper {paper_id}: {e}")
                continue
        # Print summary
        print(f"Processed {len(samples)} papers and found {valid_question_count} valid questions")
        
        return saved_files
    
    except Exception as e:
        print(f"Error extracting answerable questions: {e}")
        return []
        return []


def process_single_file(
    file_path: str,
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Process a single JSON file from the Qasper dataset.
    
    Args:
        file_path (str): Path to the JSON file
        output_dir (str, optional): Directory to save the output files.
                                   If None, uses the directory of the input file.
    
    Returns:
        list: List of saved file paths
    """
    try:
        # Load the JSON file
        print(f"Loading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            paper = json.load(f)
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Get paper ID
        paper_id = paper.get("paper_id", os.path.basename(file_path).replace('.json', ''))
        
        # Extract answerable questions first to check if we need to create files
        qas = paper.get('qas', {})
        answerable_questions = []
        
        if qas:
            questions = qas.get('question', [])
            answers_list = qas.get('answers', [])
            
            for q_idx, question_text in enumerate(questions):
                # Get answers for this question
                if q_idx < len(answers_list):
                    question_answers = answers_list[q_idx]
                    
                    # Get the 'answer' field which is a list of answer dictionaries
                    answer_dicts = question_answers.get('answer', [])
                    
                    # Process each answer dictionary
                    for answer_dict in answer_dicts:
                        # Check if answer is answerable and free_form_answer is not empty and not "None"
                        free_form_answer = answer_dict.get('free_form_answer', '')
                        if (not answer_dict.get('unanswerable', True) and
                            len(free_form_answer) > 0 and
                            free_form_answer != "None"):
                            # Check if any highlighted evidence contains "Table"
                            highlighted_evidence = answer_dict.get('highlighted_evidence', [])
                            has_table = any("Table" in evidence for evidence in highlighted_evidence)
                            
                            # Skip this question if it has "Table" in highlighted evidence
                            if not has_table:
                                # Extract required fields
                                question_data = {
                                    'question': question_text,
                                    'free_form_answer': free_form_answer,
                                    'evidence': answer_dict.get('evidence', []),
                                    'highlighted_evidence': highlighted_evidence
                                }
                                
                                answerable_questions.append(question_data)
        
        # Only create files if there are answerable questions
        if answerable_questions:
            # Create a directory for this paper
            paper_dir = os.path.join(output_dir, paper_id)
            os.makedirs(paper_dir, exist_ok=True)
            
            # Extract paper details for markdown
            title = paper.get('title', 'Untitled')
            abstract = paper.get('abstract', '')
            
            # Create markdown content
            markdown_content = f"# {title}\n\n"
            markdown_content += f"## Abstract\n{abstract}\n\n"
            
            # Process sections
            full_text = paper.get('full_text', {})
            section_names = full_text.get('section_name', [])
            paragraphs_list = full_text.get('paragraphs', [])
            
            for section_name, paragraphs in zip(section_names, paragraphs_list):
                if len(paragraphs) == 0:
                    continue
                markdown_content += f"## {section_name}\n"
                for paragraph in paragraphs:
                    markdown_content += f"{paragraph}\n"
                markdown_content += "\n"
            
            # Save markdown content
            md_file_path = os.path.join(paper_dir, f"{paper_id}.md")
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            saved_files.append(md_file_path)
            print(f"Saved markdown: {md_file_path}")
            
            # Save questions data to JSON
            questions_data = {
                "paper_id": paper_id,
                "title": title,
                "questions": answerable_questions
            }
            
            json_file_path = os.path.join(paper_dir, f"{paper_id}.json")
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(questions_data, f, indent=2)
            
            saved_files.append(json_file_path)
            print(f"Saved JSON with {len(answerable_questions)} answerable questions: {json_file_path}")
        else:
            print(f"No answerable questions found for paper {paper_id}, skipping file creation")
        
        return saved_files
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []