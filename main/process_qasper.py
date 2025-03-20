import random
import sys
import os

# Add the master/src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'master', 'src'))
from utils.qasper.processor import extract_answerable_questions, process_single_file


def main():
    """
    Main function to demonstrate the qasper processor utility.
    """
    # Set random seed for reproducibility
    random.seed(42)

    # Create output directory for the dataset
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "qasper")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving processed Qasper dataset to: {output_dir}")
    
    # Example 1: Process a random subset of the dataset
    # Set shuffle=True to randomly sample from the dataset
    extract_answerable_questions(
        output_dir=output_dir,
        num_samples=250,
        shuffle=True
    )  # Process 250 random samples

    # Example 2: Process a single file
    # file_path = "/app/data/dataset/qasper/1909.00694/1909.00694.json"
    # process_single_file(file_path)

    print("Processing complete!")


if __name__ == "__main__":
    main()
