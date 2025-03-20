"""
Data models for RAG evaluation.
"""
from typing import List


class Question:
    """
    Class representing a question with its answer and evidence.
    Used for evaluating RAG systems against ground truth data.
    """

    def __init__(
        self,
        paper_id: str,
        question: str,
        free_form_answer: str,
        evidence: List[str],
        highlighted_evidence: List[str],
    ):
        """
        Initialize a Question object.
        
        Args:
            paper_id (str): Identifier for the paper this question relates to
            question (str): The question text
            free_form_answer (str): The ground truth answer
            evidence (List[str]): List of evidence passages supporting the answer
            highlighted_evidence (List[str]): List of highlighted evidence passages
        """
        self.paper_id = paper_id
        self.question = question
        self.free_form_answer = free_form_answer
        self.evidence = evidence
        self.highlighted_evidence = highlighted_evidence

    def __str__(self) -> str:
        """String representation of the Question object."""
        return f"Question: {self.question}\nAnswer: {self.free_form_answer}"

    def print_details(self) -> None:
        """Print all details of the Question object."""
        print(f"Paper ID: {self.paper_id}")
        print(f"Question: {self.question}")
        print(f"Answer: {self.free_form_answer}")
        print(f"Evidence count: {len([e for e in self.evidence if e])}")
        print(f"Highlighted evidence count: {len([e for e in self.highlighted_evidence if e])}")