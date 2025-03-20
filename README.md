# RAG Chunking Evaluation

A framework for evaluating different chunking strategies for Retrieval-Augmented Generation (RAG) systems.

## Quick Start

1. Set up environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   pip install -r master/requirements.txt
   ```

2. Set up OpenAI API key (required for evaluation):
   ```bash
   # On macOS/Linux
   export OPENAI_API_KEY=your_api_key_here
   
   # On Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Or create a .env file in the project root
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. Run the pipeline:
   ```bash
   python main/process_qasper.py
   python main/main-rag_processor.py
   python main/main-pipeline.py
   ```

## Features

- Multiple chunking strategies (Recursive, Semantic Clustering, Sentence Transformers, Hierarchical)
- Evaluation metrics (Precision, Recall, F1, MRR@k, R@k)
- Answer generation with LLMs
- ChromaDB vector storage

## Requirements

- Python 3.8+
- GPU support (CUDA or MPS)
- OpenAI API key (for answer generation and evaluation)
- Hugging Face account (for dataset access)

## Dataset

The framework uses the [Qasper dataset](https://allenai.org/data/qasper) from AI2, which contains question-answering pairs for scientific papers. The dataset is automatically downloaded using the Hugging Face datasets library.

## Structure

- `main/`: Execution scripts
- `master/src/chunkers/`: Chunking strategies
- `master/src/database/`: Vector database
- `master/src/evaluation/`: Metrics and evaluation
- `master/src/processor/`: Document processing
- `master/src/utils/`: Utilities
- `data/`: Generated data (Qasper dataset and ChromaDB files)

## Customization

Modify chunking parameters in `main-rag_processor.py` by uncommenting different chunker options.

## Results

Evaluation results are saved to `data/evaluation_results/` in JSON format, including:
- Precision, Recall, and F1 scores
- MRR@k and R@k metrics
- Per-question and overall metrics

## Git Setup

When pushing to GitHub, use this `.gitignore`:
```
.venv/
__pycache__/
*.py[cod]
*$py.class
.env
data/chroma_db/
```

## License

MIT License