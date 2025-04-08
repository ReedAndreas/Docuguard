# DocuGuard: PII Detection System

DocuGuard is a system for detecting and scoring Personally Identifiable Information (PII) in text documents. It uses Large Language Models (LLMs) to identify PII entities, calculates risk scores, and supports evaluation against ground truth data.

## Features

- PII detection using state-of-the-art LLMs (via OpenRouter API)
- Risk scoring for detected PII entities 
- Support for multiple PII types (emails, names, phone numbers, etc.)
- Contextual risk analysis based on surrounding text
- Evaluation against ground truth data with standard NER metrics

## Architecture

The system is organized into the following modules:

- `config.py`: Configuration settings
- `api.py`: API interaction with OpenRouter
- `tokenization.py`: Token processing utilities
- `entity_processing.py`: Entity extraction and resolution
- `bio_tagging.py`: Conversion to BIO tagging format
- `risk_scoring.py`: Risk score calculation
- `document_processor.py`: Main document processing logic
- `main.py`: Entry point for the application

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd docuguard

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas requests spacy seqeval

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

1. Ensure you have a CSV file named `pii_dataset.csv` in the root directory
2. Run the script:

```bash
python run_docuguard.py
```

## Dataset Format

The system expects a CSV file with the following columns:
- `text`: The full text to analyze
- `tokens`: A string representation of a list of tokens
- `trailing_whitespace`: A string representation of a list of booleans indicating if the token has trailing whitespace
- `labels`: (Optional) A string representation of a list of ground truth BIO tags

## Risk Scoring

The risk scoring system considers:
- Base risk by PII type (e.g., ID numbers are higher risk than names)
- Contextual risk (presence of risky keywords near the PII)
- Linkability risk (multiple types of PII in proximity)

## License

[Add license information here]

## Acknowledgments

- OpenRouter for providing API access to large language models 