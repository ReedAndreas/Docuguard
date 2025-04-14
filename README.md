# DocuGuard - PII Detection System

DocuGuard is a system for detecting Personally Identifiable Information (PII) in text documents, calculating risk scores for each entity, and protecting sensitive data.

## Features

- Detect multiple types of PII (names, emails, phone numbers, addresses, etc.)
- Calculate risk scores for each detected entity
- Summarize document risk with profile distribution (Low, Medium, High, Critical)
- Process both benchmark datasets with ground truth and real-world text
- Support for both CSV datasets and raw text files
- Detailed reporting with context for each detected entity
- Anonymize detected PII using redaction or pseudonymization based on risk scores

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd docuguard
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the necessary API keys for LLM access (see configuration section).

## Usage

### Processing Benchmark Datasets

To process and evaluate benchmark datasets with ground truth:

```bash
python -m docuguard.main
```

This will:
- Load the `pii_dataset.csv` file
- Process a sample of documents
- Calculate risk scores for detected entities
- Evaluate performance against ground truth labels

### Running the DocuGuard Web Interface

The DocuGuard web interface is built with Django and allows for interactive PII detection and management.

1. Ensure all dependencies are installed (see Installation section and requirements.txt).
2. Navigate to the `docuguard_web` directory:
   ```bash
   cd docuguard_web
   ```
3. Apply database migrations (if needed):
   ```bash
   python manage.py migrate
   ```
4. Create a superuser for admin access (optional, for Django admin):
   ```bash
   python manage.py createsuperuser
   ```
5. Run the development server:
   ```bash
   python manage.py runserver
   ```
   This will start the server, typically at `http://127.0.0.1:8000/`. Access it in your web browser to use the interface.

6. To stop the server, use Ctrl+C in the terminal.

### Processing Real-World Text

For real-world text without ground truth labels, use the `real_world.py` module:

#### Process a text file:

```bash
python -m docuguard.real_world --file path/to/your/file.txt
```

#### Process text directly:

```bash
python -m docuguard.real_world --text "Hello, my name is John Smith and my email is john@example.com"
```

#### Process and anonymize text:

```bash
# Redaction mode
python -m docuguard.real_world --file path/to/your/file.txt --anonymize --mode redaction --threshold 0.5

# Pseudonymization mode
python -m docuguard.real_world --text "My SSN is 123-45-6789" --anonymize --mode pseudonymization --threshold 0.3
```

### Running Anonymization Tests

The `test_anonymization.py` script provides a simple way to test anonymization features:

```bash
# Basic redaction test
python test_anonymization.py

# Pseudonymization test
python test_anonymization.py --mode pseudonymization --threshold 0.4

# Test with custom text
python test_anonymization.py --text "My name is John Smith and my email is john@example.com"
```

### Options

- `--benchmark`: Use benchmark labels instead of real-world labels
- `--file`: Path to a text file to process
- `--text`: Direct text input to process
- `--anonymize`: Enable anonymization of detected PII
- `--mode`: Anonymization mode (`redaction` or `pseudonymization`)
- `--threshold`: Risk score threshold for anonymization (0.0-1.0)

## Example Output

```
Document Risk Summary
==================================================
Maximum Risk Score: 0.900 (Critical)
Total PII Items: 4

Risk Distribution:
  Critical : 1 |████
  High    : 1 |████
  Medium  : 1 |████
  Low     : 1 |████

--- Detected 4 PII entities ---

1. SSN | Risk: 0.900 (CRITICAL) | Text: '123-45-6789'
   Context: ...n Francisco, CA 94107. My [123-45-6789] and my credit card num...

2. PHONE_NUMBER | Risk: 0.700 (HIGH) | Text: '(555) 123-4567'
   Context: ...mail.com or call my mobile at [(555) 123-4567].

I currently live at...

3. EMAIL | Risk: 0.500 (MEDIUM) | Text: 'sarah.johnson@gmail.com'
   Context: ...puter Science. You can reach me at [sarah.johnson@gmail.com] or call my mobile at...

4. NAME | Risk: 0.300 (LOW) | Text: 'Sarah Johnson'
   Context: ...Hello! My name is [Sarah Johnson] and I work as a Softw...

--- Anonymized Text ---
Hello! My name is [NAME REDACTED] and I work as a Software Engineer at TechSolutions Inc.

I graduated from Stanford University in 2015 with a degree in Computer Science. 
You can reach me at [EMAIL REDACTED] or call my mobile at [PHONE_NUMBER REDACTED].

I currently live at [ADDRESS REDACTED].
My [SSN REDACTED] and my credit card number is 4111-1111-1111-1111.
```

## Configuration

### Environment Variables (.env File)

To manage sensitive configuration like API keys, create a `.env` file in the root of the project. This file is used to store environment-specific settings that are loaded automatically when running the script.

1. In the project root, create a file named `.env`.
2. Add the following variables to it (replace placeholders with your actual values):

```
USE_BENCHMARK_LABELS=True  # Set to False for real-world labels
OPENROUTER_MODEL_NAME=google/gemini-2.0-flash-001
OPENROUTER_API_KEY=your_api_key_here
```

3. For the `OPENROUTER_API_KEY`, replace `your_api_key_here` with your actual key. If you need access to an OpenRouter API key, it was provided in the project description when uploaded to Brightspace. Please refer to that link for details, and do not share it publicly.

Configuration settings are also defined in `docuguard/config.py`:

- `PII_LABELS_TO_DETECT`: Types of PII to detect
- `USE_BENCHMARK_LABELS`: Toggle between benchmark and real-world labels (overridden by .env if set)
- API settings for LLM integration

## Dataset Format

The benchmark dataset CSV should include these columns:
- `document`: Unique document ID
- `text`: Full document text
- `tokens`: Tokenized text (list as string)
- `trailing_whitespace`: Boolean indicators for trailing whitespace (list as string)
- `labels`: Optional ground truth BIO labels (list as string)
- Additional columns for extracted PII entities

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 