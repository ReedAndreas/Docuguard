import pandas as pd
import spacy
import re
import ast
from seqeval.metrics import classification_report

# Import necessary functions from existing codebase
try:
    from docuguard.tokenization import get_token_spans
    from docuguard.bio_tagging import convert_spans_to_bio
    from docuguard.entity_processing import resolve_overlapping_spans
except ImportError as e:
    print(f"Error importing necessary functions: {e}")
    print("Please ensure the script is run from the root directory containing the 'docuguard' package.")
    exit()

# Define benchmark labels (from config.py)
BENCHMARK_PII_TYPES = [
    "EMAIL", "NAME_STUDENT", "USERNAME", "PHONE_NUM",
    "STREET_ADDRESS", "URL_PERSONAL", "ID_NUM"
]

# SpaCy Label Mapping
SPACY_TO_BENCHMARK = {
    "PERSON": "NAME_STUDENT",
    "ORG": "NAME_STUDENT", # Often picks up names as ORG
    "GPE": "STREET_ADDRESS", # Geopolitical entity, might include parts of addresses
    "LOC": "STREET_ADDRESS", # Location, might include parts of addresses
    # DATE, MONEY, etc. are not in benchmark PII types, so ignore for now
}

# Regex Patterns (Refine these as needed)
REGEX_PATTERNS = {
    "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    # Basic North American phone numbers + some variations from example
    "PHONE_NUM": r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})|(?:\d{4}\s\d{7})|(?:\(\d{2}\)\s\d{5}-\d{4})',
    # Basic SSN-like, generic 9-digit, other common formats
    "ID_NUM": r'\b(?:\d{3}-\d{2}-\d{4}|\d{9}|\d{2}-\d{7})\b',
    # Basic URL
    "URL_PERSONAL": r'(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|us|info|biz)(?:/[^\s]*)?',
    # Username might be tricky - often context-dependent. Start simple.
    "USERNAME": r'\b[a-zA-Z0-9_.-]{3,20}\b', # Generic username pattern, might over-generate
}

def main():
    # Load SpaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("SpaCy model 'en_core_web_sm' loaded.")
    except OSError:
        print("Error: SpaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    # Load data
    try:
        df = pd.read_csv("pii_dataset.csv")
    except FileNotFoundError:
        print("Error: pii_dataset.csv not found.")
        return

    SAMPLE_SIZE = 100 # Use the same sample size as main.py
    sample_df = df.head(SAMPLE_SIZE)

    all_true_labels = []
    all_pred_labels = []

    for idx, row in sample_df.iterrows():
        print(f"--- Processing Document Index {idx} ---")
        full_text = str(row.get('text', ''))
        tokens_str = str(row.get('tokens', '[]'))
        trailing_ws_str = str(row.get('trailing_whitespace', '[]'))
        labels_str = str(row.get('labels', '[]'))

        if not full_text or not tokens_str or not trailing_ws_str or not labels_str:
             print(f"Skipping document {idx} due to missing data.")
             continue

        try:
            tokens = ast.literal_eval(tokens_str)
            trailing_whitespace = ast.literal_eval(trailing_ws_str)
            ground_truth_labels = ast.literal_eval(labels_str)

            if not isinstance(tokens, list) or not isinstance(trailing_whitespace, list) or not isinstance(ground_truth_labels, list):
                raise ValueError("Parsed data is not of list type")
            if len(tokens) != len(trailing_whitespace) or len(tokens) != len(ground_truth_labels):
                 print(f"Warning: Length mismatch in document {idx}. Tokens: {len(tokens)}, WS: {len(trailing_whitespace)}, Labels: {len(ground_truth_labels)}. Skipping evaluation for this doc.")
                 pass # Continue processing, but skip adding to eval lists later

        except (ValueError, SyntaxError) as e:
            print(f"Error parsing data for document {idx}: {e}. Skipping.")
            continue

        # --- Baseline Entity Extraction ---
        baseline_entities = []

        # 1. SpaCy NER
        doc = nlp(full_text)
        for ent in doc.ents:
            spacy_label = ent.label_
            benchmark_label = SPACY_TO_BENCHMARK.get(spacy_label)
            if benchmark_label: # Only add if it maps to a relevant benchmark type
                baseline_entities.append({
                    "label": benchmark_label,
                    "text": ent.text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "source": "spacy" # Add source for potential debugging/priority later
                })

        # 2. Regex Matching
        for label, pattern in REGEX_PATTERNS.items():
             if label not in BENCHMARK_PII_TYPES: continue # Ensure regex label is in scope

             try:
                for match in re.finditer(pattern, full_text):
                     # Basic check: Avoid adding generic usernames that overlap with SpaCy entities
                     is_potential_conflict = False
                     if label == "USERNAME":
                          for spacy_ent in baseline_entities:
                               if spacy_ent["source"] == "spacy" and max(match.start(), spacy_ent["start_char"]) < min(match.end(), spacy_ent["end_char"]):
                                    is_potential_conflict = True
                                    break
                     if is_potential_conflict:
                          continue # Skip adding regex username if it overlaps

                     baseline_entities.append({
                         "label": label,
                         "text": match.group(0),
                         "start_char": match.start(),
                         "end_char": match.end(),
                         "source": "regex"
                     })
             except re.error as e:
                  print(f"Regex error for label {label}: {e}")

        # --- Resolve Overlaps (using existing function) ---
        print(f"Found {len(baseline_entities)} candidate entities (SpaCy + Regex).")
        resolved_baseline_entities = resolve_overlapping_spans(baseline_entities)
        print(f"Resolved to {len(resolved_baseline_entities)} non-overlapping entities.")

        # --- Convert to BIO ---
        try:
            token_spans = get_token_spans(tokens, trailing_whitespace)
            predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, resolved_baseline_entities)
        except Exception as e:
            print(f"Error converting spans to BIO for document {idx}: {e}")
            predicted_bio_labels = ['O'] * len(tokens) # Fallback

        # --- Store for Evaluation ---
        if len(predicted_bio_labels) == len(ground_truth_labels):
            all_pred_labels.append(predicted_bio_labels)
            all_true_labels.append(ground_truth_labels)
        else:
             print(f"Skipping evaluation for document {idx} due to label length mismatch (Pred: {len(predicted_bio_labels)}, True: {len(ground_truth_labels)}) ")

    # --- Final Evaluation ---
    if all_true_labels and all_pred_labels:
        print("\n=== Baseline Seqeval Classification Report ===")
        try:
            # Ensure nested lists are handled correctly by seqeval
            report = classification_report(all_true_labels, all_pred_labels, digits=3)
            print(report)
        except Exception as e:
            print(f"Could not generate seqeval report: {e}")
    else:
        print("\nNo valid samples found for seqeval report.")

if __name__ == "__main__":
    main() 