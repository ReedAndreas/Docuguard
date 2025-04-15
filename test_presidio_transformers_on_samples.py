import pandas as pd
import ast
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine # Keep if anonymization is added later
from seqeval.metrics import classification_report
import time

# Attempt to import helper functions from docuguard
try:
    from docuguard.tokenization import get_token_spans
    from docuguard.bio_tagging import convert_spans_to_bio
except ImportError:
    print("Error: Could not import helper functions from docuguard.")
    print("Ensure 'docuguard' package is accessible and installed correctly.")
    exit()

# --- Configuration ---
NUM_SAMPLES = len(df) # Adjust number of samples to test
TRANSFORMER_MODEL = "dslim/bert-base-NER"
SPACY_MODEL_FOR_TOK = "en_core_web_sm"

# Mapping from Presidio entity types to dataset BIO labels (adjust as needed)
# This might need adjustment based on the specific transformer model's output labels
PRESIDIO_TO_BENCHMARK = {
    "PERSON": "NAME_STUDENT",
    "EMAIL_ADDRESS": "EMAIL", # Presidio uses EMAIL_ADDRESS
    "PHONE_NUMBER": "PHONE_NUM", # Presidio uses PHONE_NUMBER
    "LOCATION": "STREET_ADDRESS", # Presidio's LOCATION might map here
    "URL": "URL_PERSONAL",
    "US_SSN": "ID_NUM", # Example mapping
    # Add/verify mappings based on dslim/bert-base-NER labels if different
    # Common NER labels: PER (Person), LOC (Location), ORG (Organization), MISC (Miscellaneous)
    # May need to refine mapping based on experimentation
}

# --- Engine Setup ---
print(f"Setting up TransformersNlpEngine with model: {TRANSFORMER_MODEL}...")
try:
    # Define which transformers model to use
    model_config = [{
        "lang_code": "en",
        "model_name": {
            "spacy": SPACY_MODEL_FOR_TOK, # Use specified spaCy model for lemmas, tokens etc.
            "transformers": TRANSFORMER_MODEL
        }
    }]
    nlp_engine = TransformersNlpEngine(models=model_config)
    # Set up the engine with the Transformer NLP engine
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    print("Analyzer engine initialized successfully.")
except Exception as e:
    print(f"Error initializing NLP Engine or Analyzer: {e}")
    print("Make sure the required models are downloaded and accessible.")
    print("Ensure 'torch' or 'tensorflow' is installed.")
    exit()

# --- Main Processing ---
try:
    # Load samples, ensuring text and list columns are read as strings initially
    df = pd.read_csv("pii_dataset.csv", on_bad_lines='skip', dtype=str).fillna('')
    samples = df.head(NUM_SAMPLES)
except FileNotFoundError:
    print("Error: pii_dataset.csv not found.")
    exit()

all_true_labels = []
all_pred_labels = []

print(f"--- Processing {NUM_SAMPLES} Samples for Seqeval using Transformers ---")
start_time = time.time()

for idx, row in samples.iterrows():
    doc_start_time = time.time()
    print(f"--- Processing Document Index {idx} ---")
    full_text = row.get('text', '')
    tokens_str = row.get('tokens', '[]')
    trailing_ws_str = row.get('trailing_whitespace', '[]')
    labels_str = row.get('labels', '[]')

    if not all([full_text, tokens_str, trailing_ws_str, labels_str]):
        print(f"Skipping document {idx} due to missing data.")
        continue

    try:
        # Safely parse string representations of lists
        tokens = ast.literal_eval(tokens_str)
        trailing_whitespace = ast.literal_eval(trailing_ws_str)
        ground_truth_labels = ast.literal_eval(labels_str)

        if not isinstance(tokens, list) or \
           not isinstance(trailing_whitespace, list) or \
           not isinstance(ground_truth_labels, list):
            raise ValueError("Parsed data is not of list type")

        if len(tokens) != len(trailing_whitespace) or len(tokens) != len(ground_truth_labels):
             print(f"Warning: Length mismatch in document {idx}. Tokens: {len(tokens)}, WS: {len(trailing_whitespace)}, Labels: {len(ground_truth_labels)}. Skipping evaluation for this doc.")
             continue

    except (ValueError, SyntaxError) as e:
        print(f"Error parsing data for document {idx}: {e}. Skipping.")
        continue

    # --- Presidio Analysis (Transformers) ---
    try:
        presidio_results = analyzer.analyze(text=full_text, language='en')
    except Exception as analyze_e:
        print(f"Error analyzing document {idx} with Presidio: {analyze_e}")
        continue # Skip this document if analysis fails

    # --- Convert Presidio results to expected format for BIO tagging ---
    presidio_entities_for_bio = []
    for p_res in presidio_results:
        # Note: Transformer models might output different entity types.
        # Check p_res.entity_type and adjust PRESIDIO_TO_BENCHMARK mapping.
        benchmark_label = PRESIDIO_TO_BENCHMARK.get(p_res.entity_type)
        if benchmark_label: # Only include entities that map to our benchmark types
            presidio_entities_for_bio.append({
                "label": benchmark_label,
                "start_char": p_res.start,
                "end_char": p_res.end,
            })

    # --- Generate BIO Labels ---
    try:
        token_spans = get_token_spans(tokens, trailing_whitespace)
        predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, presidio_entities_for_bio)

        # --- Store for Evaluation ---
        if len(predicted_bio_labels) == len(ground_truth_labels):
            all_pred_labels.append(predicted_bio_labels)
            all_true_labels.append(ground_truth_labels)
            doc_end_time = time.time()
            print(f"Document {idx} processed and added ({len(presidio_results)} Presidio ents -> {len(presidio_entities_for_bio)} mapped). Time: {doc_end_time - doc_start_time:.2f}s")
        else:
             print(f"Skipping evaluation for document {idx} due to final label length mismatch (Pred: {len(predicted_bio_labels)}, True: {len(ground_truth_labels)}) ")

    except Exception as e:
        print(f"Error during BIO conversion or storing results for document {idx}: {e}")

end_time = time.time()
print(f"\n--- Finished processing {NUM_SAMPLES} samples in {end_time - start_time:.2f} seconds ---")

# --- Final Evaluation ---
print("\n--- Seqeval Report (Presidio + Transformers) ---")
if all_true_labels and all_pred_labels:
    valid_indices = [i for i, (true, pred) in enumerate(zip(all_true_labels, all_pred_labels))
                     if len(true) == len(pred) and len(true) > 0]

    if valid_indices:
        filtered_true = [all_true_labels[i] for i in valid_indices]
        filtered_pred = [all_pred_labels[i] for i in valid_indices]

        print(f"Evaluating {len(filtered_true)} valid document(s) out of {NUM_SAMPLES} processed.")
        try:
            print(classification_report(filtered_true, filtered_pred, digits=3))
        except Exception as eval_e:
            print(f"Could not generate seqeval report: {eval_e}")
    else:
         print("No valid samples found for seqeval report (check for length mismatches or processing errors).")

elif len(all_true_labels) != len(all_pred_labels):
     print(f"Could not generate report: Mismatch between number of true ({len(all_true_labels)}) and predicted ({len(all_pred_labels)}) label sets.")
else:
    print("No results to evaluate (all samples might have been skipped).") 