"""
Main entry point for the DocuGuard PII detection system.
"""
import pandas as pd
import os
import pickle
from seqeval.metrics import classification_report
from docuguard.document_processor import process_document_with_scoring
from docuguard.config import USE_BENCHMARK_LABELS

# Checkpoint configuration
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "main_progress.pkl")
SAVE_INTERVAL = 100  # Save every 100 documents

def save_checkpoint(data, filename):
    """Saves checkpoint data to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"--- Checkpoint saved at index {data['start_index'] - 1} ---")

def load_checkpoint(filename):
    """Loads checkpoint data from a file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def print_intermediate_report(true_labels, pred_labels, current_index, start_index=0):
    """Prints the classification report based on current data."""
    if not true_labels or not pred_labels or len(true_labels) != len(pred_labels):
        print("Skipping intermediate report: Data unavailable or mismatched.")
        return

    valid_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                     if len(true) == len(pred) and len(true) > 0]
    if valid_indices:
        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [pred_labels[i] for i in valid_indices]
        
        total_docs_processed = current_index - start_index + 1
        total_docs_evaluated = len(true_labels)
        docs_skipped = total_docs_processed - total_docs_evaluated
        
        print(f"=== Intermediate Seqeval Report (up to index {current_index}) ===")
        print(f"Documents processed: {total_docs_processed}")
        print(f"Documents evaluated: {total_docs_evaluated}")
        print(f"Documents skipped (due to LLM errors): {docs_skipped}")
        print(f"Valid samples in evaluation: {len(filtered_true)}")
        
        try:
            print(classification_report(filtered_true, filtered_pred, digits=3))
        except Exception as eval_e:
            print(f"Could not generate intermediate seqeval report: {eval_e}")
    else:
        print(f"No valid samples found for intermediate seqeval report (up to index {current_index}).")

def main():
    """
    Main function for running the DocuGuard PII detection system with checkpointing.
    
    Processes documents from the dataset, detects PII entities,
    calculates risk scores, saves progress periodically, and evaluates performance.
    """
    # Load the CSV file
    try:
        df = pd.read_csv("pii_dataset.csv")
    except FileNotFoundError:
        print("Error: pii_dataset.csv not found. Please place it in the current directory.")
        return

    # --- Checkpoint Loading ---
    start_index = 0
    all_true_labels = []
    all_pred_labels = []
    all_scored_entities = []
    
    checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint_data:
        start_index = checkpoint_data.get('start_index', 0)
        all_true_labels = checkpoint_data.get('true_labels', [])
        all_pred_labels = checkpoint_data.get('pred_labels', [])
        all_scored_entities = checkpoint_data.get('scored_entities', [])
        print(f"--- Resuming from index {start_index} ---")
    else:
        print("--- Starting fresh processing ---")

    # Toggle between benchmark and real-world labels
    use_benchmark_labels = USE_BENCHMARK_LABELS
    print(f"Using {'benchmark' if use_benchmark_labels else 'real-world'} labels for PII detection")

    for idx, row in df.iterrows():
        # Skip already processed documents if resuming
        if idx < start_index:
            continue
            
        print(f"--- Processing Document Index {idx} ---")
        full_text = row['text']
        # Handle potential NaN or non-string data robustly
        tokens_str = str(row.get('tokens', '[]'))
        trailing_ws_str = str(row.get('trailing_whitespace', '[]'))
        labels_str = str(row.get('labels', '[]'))

        # Use the processing function with risk scoring
        scored_entities, pred_labels, true_labels = process_document_with_scoring(
            full_text, tokens_str, trailing_ws_str, labels_str, use_benchmark_labels=use_benchmark_labels
        )

        # Store results for overall evaluation only if we have valid predictions
        # If LLM failed to return valid entities, pred_labels will be None
        if pred_labels is not None:
            all_scored_entities.append(scored_entities)
            all_pred_labels.append(pred_labels)
            if true_labels is not None: all_true_labels.append(true_labels)
            print(f"Document {idx} processed and included in evaluation.")
        else:
            print(f"Document {idx} processed but excluded from evaluation due to invalid LLM response.")
        
        # --- Checkpoint Saving and Intermediate Report ---
        # Save if the interval is reached OR it's the last item
        if (idx + 1) % SAVE_INTERVAL == 0 or idx == len(df) - 1:
            checkpoint_data = {
                'start_index': idx + 1, # Next index to start from
                'true_labels': all_true_labels,
                'pred_labels': all_pred_labels,
                'scored_entities': all_scored_entities
            }
            save_checkpoint(checkpoint_data, CHECKPOINT_FILE)
            # Print intermediate report
            print_intermediate_report(all_true_labels, all_pred_labels, idx, start_index)

    # Final results summary (optional, as intermediate reports are printed)
    print("--- Final Processing Summary ---")
    total_docs_processed = len(df) - start_index
    total_docs_evaluated = len(all_scored_entities)
    docs_skipped = total_docs_processed - total_docs_evaluated
    print(f"Total documents processed in this run: {total_docs_processed}")
    print(f"Documents evaluated: {total_docs_evaluated}")
    print(f"Documents skipped (due to LLM errors): {docs_skipped}")
    

    # Print overall evaluation report (only if ground truth was available)
    if all_true_labels and all_pred_labels and len(all_true_labels) == len(all_pred_labels):
        # Filter out cases where processing might have failed and returned empty lists incorrectly
        valid_indices = [i for i, (true, pred) in enumerate(zip(all_true_labels, all_pred_labels))
                         if len(true) == len(pred) and len(true) > 0]
        if valid_indices:
            filtered_true = [all_true_labels[i] for i in valid_indices]
            filtered_pred = [all_pred_labels[i] for i in valid_indices]
            print(f"=== Final Seqeval Classification Report ===")
            print(f"Valid samples in evaluation: {len(filtered_true)}")
            try:
                print(classification_report(filtered_true, filtered_pred, digits=3))
            except Exception as eval_e:
                print(f"Could not generate final seqeval report: {eval_e}")
        else:
            print("No valid samples found for final seqeval report.")
    else:
        print("Could not generate final seqeval report (missing ground truth or prediction mismatch).")

if __name__ == "__main__":
    main()
