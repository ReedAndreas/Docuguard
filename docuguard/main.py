"""
Main entry point for the DocuGuard PII detection system.
"""
import pandas as pd
from seqeval.metrics import classification_report
from docuguard.document_processor import process_document_with_scoring
from docuguard.config import USE_BENCHMARK_LABELS

def main():
    """
    Main function for running the DocuGuard PII detection system.
    
    Processes documents from the dataset, detects PII entities,
    calculates risk scores, and evaluates performance.
    """
    # Load the CSV file
    try:
        df = pd.read_csv("pii_dataset.csv")
    except FileNotFoundError:
        print("Error: pii_dataset.csv not found. Please place it in the current directory.")
        return

    # Select a sample size for testing
    SAMPLE_SIZE = 100  # Adjust as needed
    sample_df = df.head(SAMPLE_SIZE)
    
    # Toggle between benchmark and real-world labels
    # For evaluation against benchmark dataset: True
    # For real-world detection with real-world labels: False
    use_benchmark_labels = USE_BENCHMARK_LABELS
    
    print(f"Using {'benchmark' if use_benchmark_labels else 'real-world'} labels for PII detection")

    all_true_labels = []
    all_pred_labels = []
    all_scored_entities = []

    for idx, row in sample_df.iterrows():
        print(f"\n--- Processing Document Index {idx} ---")
        full_text = row['text']
        # Handle potential NaN or non-string data robustly
        tokens_str = str(row.get('tokens', '[]'))
        trailing_ws_str = str(row.get('trailing_whitespace', '[]'))
        labels_str = str(row.get('labels', '[]'))

        # Use the processing function with risk scoring
        scored_entities, pred_labels, true_labels = process_document_with_scoring(
            full_text, tokens_str, trailing_ws_str, labels_str, use_benchmark_labels=use_benchmark_labels
        )

        print("\n--- Results ---")
        print(f"Detected {len(scored_entities)} entities with scores:")
        for entity in scored_entities:
            # Truncate long text for cleaner printing
            text_preview = entity['text'][:30] + "..." if len(entity['text']) > 30 else entity['text']
            print(f"  - Label: {entity['label']:<15} Score: {entity['risk_score']:.3f} Text: '{text_preview}'")
        
        # Store results for overall evaluation
        all_scored_entities.append(scored_entities)
        if pred_labels is not None: all_pred_labels.append(pred_labels)
        if true_labels is not None: all_true_labels.append(true_labels)

        print("\nPredicted BIO labels:")
        print(pred_labels)
        print("Ground truth BIO labels:")
        print(true_labels)

    # Print overall evaluation report (only if ground truth was available)
    if all_true_labels and all_pred_labels and len(all_true_labels) == len(all_pred_labels):
        # Filter out cases where processing might have failed and returned empty lists incorrectly
        valid_indices = [i for i, (true, pred) in enumerate(zip(all_true_labels, all_pred_labels)) 
                        if len(true) == len(pred) and len(true) > 0]
        if valid_indices:
            filtered_true = [all_true_labels[i] for i in valid_indices]
            filtered_pred = [all_pred_labels[i] for i in valid_indices]
            print(f"\n=== Seqeval Classification Report on {len(filtered_true)} valid samples ===")
            try:
                print(classification_report(filtered_true, filtered_pred, digits=3))
            except Exception as eval_e:
                print(f"Could not generate seqeval report: {eval_e}")
        else:
            print("\nNo valid samples found for seqeval report.")
    else:
        print("\nCould not generate seqeval report (missing ground truth or prediction mismatch).")

if __name__ == "__main__":
    main()
