"""
DocuGuard Real-World PII Detection

This script processes real-world documents (text or PDF) to detect PII entities and calculate risk scores.
It supports both plain text files (.txt) and PDF files (.pdf), automatically detecting the file type.

PDF support is powered by PyMuPDF (fitz), which extracts text content from PDF pages before processing.

Usage:
    python -m docuguard.real_world --file path/to/sample.txt
    python -m docuguard.real_world --file path/to/sample.pdf
    python -m docuguard.real_world --text "Paste your text here"

Options:
    --file       Path to a .txt or .pdf file to process
    --text       Directly provide text input instead of a file
    --benchmark  Use benchmark-specific PII labels
    --anonymize  Anonymize detected PII in the output

Example:
    python -m docuguard.real_world --file test_samples/sample.pdf --anonymize

The script will extract text, tokenize, detect PII using an LLM, calculate risk scores, and optionally anonymize the output.
"""
import sys
import os
import argparse
import pandas as pd
from docuguard.text_processor import process_text_file, prepare_document_from_text
from docuguard.document_processor import process_document_with_scoring
from docuguard.anonymization import anonymize_text, REDACTION_MODE, PSEUDONYMIZATION_MODE
from docuguard.risk_summary import summarize_risk_profile, format_risk_summary
from docuguard.config import USE_BENCHMARK_LABELS

def process_single_text_file(file_path, use_benchmark_labels=False):
    """
    Process a single text file and detect PII with risk scoring.
    
    Args:
        file_path (str): Path to the text file
        use_benchmark_labels (bool): Whether to use benchmark-specific labels
        
    Returns:
        tuple: (document_data, scored_entities, predicted_bio_labels)
    """
    # Process the text file into document format
    document = process_text_file(file_path)
    if not document:
        print(f"Failed to process file: {file_path}")
        return None, [], []
    
    # Convert data types to expected string format
    tokens_str = str(document['tokens'])
    trailing_ws_str = str(document['trailing_whitespace'])
    
    # Process with the scoring function (we pass None for labels as this is real-world data)
    scored_entities, pred_labels, _ = process_document_with_scoring(
        document['text'], tokens_str, trailing_ws_str, 
        ground_truth_labels_str=None, 
        use_benchmark_labels=use_benchmark_labels
    )
    
    return document, scored_entities, pred_labels

def process_single_pdf_file(file_path, use_benchmark_labels=False):
    """
    Process a single PDF file and detect PII with risk scoring.

    Args:
        file_path (str): Path to the PDF file
        use_benchmark_labels (bool): Whether to use benchmark-specific labels

    Returns:
        tuple: (document_data, scored_entities, predicted_bio_labels)
    """
    from docuguard.text_processor import process_pdf_file

    # Process the PDF file into document format
    document = process_pdf_file(file_path)
    if not document:
        print(f"Failed to process PDF file: {file_path}")
        return None, [], []

    # Convert data types to expected string format
    tokens_str = str(document['tokens'])
    trailing_ws_str = str(document['trailing_whitespace'])

    # Process with the scoring function (we pass None for labels as this is real-world data)
    scored_entities, pred_labels, _ = process_document_with_scoring(
        document['text'], tokens_str, trailing_ws_str,
        ground_truth_labels_str=None,
        use_benchmark_labels=use_benchmark_labels
    )

    return document, scored_entities, pred_labels

def process_text_input(text, use_benchmark_labels=False):
    """
    Process text directly from a string input.
    
    Args:
        text (str): Raw text to process
        use_benchmark_labels (bool): Whether to use benchmark-specific labels
        
    Returns:
        tuple: (document_data, scored_entities, predicted_bio_labels)
    """
    # Prepare document from text
    document = prepare_document_from_text(text)
    
    # Convert data types to expected string format
    tokens_str = str(document['tokens'])
    trailing_ws_str = str(document['trailing_whitespace'])
    
    # Process with the scoring function
    scored_entities, pred_labels, _ = process_document_with_scoring(
        document['text'], tokens_str, trailing_ws_str, 
        ground_truth_labels_str=None, 
        use_benchmark_labels=use_benchmark_labels
    )
    
    return document, scored_entities, pred_labels

def display_results(document, scored_entities):
    """
    Display processing results in a user-friendly way.
    
    Args:
        document (dict): Document data
        scored_entities (list): List of detected entities with risk scores
    """
    if not scored_entities:
        print("\n--- No PII detected ---")
        return
    
    # First display the risk summary
    risk_summary = summarize_risk_profile(scored_entities)
    print("\n" + format_risk_summary(risk_summary))
    
    print(f"\n--- Detected {len(scored_entities)} PII entities ---")
    
    # Sort entities by risk score (highest first)
    sorted_entities = sorted(scored_entities, key=lambda x: x.get('risk_score', 0), reverse=True)
    
    for i, entity in enumerate(sorted_entities):
        # Get entity values
        label = entity.get('label', 'UNKNOWN')
        score = entity.get('risk_score', 0.0)
        text = entity.get('text', '')
        start = entity.get('start_char', 0)
        end = entity.get('end_char', 0)
        
        # Truncate long text for display
        display_text = text[:40] + "..." if len(text) > 40 else text
        
        # Format risk level label based on the score
        risk_level = "LOW"
        if score >= 0.9:
            risk_level = "CRITICAL"
        elif score >= 0.7:
            risk_level = "HIGH"
        elif score >= 0.4:
            risk_level = "MEDIUM"
        
        print(f"{i+1}. {label} | Risk: {score:.3f} ({risk_level}) | Text: '{display_text}'")
        
        # Show context (surrounding text)
        if 'start_char' in entity and 'end_char' in entity:
            full_text = document['text']
            context_start = max(0, start - 20)
            context_end = min(len(full_text), end + 20)
            
            # Create context display with highlighting
            context_before = full_text[context_start:start]
            context_after = full_text[end:context_end]
            highlighted = f"...{context_before}[{text}]{context_after}..."
            
            print(f"   Context: {highlighted}")
        
        print()

def anonymize_and_display(document, scored_entities, privacy_threshold, mode):
    """
    Anonymize and display the processed text.
    
    Args:
        document (dict): Document data
        scored_entities (list): List of detected entities with risk scores
        privacy_threshold (float): Risk score threshold for anonymization
        mode (str): Anonymization mode (redaction or pseudonymization)
    """
    if not scored_entities:
        print("\n--- No PII to anonymize ---")
        return
    
    print(f"\n--- Anonymizing entities with risk score >= {privacy_threshold} ---")
    print(f"--- Mode: {mode} ---")
    
    original_text = document['text']
    anonymized_text, stats = anonymize_text(original_text, scored_entities, privacy_threshold, mode)
    
    print(f"\nAnonymization complete: {stats['entities_anonymized']} of {stats['entities_processed']} entities anonymized")
    
    if mode == PSEUDONYMIZATION_MODE and 'entity_mappings' in stats:
        print("\nEntity mappings:")
        for original, placeholder in stats['entity_mappings'].items():
            print(f"- '{original}' → '{placeholder}'")
    
    print("\n--- Original Text ---")
    print(original_text)
    
    print("\n--- Anonymized Text ---")
    print(anonymized_text)
    
    # Optionally save the anonymized text to a file
    if hasattr(document, 'get') and document.get('document'):
        anonymized_path = f"anonymized_{document['document']}"
        try:
            with open(anonymized_path, 'w') as f:
                f.write(anonymized_text)
            print(f"\nAnonymized text saved to: {anonymized_path}")
        except Exception as e:
            print(f"Error saving anonymized text: {e}")

def main():
    """Main function for running DocuGuard PII detection on real-world data."""
    parser = argparse.ArgumentParser(description='DocuGuard PII Detection for Real-World Text')
    parser.add_argument('--file', type=str, help='Path to text file to process')
    parser.add_argument('--text', type=str, help='Text to process directly')
    parser.add_argument('--benchmark', action='store_true', help='Use benchmark labels')
    parser.add_argument('--anonymize', action='store_true', help='Anonymize detected PII')

    args = parser.parse_args()

    if args.text:
        # Process direct text input
        document, scored_entities, pred_labels = None, [], []
        document_data = {
            'text': args.text,
            'document': 'input_text'
        }
        document_data.update(prepare_document_from_text(args.text))
        tokens_str = str(document_data['tokens'])
        trailing_ws_str = str(document_data['trailing_whitespace'])
        scored_entities, pred_labels, _ = process_document_with_scoring(
            document_data['text'], tokens_str, trailing_ws_str,
            ground_truth_labels_str=None,
            use_benchmark_labels=args.benchmark
        )
        document = document_data
    elif args.file:
        import os
        ext = os.path.splitext(args.file)[1].lower()
        if ext == '.pdf':
            document, scored_entities, pred_labels = process_single_pdf_file(args.file, use_benchmark_labels=args.benchmark)
        else:
            document, scored_entities, pred_labels = process_single_text_file(args.file, use_benchmark_labels=args.benchmark)
    else:
        print("Please provide either --file or --text input.")
        return

    # Display results
    display_results(document, scored_entities)

    # Optionally anonymize
    if args.anonymize:
        privacy_threshold = 0.5  # Default threshold
        mode = 'redact'  # Default mode
        anonymize_and_display(document, scored_entities, privacy_threshold, mode)
    parser.add_argument('--mode', type=str, choices=[REDACTION_MODE, PSEUDONYMIZATION_MODE], 
                        default=REDACTION_MODE, help='Anonymization mode')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Privacy threshold for anonymization (0.0-1.0)')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    args = parser.parse_args()
    
    if not args.file and not args.text:
        parser.print_help()
        return
    
    use_benchmark_labels = args.benchmark if args.benchmark is not None else USE_BENCHMARK_LABELS
    use_color = not args.no_color
    
    # Process file or direct text input
    if args.file:
        import os
        ext = os.path.splitext(args.file)[1].lower()
        print(f"Processing file: {args.file}")
        if ext == '.pdf':
            document, scored_entities, pred_labels = process_single_pdf_file(args.file, use_benchmark_labels)
        else:
            document, scored_entities, pred_labels = process_single_text_file(args.file, use_benchmark_labels)
        if document:
            display_results(document, scored_entities)
            if args.anonymize:
                anonymize_and_display(document, scored_entities, args.threshold, args.mode)
    elif args.text:
        print("Processing text input...")
        document, scored_entities, pred_labels = process_text_input(args.text, use_benchmark_labels)
        display_results(document, scored_entities)
        if args.anonymize:
            anonymize_and_display(document, scored_entities, args.threshold, args.mode)

if __name__ == "__main__":
    main() 