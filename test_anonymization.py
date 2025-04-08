#!/usr/bin/env python3
"""
Test script to demonstrate DocuGuard's anonymization capabilities.
"""
import argparse
from docuguard.text_processor import prepare_document_from_text
from docuguard.document_processor import process_document_with_scoring
from docuguard.anonymization import (
    anonymize_text,
    redact_entities,
    pseudonymize_entities,
    REDACTION_MODE,
    PSEUDONYMIZATION_MODE
)
from docuguard.risk_summary import summarize_risk_profile, format_risk_summary

# Example text with PII
SAMPLE_TEXT = """
Hello! My name is Sarah Johnson and I work as a Software Engineer at TechSolutions Inc.

I graduated from Stanford University in 2015 with a degree in Computer Science. 
You can reach me at sarah.johnson@gmail.com or call my mobile at (555) 123-4567.

I currently live at 1234 Oak Street, Apt 567, San Francisco, CA 94107.
My SSN is 123-45-6789 and my credit card number is 4111-1111-1111-1111.
"""

def process_and_anonymize(text, mode, threshold, no_color=False):
    """Process text, detect PII, and anonymize based on risk scores."""
    # Prepare document from text
    document = prepare_document_from_text(text)
    
    # Convert data types to expected string format
    tokens_str = str(document['tokens'])
    trailing_ws_str = str(document['trailing_whitespace'])
    
    # Process with the scoring function
    print("\n--- Detecting PII entities ---")
    scored_entities, pred_labels, _ = process_document_with_scoring(
        document['text'], tokens_str, trailing_ws_str, ground_truth_labels_str=None
    )
    
    # Generate and display risk summary
    risk_summary = summarize_risk_profile(scored_entities)
    print("\n" + format_risk_summary(risk_summary, use_color=not no_color))
    
    # Display detected entities with risk scores
    print(f"\n--- Detected {len(scored_entities)} PII entities ---")
    for i, entity in enumerate(sorted(scored_entities, key=lambda e: e.get('risk_score', 0), reverse=True)):
        label = entity.get('label', 'UNKNOWN')
        score = entity.get('risk_score', 0.0)
        text = entity.get('text', '')
        
        # Determine risk level
        risk_level = "LOW"
        if score >= 0.9:
            risk_level = "CRITICAL"
        elif score >= 0.7:
            risk_level = "HIGH"
        elif score >= 0.4:
            risk_level = "MEDIUM"
            
        print(f"{i+1}. {label} | Risk: {score:.3f} ({risk_level}) | Text: '{text}'")
    
    # Anonymize the text based on mode
    print(f"\n--- Anonymizing with mode: {mode}, threshold: {threshold} ---")
    if mode == REDACTION_MODE:
        anonymized, stats = redact_entities(document['text'], scored_entities, threshold)
    else:
        anonymized, stats = pseudonymize_entities(document['text'], scored_entities, threshold)
    
    # Display statistics
    print(f"Anonymized {stats['entities_anonymized']} of {stats['entities_processed']} entities")
    
    # Display entity mappings for pseudonymization
    if mode == PSEUDONYMIZATION_MODE and 'entity_mappings' in stats:
        print("\nEntity mappings:")
        for original, placeholder in stats['entity_mappings'].items():
            print(f"- '{original}' → '{placeholder}'")
    
    # Display original and anonymized text
    print("\n=== Original Text ===")
    print(document['text'])
    
    print("\n=== Anonymized Text ===")
    print(anonymized)
    
    return document, scored_entities, anonymized, stats

def main():
    parser = argparse.ArgumentParser(description='Test DocuGuard anonymization')
    parser.add_argument('--mode', type=str, choices=[REDACTION_MODE, PSEUDONYMIZATION_MODE], 
                        default=REDACTION_MODE, help='Anonymization mode')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Privacy threshold (0.0-1.0)')
    parser.add_argument('--text', type=str, help='Custom text to process (uses built-in sample if not provided)')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    args = parser.parse_args()
    
    # Use custom text or sample
    text = args.text if args.text else SAMPLE_TEXT
    
    # Process and anonymize
    process_and_anonymize(text, args.mode, args.threshold, args.no_color)
    
if __name__ == "__main__":
    main() 