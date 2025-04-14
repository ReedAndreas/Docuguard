"""
Document processing module for PII detection with risk scoring.
"""
import ast
from docuguard.tokenization import get_token_spans
from docuguard.api import call_openrouter_llm
from docuguard.entity_processing import find_all_occurrences, resolve_overlapping_spans
from docuguard.bio_tagging import convert_spans_to_bio
from docuguard.risk_scoring import calculate_risk_scores
from docuguard.config import (
    PII_LABELS_TO_DETECT,
    BENCHMARK_PII_TYPES,
    USE_BENCHMARK_LABELS,
    map_pii_type
)

def get_normalized_entities(entities, use_benchmark_labels):
    """
    Ensures entities have consistent label types based on the selected mode.
    
    Args:
        entities (list): List of entity dictionaries
        use_benchmark_labels (bool): Whether to use benchmark labels
        
    Returns:
        list: List of entities with normalized type labels
    """
    normalized_entities = []
    
    for entity in entities:
        # Create a copy to avoid modifying the original
        normalized_entity = entity.copy()
        
        # Convert type if necessary
        if use_benchmark_labels != USE_BENCHMARK_LABELS:
            if 'label' in entity:
                # Map the label (keeping the original key name)
                normalized_entity['label'] = map_pii_type(
                    entity['label'], 
                    to_benchmark=use_benchmark_labels
                )
            
        normalized_entities.append(normalized_entity)
        
    return normalized_entities

def process_document_with_scoring(full_text, tokens_str, trailing_whitespace_str, ground_truth_labels_str=None, use_benchmark_labels=USE_BENCHMARK_LABELS):
    """
    Processes a single document, performs PII detection, and calculates risk scores.
    
    Args:
        full_text (str): The text to analyze
        tokens_str (str): String representation of token list
        trailing_whitespace_str (str): String representation of trailing whitespace boolean list
        ground_truth_labels_str (str, optional): String representation of ground truth labels list
        use_benchmark_labels (bool): Whether to use benchmark-specific labels
        
    Returns:
        tuple: (entities_with_scores, predicted_bio_labels, ground_truth_labels)
    """
    entities_with_scores = []
    predicted_bio_labels = None
    ground_truth_labels = None
    
    try:
        # Safely evaluate string representations of lists
        tokens = ast.literal_eval(tokens_str)
        trailing_whitespace = ast.literal_eval(trailing_whitespace_str)
        if ground_truth_labels_str:
            ground_truth_labels = ast.literal_eval(ground_truth_labels_str)
            
        # --- Phase 1: Preparation ---
        print("Calculating token spans...")
        token_spans = get_token_spans(tokens, trailing_whitespace)
        
        # --- Phase 2: LLM Identification ---
        print("Calling LLM for PII identification...")
        llm_identified_entities = call_openrouter_llm(full_text, PII_LABELS_TO_DETECT)
        if not llm_identified_entities:
             print("LLM did not return valid entities. Skipping this document in evaluation.")
             # Return empty list for entities and None for predicted labels to exclude from evaluation
             return [], None, ground_truth_labels

        print(f"LLM identified {len(llm_identified_entities)} potential entities.")
        
        # --- Phase 3: Positioning and Disambiguation ---
        print("Finding all occurrences and verifying spans...")
        candidate_spans = find_all_occurrences(full_text, llm_identified_entities)
        print(f"Found {len(candidate_spans)} occurrences in text.")
        
        print("Resolving overlapping spans...")
        resolved_entities = resolve_overlapping_spans(candidate_spans)
        print(f"Resolved to {len(resolved_entities)} non-overlapping entities.")

        # Normalize entity types for risk calculation
        normalized_entities = get_normalized_entities(resolved_entities, use_benchmark_labels)

        # --- Phase 3.5: Risk Scoring ---
        print("Calculating risk scores...")
        entities_with_scores = calculate_risk_scores(normalized_entities, full_text)
        print(f"Calculated scores for {len(entities_with_scores)} entities.")

        # --- Phase 4: Conversion to BIO (for evaluation) ---
        print("Converting spans to BIO tags (for evaluation)...")
        # For BIO conversion, ensure we're using benchmark labels if evaluating against benchmark
        bio_entities = get_normalized_entities(resolved_entities, True) if use_benchmark_labels else normalized_entities
        predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, bio_entities) 
        
        print("Processing complete.")
        # Return entities with scores AND the BIO labels for separate use
        return entities_with_scores, predicted_bio_labels, ground_truth_labels

    except Exception as e:
        print(f"An unexpected error occurred during document processing: {e}")
        import traceback
        traceback.print_exc()
        # Return None for predicted labels to exclude this document from evaluation
        return [], None, None


def process_document(full_text, tokens_str, trailing_whitespace_str, ground_truth_labels_str=None, use_benchmark_labels=USE_BENCHMARK_LABELS):
    """
    Processes a single document end-to-end without risk scoring.
    
    Args:
        full_text (str): The text to analyze
        tokens_str (str): String representation of token list
        trailing_whitespace_str (str): String representation of trailing whitespace boolean list
        ground_truth_labels_str (str, optional): String representation of ground truth labels list
        use_benchmark_labels (bool): Whether to use benchmark-specific labels
        
    Returns:
        tuple: (predicted_bio_labels, ground_truth_labels)
    """
    try:
        # Safely evaluate string representations of lists
        tokens = ast.literal_eval(tokens_str)
        trailing_whitespace = ast.literal_eval(trailing_whitespace_str)
        if ground_truth_labels_str:
            ground_truth_labels = ast.literal_eval(ground_truth_labels_str)
        else:
            ground_truth_labels = None  # No ground truth available for prediction only
            
        # --- Phase 1: Preparation ---
        print("Calculating token spans...")
        token_spans = get_token_spans(tokens, trailing_whitespace)
        
        # --- Phase 2: LLM Identification ---
        print("Calling LLM for PII identification...")
        llm_identified_entities = call_openrouter_llm(full_text, PII_LABELS_TO_DETECT)
        if not llm_identified_entities:
             print("LLM did not return valid entities. Skipping this document in evaluation.")
             # Return None instead of 'O' labels to exclude this document from evaluation
             return None, ground_truth_labels 

        print(f"LLM identified {len(llm_identified_entities)} potential entities.")
        # print raw llm output
        print(llm_identified_entities)
        
        # --- Phase 3: Positioning and Disambiguation ---
        print("Finding all occurrences and verifying spans...")
        candidate_spans = find_all_occurrences(full_text, llm_identified_entities)
        print(f"Found {len(candidate_spans)} occurrences in text.")
        
        print("Resolving overlapping spans...")
        resolved_entities = resolve_overlapping_spans(candidate_spans)
        print(f"Resolved to {len(resolved_entities)} non-overlapping entities.")

        # For BIO conversion, ensure we're using benchmark labels if evaluating against benchmark
        bio_entities = get_normalized_entities(resolved_entities, True) if use_benchmark_labels else resolved_entities

        # --- Phase 4: Conversion to BIO ---
        print("Converting spans to BIO tags...")
        predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, bio_entities)
        
        print("Processing complete.")
        return predicted_bio_labels, ground_truth_labels

    except Exception as e:
        print(f"An unexpected error occurred during document processing: {e}")
        import traceback
        traceback.print_exc()
        # Return None instead of empty or 'O' labels to exclude this document
        return None, None
