"""
Entity processing utilities for PII detection.
"""
import re

def find_all_occurrences(full_text, llm_identified_entities):
    """
    Finds all occurrences of LLM-identified text and returns spans.
    
    Args:
        full_text (str): The original text to search in
        llm_identified_entities (list): List of entities identified by LLM
        
    Returns:
        list: List of found entity spans with label, text, start_char, and end_char
    """
    candidate_spans = []
    for entity in llm_identified_entities:
        pii_text = entity.get('text')
        pii_label = entity.get('label')

        if not pii_text or not pii_label:
            print(f"Warning: Skipping invalid LLM entity: {entity}")
            continue
            
        try:
            # Escape regex special characters for safe searching
            escaped_pii_text = re.escape(pii_text)
            for match in re.finditer(escaped_pii_text, full_text):
                candidate_spans.append({
                    "label": pii_label,
                    "text": pii_text,  # Keep text for potential debugging
                    "start_char": match.start(),
                    "end_char": match.end()
                })
        except re.error as e:
            print(f"Regex error processing '{pii_text}': {e}. Skipping this entity.")
        except Exception as e:
            print(f"Error finding occurrences for '{pii_text}': {e}. Skipping this entity.")
            
    return candidate_spans

def resolve_overlapping_spans(candidate_spans):
    """
    Resolves overlapping spans, prioritizing longer spans first.
    
    Args:
        candidate_spans (list): List of entity spans to resolve
        
    Returns:
        list: Resolved list of non-overlapping entity spans
    """
    # Sort by start_char (asc), then by length (end_char - start_char) (desc)
    candidate_spans.sort(key=lambda x: (x['start_char'], -(x['end_char'] - x['start_char'])))

    resolved_entities = []
    covered_chars = set()  # Keep track of character indices already covered

    for candidate in candidate_spans:
        start = candidate['start_char']
        end = candidate['end_char']
        
        # Check if this span is already completely covered by previously added spans
        is_covered = True
        if start == end:  # Skip zero-length spans
             continue
        for char_index in range(start, end):
            if char_index not in covered_chars:
                is_covered = False
                break
        
        if not is_covered:
            # Add this entity to resolved list
            resolved_entities.append(candidate)
            # Mark its characters as covered
            for char_index in range(start, end):
                covered_chars.add(char_index)
                
    # Optional: Sort back by start_char if needed
    resolved_entities.sort(key=lambda x: x['start_char'])
    return resolved_entities
