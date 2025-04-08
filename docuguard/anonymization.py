"""
Anonymization utilities for DocuGuard PII detection system.

This module provides functionality to anonymize detected PII entities
based on their risk scores and user-defined privacy thresholds.
"""
from typing import List, Dict, Any, Tuple

# Constants for anonymization modes
REDACTION_MODE = "redaction"
PSEUDONYMIZATION_MODE = "pseudonymization"

def anonymize_text(
    text: str, 
    entities: List[Dict[str, Any]], 
    privacy_threshold: float = 0.3,
    mode: str = REDACTION_MODE
) -> Tuple[str, Dict[str, Any]]:
    """
    Anonymize PII entities in text based on their risk scores and a privacy threshold.
    
    Args:
        text (str): Original text containing PII
        entities (list): List of entities with risk scores (from process_document_with_scoring)
        privacy_threshold (float): Risk score threshold for anonymization (0.0 to 1.0)
        mode (str): Anonymization mode - either "redaction" or "pseudonymization"
        
    Returns:
        tuple: (anonymized_text, anonymization_stats)
    """
    if not entities:
        return text, {"entities_processed": 0, "entities_anonymized": 0}
    
    # Sort entities by start position in reverse order to process from end to beginning
    # This prevents position shifts when replacing text
    sorted_entities = sorted(entities, key=lambda e: e.get('start_char', 0), reverse=True)
    
    # Filter entities by risk score threshold
    entities_to_anonymize = [e for e in sorted_entities if e.get('risk_score', 0) >= privacy_threshold]
    
    # Statistics to return
    stats = {
        "entities_processed": len(sorted_entities),
        "entities_anonymized": len(entities_to_anonymize),
        "anonymization_mode": mode,
        "privacy_threshold": privacy_threshold
    }
    
    # For pseudonymization, we need to track entity types to maintain consistency
    entity_counters = {}  # For pseudonymization: {'NAME': 1, 'EMAIL': 1, ...}
    entity_mappings = {}  # For reference: {'john.doe@example.com': 'EMAIL_1', ...}
    
    # Create a mutable copy of the text
    anonymized_text = text
    
    # Process each entity (working from end to beginning to preserve positions)
    for entity in entities_to_anonymize:
        # Extract entity information
        label = entity.get('label', 'UNKNOWN')
        start = entity.get('start_char', 0)
        end = entity.get('end_char', 0)
        original_text = entity.get('text', '')
        
        # Skip if missing required information
        if not all([start >= 0, end > start, original_text]):
            continue
            
        # Determine replacement text based on mode
        if mode == REDACTION_MODE:
            # Simple redaction - replace with [TYPE REDACTED]
            replacement = f"[{label} REDACTED]"
        
        elif mode == PSEUDONYMIZATION_MODE:
            # Pseudonymization - replace with consistent placeholder like NAME_1
            # Check if we've seen this exact text before (maintain consistency)
            if original_text in entity_mappings:
                replacement = entity_mappings[original_text]
            else:
                # Initialize counter for this entity type if not seen before
                if label not in entity_counters:
                    entity_counters[label] = 1
                
                # Create replacement text and increment counter
                replacement = f"{label}_{entity_counters[label]}"
                entity_counters[label] += 1
                
                # Store the mapping for future reference
                entity_mappings[original_text] = replacement
        
        else:
            # Invalid mode, skip this entity
            continue
            
        # Replace the text
        anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
    
    # Add entity mappings to stats if pseudonymization was used
    if mode == PSEUDONYMIZATION_MODE:
        stats["entity_mappings"] = entity_mappings
    
    return anonymized_text, stats

def redact_entities(text: str, entities: List[Dict[str, Any]], privacy_threshold: float = 0.3) -> Tuple[str, Dict[str, Any]]:
    """
    Redact PII entities in text based on their risk scores.
    
    Args:
        text (str): Original text containing PII
        entities (list): List of entities with risk scores 
        privacy_threshold (float): Risk score threshold for redaction
        
    Returns:
        tuple: (redacted_text, redaction_stats)
    """
    return anonymize_text(text, entities, privacy_threshold, REDACTION_MODE)

def pseudonymize_entities(text: str, entities: List[Dict[str, Any]], privacy_threshold: float = 0.3) -> Tuple[str, Dict[str, Any]]:
    """
    Pseudonymize PII entities in text based on their risk scores.
    
    Args:
        text (str): Original text containing PII  
        entities (list): List of entities with risk scores
        privacy_threshold (float): Risk score threshold for pseudonymization
        
    Returns:
        tuple: (pseudonymized_text, pseudonymization_stats)
    """
    return anonymize_text(text, entities, privacy_threshold, PSEUDONYMIZATION_MODE) 