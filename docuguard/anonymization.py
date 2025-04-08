"""
Anonymization utilities for DocuGuard PII detection system.

This module provides functionality to anonymize detected PII entities
based on their risk scores and user-defined privacy thresholds.
"""
from typing import List, Dict, Any, Tuple
import os
import fitz  # PyMuPDF

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

def redact_pdf_entities(
    pdf_path: str,
    entities: List[Dict[str, Any]],
    privacy_threshold: float = 0.3,
    output_path: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Apply true PDF redaction by removing sensitive text content using PyMuPDF's redaction annotations.
    
    Args:
        pdf_path (str): Path to the original PDF file
        entities (list): List of entities with risk scores
        privacy_threshold (float): Risk score threshold for redaction
        output_path (str, optional): Path to save the redacted PDF.
        
    Returns:
        tuple: (output_pdf_path, redaction_stats)
    """
    if not entities:
        # Nothing to redact, just copy the file if output_path is provided
        if output_path:
            import shutil
            shutil.copy2(pdf_path, output_path)
            return output_path, {"entities_processed": 0, "entities_redacted": 0}
        return pdf_path, {"entities_processed": 0, "entities_redacted": 0}
    
    # Filter entities by risk score threshold
    entities_to_redact = [e for e in entities if e.get('risk_score', 0) >= privacy_threshold]
    
    stats = {
        "entities_processed": len(entities),
        "entities_redacted": len(entities_to_redact),
        "redacted_entities": [],
        "privacy_threshold": privacy_threshold
    }
    
    if not entities_to_redact:
        if output_path:
            import shutil
            shutil.copy2(pdf_path, output_path)
            return output_path, stats
        return pdf_path, stats
    
    if not output_path:
        file_name, file_ext = os.path.splitext(pdf_path)
        output_path = f"{file_name}_redacted{file_ext}"
    
    doc = None
    new_doc = None # Initialize new_doc
    try:
        doc = fitz.open(pdf_path)
        new_doc = fitz.open() # Create a new PDF for output
        redaction_count = 0
        entity_details = {entity.get('text', ''): entity for entity in entities_to_redact if entity.get('text')}
        
        for page_num in range(len(doc)):
            # Copy page from original to new document
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            # Get reference to the page in the new document
            new_page = new_doc[page_num]
            
            page_redaction_count = 0
            
            # Search and add redaction annotations on the new page
            for text_to_find, entity in entity_details.items():
                instances = new_page.search_for(text_to_find)
                
                if instances:
                    for rect in instances:
                        padded_rect = fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
                        padded_rect.normalize()
                        padded_rect.intersect(new_page.rect)
                        if not padded_rect.is_empty:
                            # Add annotation to the page in the new document
                            new_page.add_redact_annot(padded_rect, fill=(0, 0, 0))
                            page_redaction_count += 1
                    
                    # Update stats for this entity (logic remains the same)
                    if text_to_find not in [re.get('text') for re in stats["redacted_entities"]]:
                        stats["redacted_entities"].append({
                            "label": entity.get('label', 'UNKNOWN'),
                            "text": text_to_find,
                            "instances_redacted": len(instances)
                        })
                    else:
                        for re in stats["redacted_entities"]:
                            if re['text'] == text_to_find:
                                re["instances_redacted"] += len(instances)
                                break
            
            # Apply all redactions on the page within the new document
            new_page.apply_redactions()
            redaction_count += page_redaction_count
        
        stats["total_redaction_instances"] = redaction_count
        # Save the new document, not the original one
        new_doc.save(output_path, garbage=3, deflate=True, clean=True)
        return output_path, stats
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return pdf_path, {"error": str(e), "entities_processed": len(entities), "entities_redacted": 0}
    finally:
        # Ensure both documents are closed
        if doc and not doc.is_closed:
            doc.close()
        if new_doc and not new_doc.is_closed: # Add closing for new_doc
            new_doc.close()