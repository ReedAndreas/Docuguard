"""
BIO tagging utilities for converting entity spans to token-level BIO tags.
"""

def convert_spans_to_bio(tokens, token_char_spans, resolved_entities):
    """
    Converts resolved entity spans to BIO tags aligned with tokens.
    
    Args:
        tokens (list): List of tokens in the text
        token_char_spans (list): List of (start, end) character spans for each token
        resolved_entities (list): List of resolved entity dictionaries
        
    Returns:
        list: List of BIO tags for each token
    """
    predicted_labels = ['O'] * len(tokens)
    
    # Create a map from token index to the entity covering it
    token_entity_assignments = {}  # key: token_idx, value: entity dict from resolved_entities
    
    # Assign entities to tokens (handle overlaps based on resolved_entities)
    entity_id_counter = 0  # Assign a unique ID to each resolved entity instance
    for entity in resolved_entities:
        entity_start = entity['start_char']
        entity_end = entity['end_char']
        entity_label = entity['label']
        current_entity_id = entity_id_counter
        entity_id_counter += 1

        for i, (tok_start, tok_end) in enumerate(token_char_spans):
            # Check for overlap (token span is within or intersects entity span)
            if max(tok_start, entity_start) < min(tok_end, entity_end):
                # If token is already assigned, potentially overwrite based on rules (e.g. longest span)
                # But since we resolved overlaps earlier, this shouldn't happen if resolution is perfect.
                # We'll assume the resolution step handled conflicts.
                if i not in token_entity_assignments:  # Assign if not already assigned
                     token_entity_assignments[i] = {
                        'label': entity_label,
                        'id': current_entity_id  # Store the unique ID
                    }
                # else: keep the existing assignment (based on the order/resolution)

    # Generate B/I tags from the final assignments
    for i in range(len(tokens)):
        assignment = token_entity_assignments.get(i)
        if assignment:
            entity_label = assignment['label']
            entity_id = assignment['id']
            
            is_beginning = True
            if i > 0:
                prev_assignment = token_entity_assignments.get(i - 1)
                if prev_assignment and prev_assignment['id'] == entity_id:
                    is_beginning = False
            
            if is_beginning:
                predicted_labels[i] = f"B-{entity_label}"
            else:
                predicted_labels[i] = f"I-{entity_label}"
        # Else: label remains 'O'

    return predicted_labels
