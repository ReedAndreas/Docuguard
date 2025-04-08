"""
Risk scoring utilities for PII entities.
"""
import spacy
from docuguard.config import BASE_SCORES, HIGH_RISK_KEYWORDS, KEYWORD_CONTEXT_WINDOW
from docuguard.config import KEYWORD_BOOST_FACTOR, LINK_BOOST_INCREMENT

# Load spaCy model once
try:
    # Using a small, efficient model suitable for sentence boundary detection
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"]) 
    nlp.add_pipe('sentencizer')  # Ensure sentence boundary detection is enabled
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None  # Set to None if loading fails

def calculate_risk_scores(resolved_entities, full_text):
    """
    Calculates risk scores for resolved PII entities using a simplified graph approach.
    Adds the 'risk_score' key to each entity dictionary in the list.
    
    Args:
        resolved_entities (list): List of resolved entity dictionaries
        full_text (str): Original text for context analysis
        
    Returns:
        list: The resolved entities with added risk_score values
    """
    if not resolved_entities:
        return []
        
    if nlp is None:
        print("SpaCy model not loaded. Cannot perform sentence segmentation for risk scoring.")
        # Fallback: Assign base scores only
        for i, entity in enumerate(resolved_entities):
             base_score = BASE_SCORES.get(entity['label'], BASE_SCORES["DEFAULT"])
             entity['risk_score'] = min(1.0, max(0.0, base_score))  # Clamp
        return resolved_entities


    # 1. Calculate R_local (Base Score + Optional Keyword Boost)
    r_local = []
    entities_with_context = []  # Store entities along with their context window text

    for i, entity in enumerate(resolved_entities):
        base_score = BASE_SCORES.get(entity['label'], BASE_SCORES["DEFAULT"])
        
        # Optional Keyword Boost
        boosted_score = base_score
        try:
            start = entity['start_char']
            end = entity['end_char']
            window_start = max(0, start - KEYWORD_CONTEXT_WINDOW)
            window_end = min(len(full_text), end + KEYWORD_CONTEXT_WINDOW)
            context_text = full_text[window_start:window_end].lower()
            
            keyword_found = False
            for keyword in HIGH_RISK_KEYWORDS:
                if keyword in context_text:
                    boosted_score = base_score * KEYWORD_BOOST_FACTOR
                    keyword_found = True
                    break 
            entities_with_context.append({"entity_index": i, "context": context_text, "keyword_found": keyword_found})
            
        except Exception as e:
            print(f"Error during keyword context check for entity {i}: {e}")
            # Keep base score if context check fails
        
        r_local.append(min(1.0, max(0.0, boosted_score)))  # Clamp R_local
        
    # 2. Segment Sentences & Map Entities
    print("Segmenting sentences...")
    doc = nlp(full_text)
    entity_to_sentence_index = {}  # Map entity index to sentence index
    
    for i, entity in enumerate(resolved_entities):
        start = entity['start_char']
        end = entity['end_char']
        # Find the sentence containing the middle point of the entity span
        # Using char_span is more robust if available and entities align well
        try:
            # Attempt to get a span object covering the entity
            span = doc.char_span(start, end, label=entity['label'])
            if span is not None and span.sent is not None:
                 # Find the index of the sentence
                 sent_index = -1
                 for idx, sent in enumerate(doc.sents):
                      if sent == span.sent:
                           sent_index = idx
                           break
                 if sent_index != -1:
                     entity_to_sentence_index[i] = sent_index
                 else:
                     # Fallback if sentence object comparison fails? Less likely.
                      entity_to_sentence_index[i] = -i  # Assign unique negative index if no sentence found
            else:
                 # Fallback: find sentence containing the start character
                 char_index = start
                 found_sent = -1
                 for idx, sent in enumerate(doc.sents):
                     if sent.start_char <= char_index < sent.end_char:
                         found_sent = idx
                         break
                 entity_to_sentence_index[i] = found_sent if found_sent != -1 else -i
        except Exception as e:
            print(f"Error mapping entity {i} to sentence: {e}")
            entity_to_sentence_index[i] = -i  # Assign unique negative index on error

    # 3. Build Proximity Graph (Same Sentence)
    print("Building proximity graph...")
    num_entities = len(resolved_entities)
    graph = {i: [] for i in range(num_entities)}
    for i in range(num_entities):
        for j in range(i + 1, num_entities):
            sent_i = entity_to_sentence_index.get(i, -i)  # Use unique default if lookup failed
            sent_j = entity_to_sentence_index.get(j, -j)
            if sent_i == sent_j and sent_i >= 0:  # Check they are in the same *valid* sentence
                graph[i].append(j)
                graph[j].append(i)

    # 4. Calculate Linkability Boost & Final Scores
    print("Calculating linkability boost and final scores...")
    final_scores = r_local[:]  # Start with R_local scores

    for i in range(num_entities):
        entity_i_label = resolved_entities[i]['label']
        neighbor_labels = {resolved_entities[j]['label'] for j in graph[i]}
        num_different_neighbors = len(neighbor_labels - {entity_i_label})
        
        total_link_boost = num_different_neighbors * LINK_BOOST_INCREMENT
        
        # Apply boost (additive)
        final_scores[i] = final_scores[i] + total_link_boost
        
        # Clamp final score
        final_scores[i] = min(1.0, max(0.0, final_scores[i]))

    # Add the final score back to the entity dictionaries
    for i, entity in enumerate(resolved_entities):
        entity['risk_score'] = final_scores[i]
        
    return resolved_entities  # Return the list with scores added
