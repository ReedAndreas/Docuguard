import requests
import os
import pandas as pd
import re
import json
import ast # To evaluate list/dict strings if loaded from CSV
from seqeval.metrics import classification_report
import spacy # Added for sentence segmentation

# --- Configuration ---
# Choose a model available on OpenRouter (check their website for identifiers)
OPENROUTER_MODEL_NAME = 'google/gemini-2.0-flash-001' 
# flash 2.0 got 86% weighted F1 on 100
PII_LABELS_TO_DETECT = [
    "EMAIL", "NAME_STUDENT", "USERNAME", "PHONE_NUM", 
    "STREET_ADDRESS", "URL_PERSONAL", "ID_NUM" # Add/remove based on dataset
]

# --- Risk Scoring Configuration ---
BASE_SCORES = {
    "ID_NUM": 0.9,
    "PHONE_NUM": 0.7,
    "STREET_ADDRESS": 0.6,
    "EMAIL": 0.5,
    "URL_PERSONAL": 0.4, # Assuming personal URLs are moderately sensitive
    "USERNAME": 0.4,
    "NAME_STUDENT": 0.3,
    "DEFAULT": 0.1 # Default score for unexpected labels
}

HIGH_RISK_KEYWORDS = [
    "salary", "password", "account number", "confidential",
    "diagnosis", "private", "secret", "ssn", "pin"
]
KEYWORD_CONTEXT_WINDOW = 30 # Characters before/after PII
KEYWORD_BOOST_FACTOR = 1.5 # Multiplicative boost if keyword found

LINK_BOOST_INCREMENT = 0.15 # Score increase per unique different neighbor PII type

# Load spaCy model once
try:
    # Using a small, efficient model suitable for sentence boundary detection
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"]) 
    nlp.add_pipe('sentencizer') # Ensure sentence boundary detection is enabled
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None # Set to None if loading fails

# --- Helper Functions ---

def get_token_spans(tokens, trailing_whitespace):
    """Calculates start and end character spans for each token."""
    spans = []
    current_char = 0
    for i, token in enumerate(tokens):
        start = current_char
        end = start + len(token)
        spans.append((start, end))
        current_char = end
        # Add 1 for the space if trailing_whitespace is True
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            current_char += 1
    return spans

def call_openrouter_llm(full_text, pii_types):
    """Calls OpenRouter API to identify PII text and labels."""
    api_key = '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'
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    prompt = f"""You are an expert system specialized in detecting Personally Identifiable Information (PII) within text. Your task is to carefully read the provided text and identify all instances of the following specific PII types:
{', '.join(pii_types)}

Your goal is to extract these entities accurately.

Return your findings STRICTLY as a JSON list. Each item in the list should be a JSON object representing one detected PII entity, containing ONLY the following keys:
- "text": The exact PII text span found in the original input text.
- "label": The corresponding PII type label from the list above (e.g., "NAME_STUDENT", "EMAIL").

If no PII of the specified types is found in the text, return an empty JSON list: [].

Text to analyze:
--- TEXT START ---
{full_text}
--- TEXT END ---

JSON Output:
"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                # Recommended headers by OpenRouter
                "HTTP-Referer": "http://localhost:8000", # Replace with your app URL if deployed
                "X-Title": "DocuGuard PII Detection", # Replace with your app name
            },
            json={
                "model": OPENROUTER_MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        content = response.json()['choices'][0]['message']['content']
        
        # The actual JSON list might be embedded within the content string
        # Find the start and end of the JSON list
        json_start = content.find('[')
        json_end = content.rfind(']')
        if json_start != -1 and json_end != -1:
             json_string = content[json_start:json_end+1]
             llm_output = json.loads(json_string)
             # Basic validation
             if isinstance(llm_output, list):
                 # Further check if items are dicts with 'text' and 'label'
                 return llm_output
             else:
                 print("Warning: LLM output was not a JSON list.")
                 return []
        else:
             print("Warning: Could not find JSON list in LLM output.")
             print(f"LLM Raw Content: {content}")
             return []

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM Raw Content: {content}") # Log raw content for debugging
        return []
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return []


def find_all_occurrences(full_text, llm_identified_entities):
    """Finds all occurrences of LLM-identified text and returns spans."""
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
                    "text": pii_text, # Keep text for potential debugging
                    "start_char": match.start(),
                    "end_char": match.end()
                })
        except re.error as e:
            print(f"Regex error processing '{pii_text}': {e}. Skipping this entity.")
        except Exception as e:
            print(f"Error finding occurrences for '{pii_text}': {e}. Skipping this entity.")
            
    return candidate_spans

def resolve_overlapping_spans(candidate_spans):
    """Resolves overlapping spans, prioritizing longer spans first."""
    # Sort by start_char (asc), then by length (end_char - start_char) (desc)
    candidate_spans.sort(key=lambda x: (x['start_char'], -(x['end_char'] - x['start_char'])))

    resolved_entities = []
    covered_chars = set() # Keep track of character indices already covered

    for candidate in candidate_spans:
        start = candidate['start_char']
        end = candidate['end_char']
        
        # Check if this span is already completely covered by previously added spans
        is_covered = True
        if start == end: # Skip zero-length spans
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
                
    # Optional: Sort back by start_char if needed, though not strictly necessary for BIO tagging
    resolved_entities.sort(key=lambda x: x['start_char'])
    return resolved_entities


def convert_spans_to_bio(tokens, token_char_spans, resolved_entities):
    """Converts resolved entity spans to BIO tags aligned with tokens."""
    predicted_labels = ['O'] * len(tokens)
    
    # Create a map from token index to the entity covering it
    token_entity_assignments = {} # key: token_idx, value: entity dict from resolved_entities
    
    # Assign entities to tokens (handle overlaps based on resolved_entities)
    entity_id_counter = 0 # Assign a unique ID to each resolved entity instance
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
                if i not in token_entity_assignments: # Assign if not already assigned
                     token_entity_assignments[i] = {
                        'label': entity_label,
                        'id': current_entity_id # Store the unique ID
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

def calculate_risk_scores(resolved_entities, full_text):
    """
    Calculates risk scores for resolved PII entities using a simplified graph approach.
    Adds the 'risk_score' key to each entity dictionary in the list.
    """
    if not resolved_entities:
        return []
        
    if nlp is None:
        print("SpaCy model not loaded. Cannot perform sentence segmentation for risk scoring.")
        # Fallback: Assign base scores only
        for i, entity in enumerate(resolved_entities):
             base_score = BASE_SCORES.get(entity['label'], BASE_SCORES["DEFAULT"])
             entity['risk_score'] = min(1.0, max(0.0, base_score)) # Clamp
        return resolved_entities


    # 1. Calculate R_local (Base Score + Optional Keyword Boost)
    r_local = []
    entities_with_context = [] # Store entities along with their context window text

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
        
        r_local.append(min(1.0, max(0.0, boosted_score))) # Clamp R_local
        
    # 2. Segment Sentences & Map Entities
    print("Segmenting sentences...")
    doc = nlp(full_text)
    entity_to_sentence_index = {} # Map entity index to sentence index
    
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
                      entity_to_sentence_index[i] = -i # Assign unique negative index if no sentence found
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
            entity_to_sentence_index[i] = -i # Assign unique negative index on error

    # 3. Build Proximity Graph (Same Sentence)
    print("Building proximity graph...")
    num_entities = len(resolved_entities)
    graph = {i: [] for i in range(num_entities)}
    for i in range(num_entities):
        for j in range(i + 1, num_entities):
            sent_i = entity_to_sentence_index.get(i, -i) # Use unique default if lookup failed
            sent_j = entity_to_sentence_index.get(j, -j)
            if sent_i == sent_j and sent_i >= 0: # Check they are in the same *valid* sentence
                graph[i].append(j)
                graph[j].append(i)

    # 4. Calculate Linkability Boost & Final Scores
    print("Calculating linkability boost and final scores...")
    final_scores = r_local[:] # Start with R_local scores

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
        
    return resolved_entities # Return the list with scores added

# --- NEW Process Document With Scoring Function ---
def process_document_with_scoring(full_text, tokens_str, trailing_whitespace_str, ground_truth_labels_str=None):
    """Processes a single document, performs PII detection, and calculates risk scores."""
    entities_with_scores = []
    predicted_bio_labels = []
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
             print("LLM did not return valid entities.")
             return [], ['O'] * len(tokens), ground_truth_labels # Return empty scores, 'O' labels

        print(f"LLM identified {len(llm_identified_entities)} potential entities.")
        
        # --- Phase 3: Positioning and Disambiguation ---
        print("Finding all occurrences and verifying spans...")
        candidate_spans = find_all_occurrences(full_text, llm_identified_entities)
        print(f"Found {len(candidate_spans)} occurrences in text.")
        
        print("Resolving overlapping spans...")
        resolved_entities = resolve_overlapping_spans(candidate_spans)
        print(f"Resolved to {len(resolved_entities)} non-overlapping entities.")

        # --- *** NEW: Phase 3.5: Risk Scoring *** ---
        print("Calculating risk scores...")
        entities_with_scores = calculate_risk_scores(resolved_entities, full_text)
        print(f"Calculated scores for {len(entities_with_scores)} entities.")

        # --- Phase 4: Conversion to BIO (for evaluation) ---
        print("Converting spans to BIO tags (for evaluation)...")
        # Pass resolved_entities (which now include scores, but BIO conversion doesn't use them)
        predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, resolved_entities) 
        
        print("Processing complete.")
        # Return entities with scores AND the BIO labels for separate use
        return entities_with_scores, predicted_bio_labels, ground_truth_labels

    except SyntaxError as e:
        print(f"Error parsing list strings (maybe invalid format?): {e}")
        num_tokens = 0
        try: num_tokens = len(ast.literal_eval(tokens_str))
        except: pass
        return [], ['O'] * num_tokens if num_tokens else [], None
    except Exception as e:
        print(f"An unexpected error occurred during document processing: {e}")
        num_tokens = 0
        try: num_tokens = len(ast.literal_eval(tokens_str))
        except: pass
        return [], ['O'] * num_tokens if num_tokens else [], None

# --- Main Processing Logic ---

def process_document(full_text, tokens_str, trailing_whitespace_str, ground_truth_labels_str=None):
    """Processes a single document end-to-end."""
    try:
        # Safely evaluate string representations of lists
        tokens = ast.literal_eval(tokens_str)
        trailing_whitespace = ast.literal_eval(trailing_whitespace_str)
        if ground_truth_labels_str:
            ground_truth_labels = ast.literal_eval(ground_truth_labels_str)
        else:
            ground_truth_labels = None # No ground truth available for prediction only
            
        # --- Phase 1: Preparation ---
        print("Calculating token spans...")
        token_spans = get_token_spans(tokens, trailing_whitespace)
        
        # --- Phase 2: LLM Identification ---
        print("Calling LLM for PII identification...")
        llm_identified_entities = call_openrouter_llm(full_text, PII_LABELS_TO_DETECT)
        if not llm_identified_entities:
             print("LLM did not return valid entities.")
             # Return 'O' for all tokens if LLM fails
             return ['O'] * len(tokens), ground_truth_labels 

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

        # --- Phase 4: Conversion to BIO ---
        print("Converting spans to BIO tags...")
        predicted_bio_labels = convert_spans_to_bio(tokens, token_spans, resolved_entities)
        
        print("Processing complete.")
        return predicted_bio_labels, ground_truth_labels

    except SyntaxError as e:
        print(f"Error parsing list strings (maybe invalid format?): {e}")
        # Decide how to handle this, e.g., skip document or return empty predictions
        if tokens_str: # If we know the number of tokens, return 'O's
            try:
                 num_tokens = len(ast.literal_eval(tokens_str))
                 return ['O'] * num_tokens, None
            except: pass # Ignore further errors
        return [], None # Default empty return on error
    except Exception as e:
        print(f"An unexpected error occurred during document processing: {e}")
        return [], None


# --- Example Usage ---

if __name__ == "__main__":
    # Load the CSV file
    try:
        df = pd.read_csv("pii_dataset.csv")
    except FileNotFoundError:
        print("Error: pii_dataset.csv not found. Please place it in the current directory.")
        exit()

    # Select a sample size for testing
    SAMPLE_SIZE = 1
    sample_df = df.head(SAMPLE_SIZE)

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

        # Use the new processing function with risk scoring
        scored_entities, pred_labels, true_labels = process_document_with_scoring(
            full_text, tokens_str, trailing_ws_str, labels_str
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
         valid_indices = [i for i, (true, pred) in enumerate(zip(all_true_labels, all_pred_labels)) if len(true) == len(pred) and len(true) > 0]
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