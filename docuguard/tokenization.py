"""
Token processing utilities for PII detection.
"""

def get_token_spans(tokens, trailing_whitespace):
    """
    Calculates start and end character spans for each token.
    
    Args:
        tokens (list): List of token strings
        trailing_whitespace (list): Boolean list indicating if token has trailing space
        
    Returns:
        list: List of (start, end) tuples representing character spans
    """
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
