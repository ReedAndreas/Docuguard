"""
Simple text processing utilities for handling raw text input in DocuGuard.
"""
import re
import uuid
import nltk
from typing import List, Dict, Tuple, Any

# Import PyMuPDF for PDF processing
import fitz

# Download required NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize_text(text: str) -> Tuple[List[str], List[bool]]:
    """
    Tokenizes raw text into tokens and trailing whitespace indicators.
    
    Args:
        text (str): Raw text to tokenize
        
    Returns:
        tuple: (tokens, trailing_whitespace) lists
    """
    # Simple tokenization using NLTK
    tokens = []
    trailing_whitespace = []
    
    # First split into sentences, then tokenize words within sentences
    sentences = nltk.sent_tokenize(text)
    
    for sentence in sentences:
        # Tokenize words in the sentence
        sentence_tokens = nltk.word_tokenize(sentence)
        
        # Add tokens to our list
        for i, token in enumerate(sentence_tokens):
            tokens.append(token)
            
            # Check if token is followed by whitespace in original text
            # This is a simplification - more robust implementation might track positions
            if i < len(sentence_tokens) - 1:
                # If not the last token in sentence, assume it's followed by whitespace
                trailing_whitespace.append(True)
            else:
                # Last token in sentence might not have trailing space
                trailing_whitespace.append(True)
    
    # Verify token reconstruction roughly matches original
    reconstructed = ""
    for i, token in enumerate(tokens):
        reconstructed += token
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            reconstructed += " "
    
    # If reconstruction lost significant content, use more basic approach
    if len(reconstructed) < len(text) * 0.8:
        return basic_tokenize(text)
    
    return tokens, trailing_whitespace

def basic_tokenize(text: str) -> Tuple[List[str], List[bool]]:
    """
    Basic tokenization as a fallback.
    
    Args:
        text (str): Raw text to tokenize
        
    Returns:
        tuple: (tokens, trailing_whitespace) lists
    """
    # Simple regex-based tokenizer
    pattern = r'(\w+|\S)'
    matches = re.finditer(pattern, text)
    
    tokens = []
    trailing_whitespace = []
    last_end = 0
    
    for match in matches:
        token = match.group(0)
        start, end = match.span()
        
        tokens.append(token)
        
        # Check if there's whitespace after this token
        next_pos = end
        if next_pos < len(text) and text[next_pos].isspace():
            trailing_whitespace.append(True)
        else:
            trailing_whitespace.append(False)
        
        last_end = end
    
    return tokens, trailing_whitespace

def prepare_document_from_text(text: str, doc_id: str = None) -> Dict[str, Any]:
    """
    Prepares a document dictionary from raw text.
    
    Args:
        text (str): Raw text content
        doc_id (str, optional): Document ID. Generated if not provided.
        
    Returns:
        dict: Document dictionary with fields needed for processing
    """
    # Generate a document ID if not provided
    if not doc_id:
        doc_id = str(uuid.uuid4())
    
    # Tokenize the text
    tokens, trailing_whitespace = tokenize_text(text)
    
    # Build the document dictionary
    document = {
        'document': doc_id,
        'text': text,
        'tokens': tokens,
        'trailing_whitespace': trailing_whitespace,
        # No labels for real-world/unlabeled data
        'labels': ['O'] * len(tokens)
    }
    
    return document

def process_text_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a text file and prepares it for processing.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        dict: Document dictionary with fields needed for processing
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use the filename as document ID
        import os
        doc_id = os.path.basename(file_path)
        
        return prepare_document_from_text(text, doc_id)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
def process_pdf_file(file_path: str) -> Dict[str, Any]:
        """
        Extracts text from a PDF file and prepares it for processing.
    
        Args:
            file_path (str): Path to the PDF file
    
        Returns:
            dict: Document dictionary with fields needed for processing
        """
        try:
            # Open the PDF
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
                text += "\n"
            doc.close()
    
            # Use the filename as document ID
            import os
            doc_id = os.path.basename(file_path)
    
            return prepare_document_from_text(text, doc_id)
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return None