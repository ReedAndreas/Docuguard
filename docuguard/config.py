"""
Configuration settings for the DocuGuard PII detection system.
"""

# OpenRouter API configuration
OPENROUTER_MODEL_NAME = 'google/gemini-2.0-flash-001'
OPENROUTER_API_KEY = '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'

# Real-world PII types for general usage
REAL_WORLD_PII_TYPES = [
    "NAME", "EMAIL", "PHONE_NUMBER", "ADDRESS", "SSN", 
    "CREDIT_CARD", "PASSWORD", "USERNAME", "DATE_OF_BIRTH",
    "BANK_ACCOUNT", "ID_NUMBER", "IP_ADDRESS", "URL"
]

# Benchmark-specific PII types
BENCHMARK_PII_TYPES = [
    "EMAIL", "NAME_STUDENT", "USERNAME", "PHONE_NUM", 
    "STREET_ADDRESS", "URL_PERSONAL", "ID_NUM"
]

# Mapping between real-world and benchmark PII types
PII_TYPE_MAPPING = {
    # Real-world to benchmark
    "NAME": "NAME_STUDENT",
    "EMAIL": "EMAIL",
    "PHONE_NUMBER": "PHONE_NUM",
    "ADDRESS": "STREET_ADDRESS",
    "USERNAME": "USERNAME",
    "ID_NUMBER": "ID_NUM",
    "URL": "URL_PERSONAL",
    
    # Benchmark to real-world
    "NAME_STUDENT": "NAME",
    "EMAIL": "EMAIL",
    "PHONE_NUM": "PHONE_NUMBER",
    "STREET_ADDRESS": "ADDRESS",
    "USERNAME": "USERNAME",
    "ID_NUM": "ID_NUMBER",
    "URL_PERSONAL": "URL"
}

# Set the active PII types based on mode (default to real-world)
USE_BENCHMARK_LABELS = True  # Default to real-world labels
PII_LABELS_TO_DETECT = BENCHMARK_PII_TYPES if USE_BENCHMARK_LABELS else REAL_WORLD_PII_TYPES

def map_pii_type(pii_type, to_benchmark=False):
    """
    Maps between real-world and benchmark PII types.
    
    Args:
        pii_type (str): The PII type to map
        to_benchmark (bool): If True, map from real-world to benchmark;
                            if False, map from benchmark to real-world
                            
    Returns:
        str: The mapped PII type, or the original if no mapping exists
    """
    if to_benchmark:
        return PII_TYPE_MAPPING.get(pii_type, pii_type)
    else:
        return PII_TYPE_MAPPING.get(pii_type, pii_type)

# Risk scoring configuration
BASE_SCORES = {
    # Benchmark PII types
    "ID_NUM": 0.9,
    "PHONE_NUM": 0.7,
    "STREET_ADDRESS": 0.6,
    "EMAIL": 0.5,
    "URL_PERSONAL": 0.4,
    "USERNAME": 0.4,
    "NAME_STUDENT": 0.3,
    
    # Real-world PII types
    "SSN": 0.95,
    "CREDIT_CARD": 0.9, 
    "ID_NUMBER": 0.9,
    "BANK_ACCOUNT": 0.85,
    "PASSWORD": 0.8,
    "PHONE_NUMBER": 0.7,
    "ADDRESS": 0.6,
    "NAME": 0.3,
    "DATE_OF_BIRTH": 0.5,
    "IP_ADDRESS": 0.4,
    
    "DEFAULT": 0.1
}

HIGH_RISK_KEYWORDS = [
    "salary", "password", "account number", "confidential",
    "diagnosis", "private", "secret", "ssn", "pin"
]

KEYWORD_CONTEXT_WINDOW = 30  # Characters before/after PII
KEYWORD_BOOST_FACTOR = 1.5  # Multiplicative boost if keyword found

LINK_BOOST_INCREMENT = 0.15  # Score increase per unique different neighbor PII type
