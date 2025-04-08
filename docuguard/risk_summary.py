"""
Risk summarization utilities for DocuGuard PII detection system.

This module provides functionality to create risk summaries and profiles
from detected PII entities and their assigned risk scores.
"""
from typing import Dict, List, Any
import collections

# Define risk categories and their bounds
RISK_CATEGORIES = {
    'Low': (0.0, 0.4),     # [0.0, 0.4)
    'Medium': (0.4, 0.7),  # [0.4, 0.7)
    'High': (0.7, 0.9),    # [0.7, 0.9)
    'Critical': (0.9, 1.01) # [0.9, 1.0] (using 1.01 to include 1.0 in the range)
}

# Color codes for risk levels (for terminal output)
RISK_COLORS = {
    'Low': '\033[92m',      # Green
    'Medium': '\033[93m',   # Yellow
    'High': '\033[91m',     # Light Red
    'Critical': '\033[31m', # Dark Red
    'reset': '\033[0m'      # Reset
}

def get_risk_category(score: float) -> str:
    """
    Determine the risk category for a given risk score.
    
    Args:
        score (float): Risk score (0.0 to 1.0)
        
    Returns:
        str: Risk category ('Low', 'Medium', 'High', or 'Critical')
    """
    for category, (lower, upper) in RISK_CATEGORIES.items():
        if lower <= score < upper:
            return category
    return 'Low'  # Default to Low if score is outside expected range

def summarize_risk_profile(entities_with_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of the risk profile for a set of PII entities.
    
    Args:
        entities_with_scores (list): List of entities with risk scores
        
    Returns:
        dict: Risk profile summary with max score, counts by category, and total count
    """
    if not entities_with_scores:
        return {
            'max_score': 0.0,
            'risk_counts': {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0},
            'total_count': 0
        }

    max_score = 0.0
    risk_counts = collections.Counter({'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0})
    
    for entity in entities_with_scores:
        score = entity.get('risk_score', 0.0)
        max_score = max(max_score, score)
        
        category = get_risk_category(score)
        risk_counts[category] += 1
            
    return {
        'max_score': max_score,
        'risk_counts': dict(risk_counts),
        'total_count': len(entities_with_scores),
        'max_category': get_risk_category(max_score)
    }

def format_risk_summary(risk_summary: Dict[str, Any], use_color: bool = True) -> str:
    """
    Format a risk summary into a readable string representation.
    
    Args:
        risk_summary (dict): Risk profile summary from summarize_risk_profile
        use_color (bool): Whether to use color codes in the output
        
    Returns:
        str: Formatted risk summary
    """
    max_score = risk_summary['max_score']
    max_category = risk_summary['max_category']
    total_count = risk_summary['total_count']
    risk_counts = risk_summary['risk_counts']
    
    # Determine the maximum count for scaling the bars
    max_count = max(risk_counts.values()) if risk_counts.values() else 1
    bar_width = 20  # Maximum width of bars in characters
    
    # Start building the summary
    lines = []
    lines.append("Document Risk Summary")
    lines.append("=" * 50)
    
    # Format the maximum risk score with color if requested
    max_score_line = f"Maximum Risk Score: {max_score:.3f}"
    if use_color:
        color = RISK_COLORS.get(max_category, RISK_COLORS['reset'])
        reset = RISK_COLORS['reset']
        max_score_line = f"Maximum Risk Score: {color}{max_score:.3f} ({max_category}){reset}"
    else:
        max_score_line = f"Maximum Risk Score: {max_score:.3f} ({max_category})"
    
    lines.append(max_score_line)
    lines.append(f"Total PII Items: {total_count}")
    lines.append("")
    lines.append("Risk Distribution:")
    
    # Add bars for each risk category
    for category in ['Critical', 'High', 'Medium', 'Low']:
        count = risk_counts[category]
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = '█' * bar_length
        
        # Format with color if requested
        if use_color:
            color = RISK_COLORS.get(category, RISK_COLORS['reset'])
            reset = RISK_COLORS['reset']
            lines.append(f"  {category:8}: {count:2} |{color}{bar}{reset}")
        else:
            lines.append(f"  {category:8}: {count:2} |{bar}")
    
    return "\n".join(lines)

def get_color_for_score(score: float) -> str:
    """
    Get the color code for a specific risk score.
    
    Args:
        score (float): Risk score (0.0 to 1.0)
        
    Returns:
        str: ANSI color code
    """
    category = get_risk_category(score)
    return RISK_COLORS.get(category, RISK_COLORS['reset']) 