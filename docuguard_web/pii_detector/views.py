from django.shortcuts import render
import sys
import os

# Add parent directory to path to import docuguard
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import docuguard functionality
from docuguard.real_world import process_text_input
from docuguard.risk_summary import summarize_risk_profile, format_risk_summary, get_risk_category
from docuguard.anonymization import anonymize_text, REDACTION_MODE, PSEUDONYMIZATION_MODE

from .forms import TextInputForm

def home_view(request):
    """Home view with form for text input and results display."""
    results = None
    
    if request.method == 'POST':
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            should_anonymize = form.cleaned_data.get('anonymize', False)
            anonymization_mode = form.cleaned_data.get('anonymization_mode', REDACTION_MODE)
            
            # Set a default threshold if not provided
            threshold = form.cleaned_data.get('threshold')
            if threshold is None:
                threshold = 0.3
                
            # Process the text using docuguard
            document, scored_entities, pred_labels = process_text_input(text)
            
            # Generate risk summary
            risk_summary_data = summarize_risk_profile(scored_entities)
            
            # Prepare data for template
            risk_summary = {
                'max_score': risk_summary_data['max_score'],
                'max_risk_category': risk_summary_data.get('max_category', 'Low'),
                'total_count': risk_summary_data['total_count'],
                'critical_count': risk_summary_data['risk_counts']['Critical'],
                'high_count': risk_summary_data['risk_counts']['High'],
                'medium_count': risk_summary_data['risk_counts']['Medium'],
                'low_count': risk_summary_data['risk_counts']['Low'],
            }
            
            # Calculate percentages for progress bars
            total_entities = len(scored_entities)
            if total_entities > 0:
                risk_summary['critical_percent'] = (risk_summary['critical_count'] / total_entities) * 100
                risk_summary['high_percent'] = (risk_summary['high_count'] / total_entities) * 100
                risk_summary['medium_percent'] = (risk_summary['medium_count'] / total_entities) * 100
                risk_summary['low_percent'] = (risk_summary['low_count'] / total_entities) * 100
            else:
                risk_summary['critical_percent'] = 0
                risk_summary['high_percent'] = 0
                risk_summary['medium_percent'] = 0
                risk_summary['low_percent'] = 0
            
            # Anonymize text if requested
            anonymized_text = None
            anonymization_stats = None
            if should_anonymize and scored_entities:
                anonymized_text, anonymization_stats = anonymize_text(
                    text, 
                    scored_entities, 
                    privacy_threshold=threshold, 
                    mode=anonymization_mode
                )
            
            results = {
                'entities': scored_entities,
                'risk_summary': risk_summary,
                'anonymized_text': anonymized_text,
                'anonymization_stats': anonymization_stats,
                'original_text': text,
                'anonymization_mode': anonymization_mode,
                'threshold': threshold
            }
    else:
        form = TextInputForm()
    
    return render(request, 'pii_detector/home.html', {
        'form': form,
        'results': results
    })
