from django.shortcuts import render
import sys
import os

# Add parent directory to path to import docuguard
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import docuguard functionality
from docuguard.real_world import process_text_input, process_single_pdf_file
from docuguard.risk_summary import summarize_risk_profile, format_risk_summary, get_risk_category
from docuguard.anonymization import anonymize_text, REDACTION_MODE, PSEUDONYMIZATION_MODE

from .forms import TextInputForm

def home_view(request):
    """Home view with form for text or PDF input and results display."""
    import tempfile
    import os

    results = None
    
    if request.method == 'POST':
        form = TextInputForm(request.POST, request.FILES)
        if form.is_valid():
            text = form.cleaned_data.get('text')
            pdf_file = form.cleaned_data.get('pdf_file')
            should_anonymize = form.cleaned_data.get('anonymize', False)
            anonymization_mode = form.cleaned_data.get('anonymization_mode', REDACTION_MODE)
            input_mode = form.cleaned_data.get('input_mode', 'text')

            # Force redaction mode if PDF input
            if input_mode == 'pdf':
                anonymization_mode = REDACTION_MODE
            
            # Set a default threshold if not provided
            threshold = form.cleaned_data.get('threshold')
            if threshold is None:
                threshold = 0.3

            document = None
            scored_entities = []
            pred_labels = []

            if pdf_file:
                import shutil
                media_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'media')
                os.makedirs(media_dir, exist_ok=True)

                # Save uploaded PDF to a temp file
                import uuid
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in pdf_file.chunks():
                        tmp.write(chunk)
                    temp_pdf_path = tmp.name

                redacted_pdf_path = None

                try:
                    # Process the PDF file
                    document, scored_entities, pred_labels = process_single_pdf_file(temp_pdf_path)
                    # Use extracted text
                    text = document.get('text', '')

                    # If anonymization requested, generate redacted PDF
                    if should_anonymize and scored_entities:
                        from docuguard.anonymization import redact_pdf_entities
                        unique_name = f"redacted_{uuid.uuid4().hex}.pdf"
                        output_pdf_path = os.path.join(media_dir, unique_name)
                        print(f"[DEBUG] Intended redacted PDF path: {output_pdf_path}")
                        redacted_pdf_path, pdf_redaction_stats = redact_pdf_entities(
                            temp_pdf_path,
                            scored_entities,
                            privacy_threshold=threshold,
                            output_path=output_pdf_path
                        )
                        print(f"[DEBUG] redact_pdf_entities returned path: {redacted_pdf_path}")
                        if os.path.exists(redacted_pdf_path):
                            print(f"[DEBUG] Redacted PDF successfully saved at: {redacted_pdf_path}")
                        else:
                            print(f"[DEBUG] Redacted PDF NOT FOUND at: {redacted_pdf_path}")
                finally:
                    os.remove(temp_pdf_path)
            else:
                # Process the text input
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
                # For PDF input, do not generate anonymized text, only redacted PDF
                if input_mode == 'pdf':
                    anonymized_text = None
                    anonymization_stats = None
                else:
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
                'threshold': threshold,
                'redacted_pdf_url': None
            }

            # If a redacted PDF was generated, add its URL
            if pdf_file and should_anonymize and 'redacted_pdf_path' in locals() and redacted_pdf_path:
                media_root_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))
                redacted_pdf_abs = os.path.abspath(redacted_pdf_path)
                if redacted_pdf_abs.startswith(media_root_abs):
                    rel_path = redacted_pdf_abs[len(media_root_abs):].lstrip('/').lstrip('\\')
                    results['redacted_pdf_url'] = f"/media/{rel_path}"
                else:
                    results['redacted_pdf_url'] = None
    else:
        form = TextInputForm()
    
    return render(request, 'pii_detector/home.html', {
        'form': form,
        'results': results
    })
