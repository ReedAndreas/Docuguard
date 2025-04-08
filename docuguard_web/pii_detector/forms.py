from django import forms

class TextInputForm(forms.Form):
    """Form for submitting text to be analyzed for PII."""
    text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'class': 'form-control'}),
        label="Enter text to analyze for PII"
    )
    anonymize = forms.BooleanField(
        required=False,
        initial=False,
        label="Anonymize detected PII",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'anonymize_toggle'})
    )
    
    ANONYMIZATION_CHOICES = [
        ('redaction', 'Redaction (replace with [TYPE])'),
        ('pseudonymization', 'Pseudonymization (replace with fake data)'),
    ]
    
    anonymization_mode = forms.ChoiceField(
        choices=ANONYMIZATION_CHOICES,
        initial='redaction',
        label="Anonymization Method",
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'anonymization_mode'}),
        required=False
    )
    
    threshold = forms.FloatField(
        min_value=0.0,
        max_value=1.0,
        initial=0.3,
        label="Risk Score Threshold",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'threshold_value',
            'step': '0.1',
            'min': '0',
            'max': '1',
            'type': 'hidden'  # Hide the actual input, we'll use the slider
        }),
        required=False
    ) 