from django import forms

class TextInputForm(forms.Form):
    """Form for submitting text or PDF to be analyzed for PII."""

    INPUT_MODE_CHOICES = [
        ('text', 'Text Input'),
        ('pdf', 'PDF Upload'),
    ]

    input_mode = forms.ChoiceField(
        choices=INPUT_MODE_CHOICES,
        initial='text',
        label="Input Type",
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        required=True
    )

    text = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 10, 'class': 'form-control'}),
        label="Enter text to analyze for PII",
        required=False
    )
    pdf_file = forms.FileField(
        required=False,
        label="Upload a PDF document",
        widget=forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': 'application/pdf'})
    )
    anonymize = forms.BooleanField(
        required=False,
        initial=False,
        label="Anonymize detected PII",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'id_anonymize'})
    )
    
    ANONYMIZATION_CHOICES = [
        ('redaction', 'Redaction (replace with [TYPE])'),
        ('pseudonymization', 'Pseudonymization (replace with fake data)'),
    ]
    
    anonymization_mode = forms.ChoiceField(
        choices=ANONYMIZATION_CHOICES,
        initial='redaction',
        label="Anonymization Method",
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_anonymization_mode'}),
        required=False
    )
    
    threshold = forms.FloatField(
        min_value=0.0,
        max_value=1.0,
        initial=0.3,
        label="Risk Score Threshold",
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'id': 'id_threshold',
            'step': '0.1',
            'min': '0',
            'max': '1'
        }),
        required=False
    )

    def clean(self):
        cleaned_data = super().clean()
        mode = cleaned_data.get('input_mode')
        text = cleaned_data.get('text')
        pdf_file = cleaned_data.get('pdf_file')

        if mode == 'text' and not text:
            raise forms.ValidationError("Please enter text for analysis.")
        if mode == 'pdf' and not pdf_file:
            raise forms.ValidationError("Please upload a PDF file for analysis.")
        return cleaned_data