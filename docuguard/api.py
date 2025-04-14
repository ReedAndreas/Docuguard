"""
API interaction with OpenRouter for PII detection.
"""
import requests
import json
from docuguard.config import OPENROUTER_MODEL_NAME, OPENROUTER_API_KEY

def call_openrouter_llm(full_text, pii_types):
    """
    Calls OpenRouter API to identify PII text and labels.
    
    Args:
        full_text (str): Text to analyze for PII
        pii_types (list): List of PII type labels to detect
        
    Returns:
        list: List of detected entities with text and label
    """
    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")

    prompt = f"""You are an expert system specialized in detecting Personally Identifiable Information (PII) within text. Your task is to carefully read the provided text and identify all instances of the following specific PII types:
{', '.join(pii_types)}

Your goal is to extract these entities accurately.

Return your findings STRICTLY as a JSON list. Each item in the list should be a JSON object representing one detected PII entity, containing ONLY the following keys:
- "text": The exact PII text span found in the original input text.
- "label": The corresponding PII type label from the list above (e.g., "NAME_STUDENT", "EMAIL").
Be careful to compare specifically to the PII and not label non-PII entities as PII. For example be sure dates are not labeled as PII unless it is a date of birth etc. It is super important you identify all key PII entities, especially names really look for these (name student), without missing any but without labeling non-PII entities as PII. Focus deeply on this.

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
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "DocuGuard PII Detection",
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
        print(f"LLM Raw Content: {content}")  # Log raw content for debugging
        return []
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return []
