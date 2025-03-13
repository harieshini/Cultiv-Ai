import re
import google.generativeai as genai
from config import GEMINI_API_KEY
from googletrans import Translator

# Configure API Key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize translator
translator = Translator()

# Set up the Gemini generative model with a system instruction tailored for agriculture.
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro-latest',
    system_instruction="""You are AGRI-GPT, an expert AI agricultural assistant with comprehensive knowledge in all aspects of agriculture.
- Answer questions about crop management, livestock care, soil health, irrigation, pest control, sustainable practices, and agricultural economics.
- Format your response clearly with:
  Answer: [concise answer]
  Details:
    - [bullet point 1]
    - [bullet point 2]
    - [bullet point 3]
- Do not include any reference links.
- If the question is not agriculture-related, respond with: "I specialize in Agriculture."
"""
)

def clean_and_localize(answer, lang):
    """
    Remove markdown formatting, URLs, and extra whitespace.
    Also replace key agricultural terms with colloquial Hindi or Tamil equivalents.
    """
    # Remove markdown markers (e.g. **)
    answer = re.sub(r'\*\*', '', answer)
    # Remove extra newlines and trim whitespace
    answer = re.sub(r'\n+', '\n', answer).strip()
    # Remove URLs
    answer = re.sub(r'http\S+', '', answer)
    
    if lang == 'hi':
        replacements = {
            "Agriculture": "खेती",
            "agriculture": "खेती",
            "Crops": "फसलें",
            "crops": "फसलें",
            "Irrigation": "सिंचाई",
            "irrigation": "सिंचाई",
            "Pest": "कीट",
            "pest": "कीट",
            "Management": "प्रबंधन",
            "management": "प्रबंधन",
            "Expert": "विशेषज्ञ",
            "expert": "विशेषज्ञ",
        }
    elif lang == 'ta':
        replacements = {
            "Agriculture": "விவசாயம்",
            "agriculture": "விவசாயம்",
            "Crops": "பயிர்கள்",
            "crops": "பயிர்கள்",
            "Irrigation": "நீர்முறை",
            "irrigation": "நீர்முறை",
            "Pest": "பூச்சி",
            "pest": "பூச்சி",
            "Management": "மேலாண்மை",
            "management": "மேலாண்மை",
            "Expert": "திறமையான",
            "expert": "திறமையான",
        }
    else:
        replacements = {}
    
    for key, value in replacements.items():
        answer = answer.replace(key, value)
    return answer

def is_valid_question(question):
    """
    Uses the Gemini API to determine if a question is agriculture-related.
    Sends a classification prompt and expects a one-word answer: 'yes' or 'no'.
    Returns True if the answer starts with 'yes', otherwise False.
    """
    classification_prompt = (
        "Determine if the following question is about agriculture (covering crops, livestock, soil, irrigation, pest control, or agroeconomics). "
        "Answer with a single word: 'yes' or 'no'.\n"
        f"Question: {question}"
    )
    try:
        response = model.generate_content(classification_prompt)
        answer = response.text.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"Classification error: {e}")
        return False

def generate_text_response(prompt):
    """
    Generates an answer for the given prompt using the Gemini API.
    If the prompt includes "in tamil" or "in hindi", it forces the answer into that language.
    The final answer is cleaned and localized for easier reading.
    """
    prompt_lower = prompt.lower()
    if "in tamil" in prompt_lower:
        original_lang = "ta"
        prompt = re.sub(r"in tamil", "", prompt, flags=re.IGNORECASE).strip()
    elif "in hindi" in prompt_lower:
        original_lang = "hi"
        prompt = re.sub(r"in hindi", "", prompt, flags=re.IGNORECASE).strip()
    else:
        # Detect language from prompt; default to English.
        detected = translator.detect(prompt)
        original_lang = detected.lang if detected else "en"

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error in text generation: {e}"
    
    # If target language is Tamil or Hindi, translate the answer.
    if original_lang in ['ta', 'hi']:
        answer = translator.translate(answer, dest=original_lang).text

    answer = clean_and_localize(answer, original_lang)
    return answer
