import re
import google.generativeai as genai
from config import GEMINI_API_KEY
from googletrans import Translator

# Configure API Key for Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize translator instance
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
    Cleans up the generated answer by removing markdown formatting, URLs, and extra whitespace.
    Also replaces key agricultural terms with colloquial equivalents for Tamil or Hindi.
    """
    # Remove markdown markers (e.g., **)
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
    Determines whether the given question is agriculture-related.
    It sends a classification prompt to the Gemini API and expects a one-word answer: 'yes' or 'no'.
    Returns True if the answer starts with 'yes'; otherwise, False.
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
    
    Efficiency improvements:
    - If the prompt contains "in tamil" or "in hindi", that phrase is removed and the target language is forced.
    - If the target language is not English, the prompt is first translated into English for robust generation.
    - The generated answer is then translated back into the target language.
    - Finally, the answer is cleaned and key agricultural terms are localized.
    
    Returns the final, efficient, and easily understandable answer.
    """
    prompt_lower = prompt.lower()
    # Check for explicit language instruction in the prompt.
    if "in tamil" in prompt_lower:
        target_lang = "ta"
        prompt = re.sub(r"in tamil", "", prompt, flags=re.IGNORECASE).strip()
    elif "in hindi" in prompt_lower:
        target_lang = "hi"
        prompt = re.sub(r"in hindi", "", prompt, flags=re.IGNORECASE).strip()
    else:
        detected = translator.detect(prompt)
        target_lang = detected.lang if detected else "en"
    
    # For robust generation, translate prompt to English if target language is not English.
    if target_lang != "en":
        prompt_en = translator.translate(prompt, dest="en").text
    else:
        prompt_en = prompt

    try:
        response = model.generate_content(prompt_en)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error in text generation: {e}"
    
    # If target language is not English, translate the answer back.
    if target_lang != "en":
        answer = translator.translate(answer, dest=target_lang).text
    
    # Clean and localize the final answer.
    answer = clean_and_localize(answer, target_lang)
    return answer
