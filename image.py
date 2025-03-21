import os
import asyncio
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import nest_asyncio
from flask import session  # Import session to check for the default language
from googletrans import Translator

# Apply nest_asyncio to allow nested async calls in environments that require it
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini models for vision and text tasks
gemini_vision = genai.GenerativeModel('gemini-1.5-flash')
gemini_text = genai.GenerativeModel('gemini-1.5-pro')

# Global translator instance for language translation
translator = Translator()

def analyze_image(image_path):
    """
    Analyze an agricultural image using the Gemini vision model.
    Returns a caption describing field conditions, crop health, and any issues.
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        prompt = (
            "Analyze this agricultural image and provide a brief caption describing the field conditions, "
            "crop health, and any visible issues."
        )
        
        async def async_analyze():
            response = await gemini_vision.generate_content_async([prompt, image])
            return response.text
        
        caption = asyncio.run(async_analyze())
        return caption
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

def generate_solution_from_analysis(analysis_text):
    """
    Generate a detailed solution based on the agricultural image analysis using the Gemini text model.
    Translates the solution to the user's default language if set.
    """
    try:
        prompt = (
            f"Based on the following agricultural image analysis, provide a detailed solution addressing any identified issues "
            f"or recommendations for improvement. Do not include any copyrighted content.\n\n"
            f"Analysis:\n{analysis_text}\n\nSolution:"
        )
        
        async def async_generate():
            response = await gemini_text.generate_content_async(prompt)
            return response.text
        
        solution_text = asyncio.run(async_generate())
        
        # Use the default language from session if available; default to English.
        target_lang = session.get('default_language', 'en')
        if target_lang != 'en':
            solution_text = translator.translate(solution_text, dest=target_lang).text
            
        return solution_text
    except Exception as e:
        return f"Error in generating solution: {str(e)}"

def generate_solution_image(solution_text, output_path="solution_image.jpg"):
    """
    Generate an image representing the solution using the Gemini vision model.
    Returns the file path of the generated image.
    """
    try:
        prompt = (
            f"Create a realistic agricultural field image that visually represents the following solution: {solution_text}"
        )
        
        async def async_generate():
            response = await gemini_vision.generate_content_async(prompt)
            # Assumes the response contains an image object in the 'image' attribute
            return response.image
        
        generated_image = asyncio.run(async_generate())
        generated_image.save(output_path)
        return output_path
    except Exception as e:
        return f"Error in generating solution image: {str(e)}"

def generate_follow_up_response(context, follow_up_query):
    """
    Generate a follow-up response based on a given context and a follow-up query.
    Translates the answer to the user's default language if set.
    """
    try:
        prompt = (
            f"Based on the following context:\n{context}\n\n"
            f"And the follow-up question:\n{follow_up_query}\n\n"
            "Provide a detailed follow-up response with additional insights or recommendations."
        )
        
        async def async_followup():
            response = await gemini_text.generate_content_async(prompt)
            return response.text
        
        follow_up_response = asyncio.run(async_followup())
        
        # Translate to the user's default language if necessary
        target_lang = session.get('default_language', 'en')
        if target_lang != 'en':
            follow_up_response = translator.translate(follow_up_response, dest=target_lang).text
        
        return follow_up_response
    except Exception as e:
        return f"Error in generating follow-up response: {str(e)}"

def analyze_and_generate_solution(image_path, solution_image_output="solution_image.jpg"):
    """
    Analyzes the provided agricultural image, generates a solution text, and creates an image representing the solution.
    Returns a tuple: (solution_text, solution_image_path).
    """
    analysis = analyze_image(image_path)
    if analysis.startswith("Error"):
        return (analysis, None)
    
    solution_text = generate_solution_from_analysis(analysis)
    if solution_text.startswith("Error"):
        return (solution_text, None)
    
    solution_image = generate_solution_image(solution_text, output_path=solution_image_output)
    if isinstance(solution_image, str) and solution_image.startswith("Error"):
        return (solution_text, None)
    
    return (solution_text, solution_image)
