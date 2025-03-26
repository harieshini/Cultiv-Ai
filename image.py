import os
import asyncio
import threading
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import nest_asyncio
from flask import session
from googletrans import Translator

# Apply nest_asyncio to allow nested async calls
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

# Create a global event loop and run it in a separate background thread
global_loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

loop_thread = threading.Thread(target=start_loop, args=(global_loop,))
loop_thread.daemon = True
loop_thread.start()

def run_async(coro):
    """
    Run an asynchronous coroutine on the global event loop.
    """
    future = asyncio.run_coroutine_threadsafe(coro, global_loop)
    return future.result()

def analyze_image(image_path):
    """
    Analyze an agricultural image using the Gemini vision model.
    Returns a caption describing field conditions, crop health, and any visible issues.
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
        
        caption = run_async(async_analyze())
        return caption
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

def analyze_image_with_status(image_path):
    """
    Analyzes the image while providing simulated loader status updates.
    Returns a tuple of (status_message, analysis_result).
    """
    loader_status = "Analyzing image..."
    analysis = analyze_image(image_path)
    if analysis.startswith("Error"):
        loader_status = analysis
    else:
        loader_status = "Image analysis complete."
    return loader_status, analysis

def generate_solution_from_analysis(context_text):
    """
    Generate a detailed solution based on the provided context (from image analysis or text/audio).
    Translates the solution to the user's default language if set.
    """
    if not context_text or context_text.startswith("Error"):
        return "No valid context available to generate a solution."
    try:
        prompt = (
            f"Based on the following context, provide a detailed solution addressing any identified issues "
            f"or recommendations for improvement. Do not include any copyrighted content.\n\n"
            f"Context:\n{context_text}\n\nSolution:"
        )
        
        async def async_generate():
            response = await gemini_text.generate_content_async(prompt)
            return response.text
        
        solution_text = run_async(async_generate())
        target_lang = session.get('default_language', 'en')
        if target_lang != 'en':
            solution_text = translator.translate(solution_text, dest=target_lang).text
            
        return solution_text
    except Exception as e:
        return f"Error in generating solution text: {str(e)}"

def generate_solution_image(solution_text, output_path="solution_image.jpg"):
    """
    Generate an image representing the solution using the Gemini vision model.
    Returns the file path of the generated image or a friendly message if generation fails.
    """
    if not solution_text or solution_text.startswith("Error"):
        return "No valid solution text available to generate an image."
    try:
        prompt = (
            f"Create a realistic agricultural field image that visually represents the following solution: {solution_text}"
        )
        
        async def async_generate():
            response = await gemini_vision.generate_content_async(prompt)
            # Assumes the response contains an image object in the 'image' attribute
            return response.image
        
        generated_image = run_async(async_generate())
        generated_image.save(output_path)
        return output_path
    except Exception as e:
        return f"Error in generating solution image: {str(e)}"

def generate_follow_up_response(context, follow_up_query):
    """
    Generate a follow-up response based on a given context and a follow-up query.
    Uses the stored analysis if context is not explicitly provided.
    Translates the answer to the user's default language if set.
    """
    if not context:
        context = session.get("image_analysis", "")
    if not context:
        return "No analysis available. Please upload an image or provide a valid context first."
    try:
        prompt = (
            f"Based on the following context:\n{context}\n\n"
            f"And the follow-up question:\n{follow_up_query}\n\n"
            "Provide a detailed follow-up response with additional insights or recommendations."
        )
        
        async def async_followup():
            response = await gemini_text.generate_content_async(prompt)
            return response.text
        
        follow_up_response = run_async(async_followup())
        target_lang = session.get('default_language', 'en')
        if target_lang != 'en':
            follow_up_response = translator.translate(follow_up_response, dest=target_lang).text
        
        return follow_up_response
    except Exception as e:
        return f"Error in generating follow-up response: {str(e)}"

def analyze_and_store_image(image_path):
    """
    Analyzes the provided agricultural image and returns the analysis result.
    Intended to be called immediately after image upload.
    """
    _, analysis = analyze_image_with_status(image_path)
    return analysis

def generate_solution_response(context_text):
    """
    Generates the solution response (both text and image) when the user triggers it.
    Uses the provided context (from image analysis or text/audio input) to generate a solution.
    Returns a tuple: (solution_text, solution_image_path).
    """
    if not context_text or context_text.startswith("Error"):
        return "No valid context available to generate a solution.", None

    solution_text = generate_solution_from_analysis(context_text)
    if solution_text.startswith("Error"):
        return solution_text, None

    solution_image = generate_solution_image(solution_text)
    if isinstance(solution_image, str) and solution_image.startswith("Error"):
        return solution_text, None
    
    return solution_text, solution_image

def generate_custom_image(prompt):
    """
    Generates an image for a custom prompt when the user instructs (e.g., using keywords like 'generate an image').
    Returns the path of the generated image or a friendly error message.
    """
    try:
        async def async_generate():
            response = await gemini_vision.generate_content_async(prompt)
            return response.image
        generated_image = run_async(async_generate())
        output_path = "custom_generated_image.jpg"
        generated_image.save(output_path)
        return output_path
    except Exception as e:
        target_lang = session.get('default_language', 'en')
        message = "Image generation prompt is unclear." if target_lang == "en" \
                  else translator.translate("Image generation prompt is unclear.", dest=target_lang).text
        return message

def generate_text_as_image(text):
    """
    Generates an image representation of the provided text.
    Returns the generated image path or an error message in the user's language if unsuccessful.
    """
    if not text or len(text.strip()) == 0:
        target_lang = session.get('default_language', 'en')
        message = "Text for image generation is unclear." if target_lang == "en" \
                  else translator.translate("Text for image generation is unclear.", dest=target_lang).text
        return message
    prompt = f"Generate an image that visually represents the following text: {text}"
    return generate_custom_image(prompt)
