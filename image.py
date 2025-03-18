from PIL import Image
import google.generativeai as genai

def analyze_image_with_gemini(image_path):
    """
    Analyze an image using the Gemini Vision API to describe plant diseases or other insights.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-vision")
        img = Image.open(image_path)
        # Optionally resize image for processing efficiency
        img = img.resize((256, 256))
        response = model.generate_content(["Describe this image", img])
        return response.text.strip()
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

