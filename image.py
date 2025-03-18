from PIL import Image
import os
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, pipeline
import pollinations as ai

# Initialize the image captioning model (PyTorch-based)
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initialize the text generation pipeline using GPT-2 (CPU-based)
text_generator = pipeline("text-generation", model="gpt2", device=-1)

def analyze_image(image_path):
    """
    Analyze an image using a PyTorch-based image captioning model.
    Returns a caption that describes the image.
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        # Generate caption with the image captioning model
        output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

def generate_solution_from_analysis(analysis_text):
    """
    Generate a solution text based on the image analysis using GPT-2.
    """
    try:
        prompt = (
            f"Based on the following image analysis, provide a detailed solution to address the issue:\n"
            f"{analysis_text}\nSolution:"
        )
        output = text_generator(prompt, max_length=100, num_return_sequences=1)
        solution_text = output[0]['generated_text']
        return solution_text
    except Exception as e:
        return f"Error in generating solution: {str(e)}"

def generate_solution_image(solution_text, output_path="solution_image.jpg"):
    """
    Generate an image representing the solution using Pollinations AI.
    """
    try:
        model_obj = ai.Model()
        # Construct a prompt for Pollinations AI image generation
        prompt = f"{solution_text} {ai.realistic}"
        # Generate image using Pollinations AI (this runs on CPU and does not require a GPU)
        image = model_obj.generate(
            prompt=prompt,
            model=ai.flux,
            width=1024,
            height=1024,
            seed=42
        )
        image.save(output_path)
        return output_path
    except Exception as e:
        return f"Error in generating solution image: {str(e)}"

def analyze_and_generate_solution(image_path, solution_image_output="solution_image.jpg"):
    """
    Analyzes the uploaded image, generates a solution text, and creates an image representing the solution.
    Returns a tuple (solution_text, solution_image_path).
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
