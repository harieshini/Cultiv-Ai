import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///chat.db")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
