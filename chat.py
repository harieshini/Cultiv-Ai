import streamlit as st
import google.generativeai as genai
import os
import torch
import cv2
import numpy as np
import tempfile
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
import asyncio
import sys
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix event loop issue for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Page Configuration ---
st.set_page_config(
    page_title="CULTIV AI",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Configuration & Environment ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- UI Styles ---
STYLES = """
<style>
    :root {
        --primary-color: #1a4a3c;
        --secondary-color: #c1e5d3;
        --background-gradient: linear-gradient(135deg, #e6f7ec 0%);
    }

    .stApp {
        background: var(--background-gradient) !important;
        background-attachment: fixed !important;
    }

    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }

    .stMarkdown {
        color: var(--primary-color) !important;
        line-height: 1.6 !important;
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.98) !important;
        border-right: 1px solid #e0e0e0 !important;
    }

    [data-testid="stChatMessage"] {
        padding: 1rem !important;
        border-radius: 1rem !important;
        margin: 0.5rem 0 !important;
        transition: transform 0.2s ease-in-out !important;
    }

    [data-testid="stChatMessage"]:hover {
        transform: translateX(5px);
    }

    .user-message .stMarkdown {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        padding: 1rem !important;
        border-radius: 1.2rem 1.2rem 0 1.2rem !important;
    }

    .assistant-message .stMarkdown {
        background: #f0fff8 !important;
        border: 1px solid var(--secondary-color) !important;
        padding: 1rem !important;
        border-radius: 1.2rem 1.2rem 1.2rem 0 !important;
    }

    .stFileUploader {
        border: 2px dashed var(--secondary-color) !important;
        border-radius: 15px !important;
        background: rgba(240, 255, 248, 0.3) !important;
    }

    @keyframes pulse {
        0%, 60%, 100% { opacity: 1; }
        30% { opacity: 0.3; }
    }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Any]:
    """Load and cache all ML models"""
    logger.info("Loading models...")
    
    models = {
        "classifier": {
            "tokenizer": AutoTokenizer.from_pretrained("smokxy/agri_bert_classifier-quantized"),
            "model": AutoModelForSequenceClassification.from_pretrained(
                "smokxy/agri_bert_classifier-quantized"
            )
        },
        "whisper": WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8",
            download_root="./whisper_cache"
        ),
        "gemini": {
            "vision": genai.GenerativeModel('gemini-1.5-flash'),
            "text": genai.GenerativeModel('gemini-pro')
        }
    }
    
    logger.info("All models loaded successfully")
    return models

models = load_models()

# --- Language Configuration ---
class LanguageConfig:
    def __init__(self, lang: str = "English"):
        self.lang = lang
        self.config = {
            "English": self.english(),
            "தமிழ்": self.tamil()
        }[lang]

    def english(self) -> Dict[str, Any]:
        return {
            "title": "🌱 CULTIV - AI Farming Assistant",
            "tabs": ["💬 Chat", "📸 Media Input"],
            "prompts": {
                "system": "You are an agricultural expert specializing in Tamil Nadu. Provide detailed advice in English about: ",
                "placeholder": "Ask about crops, soil, or schemes..."
            },
            "labels": {
                "image": "Upload Field Photo",
                "audio": "Record Voice Query",
                "video": "Upload Field Video"
            }
        }

    def tamil(self) -> Dict[str, Any]:
        return {
            "title": "🌱 CULTIV - AI விவசாய உதவியாளர்",
            "tabs": ["💬 அரட்டை", "📸 ஊடக உள்ளீடு"],
            "prompts": {
                "system": "தமிழ்நாடு விவசாயத்தில் நிபுணத்துவம் பெற்ற ஒரு AI உதவியாளர். தமிழில் விரிவான ஆலோசனைகளை வழங்குக: ",
                "placeholder": "பயிர்கள், மண், அரசுத் திட்டங்கள் பற்றிக் கேளுங்கள்..."
            },
            "labels": {
                "image": "வயல் படத்தை பதிவேற்றம் செய்க",
                "audio": "குரல் கேள்வியை பதிவு செய்க",
                "video": "வயல் வீடியோவை பதிவேற்றம் செய்க"
            }
        }

# --- Media Processing Utilities ---
class MediaProcessor:
    @staticmethod
    def process_image(file) -> Optional[Image.Image]:
        try:
            return Image.open(file)
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return None

    @staticmethod
    def process_video(file) -> Optional[list]:
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(file.read())
                cap = cv2.VideoCapture(tmp.name)
                frames = []
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % 30 == 0:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    frame_count += 1
                    
                return frames[:3]  # Return first 3 sampled frames
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return None

    @staticmethod
    async def transcribe_audio(file, lang: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                tmp.write(file.read())
                segments, _ = models["whisper"].transcribe(
                    tmp.name,
                    language="ta" if lang == "தமிழ்" else "en",
                    beam_size=5,
                    vad_filter=True
                )
                return " ".join(segment.text for segment in segments)
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""

# --- Core Chat Functionality ---
class ChatManager:
    def __init__(self):
        self.history = st.session_state.get("history", [])
        self.last_processed = st.session_state.get("last_processed", {
            "text": None,
            "image": None,
            "audio": None,
            "video": None
        })

    def add_message(self, role: str, content: Any, mtype: str = "text"):
        self.history.append({
            "role": role,
            "type": mtype,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.session_state.history = self.history

    def display_history(self):
        for entry in self.history:
            avatar = "👤" if entry["role"] == "user" else "🌾"
            with st.chat_message(entry["role"], avatar=avatar):
                if entry["type"] == "text":
                    st.markdown(entry["content"], unsafe_allow_html=True)
                    st.caption(f"*{entry['timestamp']}*")
                elif entry["type"] == "media":
                    self._display_media(entry["content"], entry.get("mtype"))

    def _display_media(self, content, mtype: str):
        if mtype == "image":
            st.image(content, use_column_width=True)
        elif mtype == "video":
            st.video(content)
        elif mtype == "audio":
            st.audio(content)

    def clear_history(self):
        self.history = []
        st.session_state.history = []
        self.last_processed = {}
        st.rerun()

# --- AI Response Generation ---
class ResponseGenerator:
    def __init__(self, lang: str):
        self.lang = lang
        self.config = LanguageConfig(lang).config

    async def generate(self, prompt: str, media=None) -> str:
        try:
            class_id, confidence = self._classify_query(prompt)
            prompt_template = f"{self.config['prompts']['system']} (Confidence: {confidence:.0%})"
            
            if media:
                return await self._generate_vision_response(prompt_template, prompt, media)
            return await self._generate_text_response(prompt_template, prompt)
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return self._error_message()

    def _classify_query(self, text: str) -> tuple:
        inputs = models["classifier"]["tokenizer"](
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        with torch.no_grad():
            outputs = models["classifier"]["model"](**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return torch.argmax(probs).item(), probs.max().item()

    async def _generate_vision_response(self, prompt_template: str, prompt: str, media) -> str:
        response = await models["gemini"]["vision"].generate_content_async(
            [prompt_template + prompt, media]
        )
        return response.text

    async def _generate_text_response(self, prompt_template: str, prompt: str) -> str:
        response = await models["gemini"]["text"].generate_content_async(
            prompt_template + prompt
        )
        return response.text

    def _error_message(self) -> str:
        return {
            "English": "⚠️ Sorry, I'm having trouble processing your request. Please try again.",
            "தமிழ்": "⚠️ மன்னிக்கவும், உங்கள் கோரிக்கையை செயலாக்க என்னால் படவில்லை. மீண்டும் முயற்சிக்கவும்."
        }[self.lang]

# --- UI Components ---
def language_selector() -> str:
    with st.sidebar:
        return st.radio(
            "🌐 Language/மொழி",
            ["English", "தமிழ்"],
            index=0,
            help="Select your preferred interface language"
        )

def sidebar_controls(chat_manager: ChatManager, lang: str):
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Chat Controls**")
        if st.button("🗑️ Clear History" if lang == "English" else "🗑️ அழி"):
            chat_manager.clear_history()

        st.markdown("---")
        st.markdown("**System Status**")
        processing_status = st.empty()
        processing_status.caption(
            "🟢 System Ready" if lang == "English" 
            else "🟢 கணினி தயாராக உள்ளது"
        )
        return processing_status

# --- Main Application ---
# ... (previous imports and configuration remain the same)

# --- Main Application Class ---
class CultivAI:
    def __init__(self):
        self.lang = "English"
        self.chat_manager = ChatManager()
        self.processing_status = None
        self.media_processor = MediaProcessor()
        self.response_generator = None

    async def run(self):
        """Main application runner"""
        self.lang = language_selector()
        config = LanguageConfig(self.lang).config
        self.response_generator = ResponseGenerator(self.lang)
        self.processing_status = sidebar_controls(self.chat_manager, self.lang)

        st.title(config["title"])
        tab_chat, tab_media = st.tabs(config["tabs"])

        user_input, media_file, media_type = self.handle_inputs(tab_chat, tab_media, config)
        self.chat_manager.display_history()

        if user_input or media_file:
            await self.process_input(user_input, media_file, media_type)

    def handle_inputs(self, tab_chat, tab_media, config):
        """Handle user inputs from both tabs"""
        user_input = None
        media_file = None
        media_type = None

        with tab_chat:
            user_input = st.chat_input(config["prompts"]["placeholder"])

        with tab_media:
            media_type = st.radio(
                "📁 Select Media Type",
                ["Image", "Audio", "Video"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            media_file = self.render_media_uploader(media_type, config)
        
        return user_input, media_file, media_type

    def render_media_uploader(self, media_type, config):
        """Render media uploader components"""
        upload_label = config["labels"][media_type.lower()]
        
        with st.container(border=True):
            cols = st.columns([1, 2], gap="large")
            
            with cols[0]:
                return st.file_uploader(
                    label=f"**{upload_label}**",
                    type=["jpg", "png", "jpeg"] if media_type == "Image" 
                        else ["wav", "mp3"] if media_type == "Audio" 
                        else ["mp4", "mov"],
                    help="Supported formats: " + {
                        "Image": "JPEG, PNG", 
                        "Audio": "WAV, MP3",
                        "Video": "MP4, MOV"
                    }[media_type]
                )
            
            with cols[1]:
                if media_file:
                    self.render_media_preview(media_type, media_file)

    def render_media_preview(self, media_type, media_file):
        """Display media preview with error handling"""
        st.markdown("**Preview**" if self.lang == "English" else "**முன்னோட்டம்**")
        
        try:
            if media_type == "Image":
                st.image(media_file, use_column_width=True)
            elif media_type == "Video":
                st.video(media_file)
            elif media_type == "Audio":
                st.audio(media_file)
        except Exception as e:
            error_msg = {
                "English": f"⚠️ Preview Error: {str(e)}",
                "தமிழ்": f"⚠️ முன்னோட்ட பிழை: {str(e)}"
            }[self.lang]
            st.error(error_msg)

    async def process_input(self, user_input, media_file, media_type):
        """Process user input and generate response"""
        media_data = None
        
        if media_file:
            self.processing_status.caption(
                f"📤 Processing {media_type}..." if self.lang == "English"
                else f"📤 {media_type} செயலாக்கப்படுகிறது..."
            )
            
            if media_type == "Image":
                media_data = self.media_processor.process_image(media_file)
            elif media_type == "Video":
                media_data = self.media_processor.process_video(media_file)
            elif media_type == "Audio":
                user_input = await self.media_processor.transcribe_audio(media_file, self.lang)

            if media_data or user_input:
                self.chat_manager.add_message(
                    "user", 
                    media_data or user_input, 
                    "media" if media_type != "Audio" else "text"
                )

        if user_input and user_input != self.chat_manager.last_processed.get("text"):
            self.chat_manager.add_message("user", user_input)
            self.chat_manager.last_processed["text"] = user_input

        if user_input or media_data:
            with st.status("🔍 Analyzing..." if self.lang == "English" else "🔍 ஆய்வு செய்கிறது..."):
                response = await self.response_generator.generate(user_input or "", media_data)
                self.chat_manager.add_message("assistant", response)
            
            self.processing_status.caption(
                "🟢 Ready" if self.lang == "English"
                else "🟢 தயார்"
            )
            st.rerun()

def main():
    """Main entry point"""
    app = CultivAI()
    asyncio.run(app.run())

if __name__ == "__main__":
    main()