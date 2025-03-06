import streamlit as st
import google.generativeai as genai
import os
import torch
import cv2
import numpy as np
import tempfile
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from faster_whisper import WhisperModel

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(page_title="CULTIV AI", page_icon="🌾", layout="centered")

# --- Configuration & Imports ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Model Loading ---
@st.cache_resource
def load_models():
    return {
        "classifier": {
            "tokenizer": AutoTokenizer.from_pretrained("smokxy/agri_bert_classifier-quantized"),
            "model": AutoModelForSequenceClassification.from_pretrained("smokxy/agri_bert_classifier-quantized")
        },
        "whisper": WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8"),
        "gemini-vision": genai.GenerativeModel('gemini-1.5-flash'),
        "gemini-text": genai.GenerativeModel('gemini-pro')
    }

models = load_models()

# --- UI Configuration ---
st.markdown("""
    <style>
    .stApp {
            background-size: cover;
            backdrop-filter: blur(2px);}
    .stChatInputContainer {background: rgba(255, 255, 255, 0.9)!important;}
    .stMarkdown {color: #1a4a3c;}
    .sidebar .sidebar-content {background: rgba(245, 245, 245, 0.95)!important;}
    .uploaderLabel {font-size: 0.85rem; margin-bottom: 3px;}
    </style>
""", unsafe_allow_html=True)

# --- Language Setup ---
LANG_CONFIG = {
    "English": {
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
    },
    "தமிழ்": {
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
}

# --- Core Functions ---
def process_image(image_file):
    # Check if we've already processed this exact file
    if "last_image" in st.session_state:
        if (st.session_state.last_image["name"] == image_file.name and
            st.session_state.last_image["size"] == image_file.size):
            return None
    st.session_state.last_image = {
        "name": image_file.name,
        "size": image_file.size
    }
    return Image.open(image_file)

def classify_query(text, lang):
    inputs = models["classifier"]["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = models["classifier"]["model"](**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(probs).item(), probs.max().item()

def transcribe_audio(audio_file, lang):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            segments, _ = models["whisper"].transcribe(
                tmp.name, 
                language="ta" if lang == "தமிழ்" else "en",
                beam_size=5
            )
        return " ".join([segment.text for segment in segments])
    except Exception as e:
        return f"Transcription Error: {str(e)}"

def process_video(video_file):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        cap = cv2.VideoCapture(tmp.name)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if len(frames) % 30 == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames[:3]

# --- UI Components ---
with st.sidebar:
    lang = st.radio("🌐 Language/மொழி", ["English", "தமிழ்"], index=0)

config = LANG_CONFIG[lang]
st.title(config["title"])

# Input Tabs
tab_chat, tab_media = st.tabs(config["tabs"])

# Chat Interface
with tab_chat:
    query = st.chat_input(config["prompts"]["placeholder"])

# Media Interface
with tab_media:
    cols = st.columns(3)
    with cols[0]:
        img_file = st.file_uploader(config["labels"]["image"], type=["jpg", "png", "jpeg"])
    with cols[1]:
        audio_file = st.file_uploader(config["labels"]["audio"], type=["wav", "mp3"])
    with cols[2]:
        vid_file = st.file_uploader(config["labels"]["video"], type=["mp4", "mov"])

# --- Chat History Management ---
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_processed = {
        "text": None,
        "image": None,
        "audio": None,
        "video": None
    }

# Display History
for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        if entry["type"] == "text":
            st.markdown(entry["content"])
        elif entry["type"] == "media":
            if entry["mtype"] == "image":
                st.image(entry["content"])
            elif entry["mtype"] == "video":
                st.video(entry["content"])

# --- Processing Logic ---
def generate_response(prompt, media=None):
    class_id, confidence = classify_query(prompt, lang)
    prompt_template = f"{config['prompts']['system']} (Confidence: {confidence:.0%})"
    
    if media:
        response = models["gemini-vision"].generate_content([prompt_template + prompt, media])
    else:
        response = models["gemini-text"].generate_content(prompt_template + prompt)
    
    return response.text

current_input = None
media_data = None
processed = False

# Check for new inputs
if any([query, img_file, audio_file, vid_file]):
    # Track current inputs
    current_media = img_file or vid_file or audio_file
    
    # Only process if new input detected
    if current_media != st.session_state.last_processed or query != st.session_state.last_processed:
        processed = True
        st.session_state.last_processed = current_media or query
        
        # Handle Image/Video
        if img_file:
            media_data = process_image(img_file)
    if media_data:  # Only process if new image
        processed = True
        st.session_state.history.append({
            "role": "user",
            "type": "media",
            "mtype": "image",
            "content": media_data
        })
    if vid_file and vid_file != st.session_state.last_processed["video"]:
        processed = True
        media_data = process_video(vid_file)
        st.session_state.last_processed["video"] = vid_file
        st.session_state.history.append({
        "role": "user",
        "type": "media",
        "mtype": "video",
        "content": vid_file
    })
        
        # Handle Audio
        if audio_file and audio_file != st.session_state.last_processed["audio"]:
            processed = True
            current_input = transcribe_audio(audio_file, lang)
            st.session_state.last_processed["audio"] = audio_file
            st.session_state.history.append({
        "role": "user",
        "type": "media",
        "mtype": "audio",
        "content": audio_file
    })
        
        # Handle Text
        if query and query != st.session_state.last_processed["text"]:
            processed = True
            current_input = query
            st.session_state.last_processed["text"] = query
            st.session_state.history.append({
        "role": "user",
        "type": "text",
        "content": query
    })
        
        # Generate Response
        if processed and (current_input or media_data):
            with st.spinner("🔍 Analyzing..." if lang == "English" else "🔍 ஆய்வு செய்கிறது..."):
                try:
                    response = generate_response(current_input or "", media_data)
                    st.session_state.history.append({
                "role": "assistant",
                "type": "text",
                "content": response
            })
                except Exception as e:
                    st.error(f"Error: {str(e)}" if lang == "English" else f"பிழை: {str(e)}")
    
    # Clear file uploaders after processing
    if img_file:
        img_file = None
    if audio_file:
        audio_file = None
    if vid_file:
        vid_file = None
        
    st.rerun()
