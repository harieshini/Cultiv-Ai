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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from faster_whisper import WhisperModel

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(page_title="CULTIV AI", page_icon="🌾", layout="centered")

# --- Configuration & Imports ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Enhanced UI Styles ---
st.markdown("""
    <style>
    /* Enhanced UI Styles */
    .stApp {
        background: linear-gradient(135deg, #e6f7ec 100%)!important;
        background-attachment: fixed;
    }
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.95)!important;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stMarkdown {
        color: #1a4a3c;
        line-height: 1.6;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.98)!important;
        border-right: 1px solid #e0e0e0;
    }
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    [data-testid="stChatMessage"] > div:first-child {
        align-items: center;
        gap: 0.5rem;
    }
    /* User message styling */
    .user-message .stMarkdown {
        background: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        padding: 1rem !important;
        border-radius: 1.2rem 1.2rem 0 1.2rem !important;
    }
    /* Bot message styling */
    .assistant-message .stMarkdown {
        background: #f0fff8 !important;
        border: 1px solid #c1e5d3 !important;
        padding: 1rem !important;
        border-radius: 1.2rem 1.2rem 1.2rem 0 !important;
    }
    /* File uploader enhancements */
    .stFileUploader {
        border: 2px dashed #c1e5d3 !important;
        border-radius: 15px !important;
        background: rgba(240, 255, 248, 0.3) !important;
    }
    .stFileUploader:hover {
        border-color: #8dc8ad !important;
    }
    .uploaderLabel {font-size: 0.85rem; margin-bottom: 3px;}
    /* Progress spinner styling */
    .stSpinner > div {
        border-color: #1a4a3c !important;
        border-right-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- Enhanced UI Components ---
def show_typing_indicator():
    """Animated typing indicator"""
    return st.markdown("""
    <div class="typing-indicator" style="display:flex;gap:4px;padding:8px 12px;background:#f0fff8;border-radius:12px;width:fit-content">
        <div style="width:6px;height:6px;background:#1a4a3c;border-radius:50%;animation:pulse 1.4s infinite"></div>
        <div style="width:6px;height:6px;background:#1a4a3c;border-radius:50%;animation:pulse 1.4s infinite 0.2s"></div>
        <div style="width:6px;height:6px;background:#1a4a3c;border-radius:50%;animation:pulse 1.4s infinite 0.4s"></div>
    </div>
    <style>
    @keyframes pulse {
        0%, 60%, 100% { opacity: 1; }
        30% { opacity: 0.3; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Chat History Display ---
def display_chat_history():
    for entry in st.session_state.history:
        avatar = "👤" if entry["role"] == "user" else "🌾"
        with st.chat_message(entry["role"], avatar=avatar):
            if entry["type"] == "text":
                st.markdown(entry["content"], unsafe_allow_html=True)
                st.caption(f"*{entry.get('timestamp', '')}*")
            elif entry["type"] == "media":
                if entry["mtype"] == "image":
                    st.image(entry["content"], use_column_width=True)
                elif entry["mtype"] == "video":
                    st.video(entry["content"])
                elif entry["mtype"] == "audio":
                    st.audio(entry["content"])

# --- Core Functions ---
def process_image(image_file):
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

# --- UI Layout ---
with st.sidebar:
    lang = st.radio("🌐 Language/மொழி", ["English", "தமிழ்"], index=0,
                   help="Select your preferred interface language")
    
    # UX Controls
    st.markdown("---")
    st.markdown("**Chat Controls**")
    if st.button("🗑️ Clear Chat History" if lang == "English" else "🗑️ அரட்டை வரலாறு அழிக்க"):
        st.session_state.history = []
        st.rerun()
    
    # Display Processing Status
    st.markdown("---")
    st.markdown("**System Status**")
    processing_status = st.empty()
    processing_status.caption("🟢 System Ready" if lang == "English" else "🟢 கணினி தயாராக உள்ளது")

config = LANG_CONFIG[lang]
st.title(config["title"])

# Input Tabs
tab_chat, tab_media = st.tabs(config["tabs"])

# Chat Interface
with tab_chat:
    query = st.chat_input(config["prompts"]["placeholder"])

# Enhanced Media Interface
with tab_media:
    # Media type selector
    media_type = st.radio("📁 Select Media Type", 
                        ["Image", "Audio", "Video"],
                        horizontal=True,
                        label_visibility="collapsed")
    
    # Unified media processing container
    with st.container(border=True):
        cols = st.columns([1, 2], gap="large")
        
        with cols[0]:
            # Dynamic uploader based on selection
            upload_label = config["labels"][media_type.lower()]
            media_file =st.file_uploader(
                label=f"**{upload_label}**",
                type={
                    "Image": ["jpg", "png", "jpeg"],
                    "Audio": ["wav", "mp3"],
                    "Video": ["mp4", "mov"]
                }[media_type],
                help=(
                    "Supported formats: " + 
                    {"Image": "JPEG, PNG", 
                    "Audio": "WAV, MP3",
                    "Video": "MP4, MOV"}[media_type]
    )
            )            
            if media_file:
                processing_status.caption(f"📤 Uploading {media_type}..." if lang == "English" 
            else f"📤 {media_type} பதிவேற்றம் செய்கிறது...")

        with cols[1]:
            # Preview section
            if media_file:
                st.markdown("**Preview**" if lang == "English" else "**முன்னோட்டம்**")
                with st.spinner("Loading preview..." if lang == "English" else "முன்னோட்டத்தை ஏற்றுகிறது..."):
                    try:
                        if media_type == "Image":
                            st.image(media_file, use_column_width=True)
                        elif media_type == "Video":
                            st.video(media_file)
                        elif media_type == "Audio":
                            st.audio(media_file)
                    except Exception as e:
                        st.error(f"⚠️ Preview Error: {str(e)}" if lang == "English" 
                                else f"⚠️ முன்னோட்ட பிழை: {str(e)}")

# --- Chat History Management ---
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_processed = {
        "text": None,
        "image": None,
        "audio": None,
        "video": None
    }

display_chat_history()

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

if any([query, media_file]):
    current_media = media_file
    
    if current_media != st.session_state.last_processed.get(media_type.lower()):
        processed = True
        st.session_state.last_processed[media_type.lower()] = current_media
        
        try:
            if media_type == "Image":
                media_data = process_image(media_file)
                mtype = "image"
            elif media_type == "Video":
                media_data = process_video(media_file)
                mtype = "video"
            elif media_type == "Audio":
                current_input = transcribe_audio(media_file, lang)
                mtype = "audio"
            
            # Add to chat history
            st.session_state.history.append({
                "role": "user",
                "type": "media" if media_type != "Audio" else "text",
                "mtype": mtype,
                "content": media_data if media_type != "Audio" else current_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            
        except Exception as e:
            st.error(f"⚠️ Processing Error: {str(e)}" if lang == "English" 
                    else f"⚠️ செயலாக்க பிழை: {str(e)}")
            processing_status.caption("🔴 Processing Error" if lang == "English" 
                                     else "🔴 செயலாக்க பிழை")

    if query and query != st.session_state.last_processed.get("text"):
        processed = True
        current_input = query
        st.session_state.last_processed["text"] = query
        st.session_state.history.append({
            "role": "user",
            "type": "text",
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M")
        })

    if processed and (current_input or media_data):
        with st.status("🔍 Analyzing..." if lang == "English" else "🔍 ஆய்வு செய்கிறது...", expanded=True) as status:
            try:
                with st.empty():
                    show_typing_indicator()
                    response = generate_response(current_input or "", media_data)
                    status.update(label="✅ Analysis Complete" if lang == "English" 
                                 else "✅ பகுப்பாய்வு முடிந்தது", state="complete")
                st.session_state.history.append({
                    "role": "assistant",
                    "type": "text",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}" if lang == "English" else f"⚠️ பிழை: {str(e)}")
                processing_status.caption("🔴 Processing Error" if lang == "English" 
                                         else "🔴 செயலாக்க பிழை")

    st.rerun()
