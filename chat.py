import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load API key from .env file for Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key is missing! Set it in the .env file.")
    st.stop()

# Configure Gemini API with your API key
genai.configure(api_key=api_key)

# --- Load the Agriculture Classifier Model ---
tokenizer = AutoTokenizer.from_pretrained("smokxy/agri_bert_classifier-quantized")
classifier_model = AutoModelForSequenceClassification.from_pretrained("smokxy/agri_bert_classifier-quantized")

# (Optional) Define a label mapping based on your classifier's training.
# Adjust these labels as needed.
label_mapping = {
    0: "Soil Testing",
    1: "Pest Detection",
    2: "Crop Management",
    3: "Irrigation",
    4: "Harvesting",
    5: "Government Schemes"
}

# Function to classify the user's query using the AgriBERT classifier.
def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    # Apply softmax to get probabilities.
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_class].item()
    label = label_mapping.get(predicted_class, "General Agriculture")
    return label, confidence

# Function to interact with the Gemini API.
def chat_with_gemini(prompt, language, classification_info):
    try:
        model = genai.GenerativeModel("gemini-pro")
        # Base prompt with agriculture expertise.
        base_prompt = (
            "You are an expert in agriculture with extensive knowledge gained from a "
            "large, curated dataset covering soil testing, pest detection, crop management, "
            "irrigation, harvesting, and local Tamil Nadu government financial aids, farming schemes, "
            "and subsidies. Your responses should provide detailed and accurate information to help "
            "improve productivity, income, food security, and double agricultural output. "
        )
        # Append the classifier context.
        base_prompt += f"\n\nThe user's query has been classified as related to: {classification_info}.\n\n"
        # Append language-specific instruction.
        language_prompt = "Please provide your response in Tamil. " if language == "தமிழ்" else "Please provide your response in English. "
        # Final prompt combining everything.
        full_prompt = base_prompt + language_prompt + "User query: " + prompt
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI setup with language selection.
selected_language = st.radio("Choose Language / மொழியை தேர்ந்தெடுக்கவும்", ["English", "தமிழ்"])

if selected_language == "தமிழ்":
    st.title("🌱 CULTIV - AI 🚜 - தமிழ்")
    st.write("விவசாயம், பயிர்கள், மண் ஆரோக்கியம், பூச்சி கட்டுப்பாடு மற்றும் தமிழக அரசின் திட்டங்கள் பற்றி என்னிடம் கேளுங்கள்!")
    placeholder_text = "விவசாயம் பற்றி கேளுங்கள்..."
else:
    st.title("🌱 CULTIV - AI 🚜")
    st.write("Ask me anything about farming, crops, soil health, pest control, and Tamil Nadu government schemes!")
    placeholder_text = "Ask me about agriculture..."

# Maintain chat history using Streamlit session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input from the user.
user_input = st.chat_input(placeholder_text, key="user_input")

if user_input:
    # Classify the query using AgriBERT.
    predicted_label, confidence = classify_query(user_input)
    classification_info = f"{predicted_label} (confidence: {confidence:.2f})"
    
    # Append and display user's message.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get the response from Gemini, including classifier context.
    bot_response = chat_with_gemini(user_input, selected_language, classification_info)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(bot_response)
