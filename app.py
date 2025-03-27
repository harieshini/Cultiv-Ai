import os
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from config import UPLOAD_FOLDER, DATABASE_URI
from models import db, ChatMessage
from text import is_valid_question, generate_text_response
from image import (
    analyze_and_store_image,
    generate_solution_response,
    generate_follow_up_response,
    generate_custom_image,
    generate_text_as_image
)
from audio import process_audio, save_audio
from video import process_video
from werkzeug.utils import secure_filename
from googletrans import Translator
from PIL import Image

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'

db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
translator = Translator()

def detect_language(text):
    if 'default_language' in session:
        return session['default_language']
    detected_lang = translator.detect(text).lang
    return detected_lang if detected_lang in ['ta', 'hi', 'en'] else 'en'

@app.route("/", methods=["GET", "POST"])
def index():
    chats = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    return render_template("index.html", chats=chats)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    user_message = data['message']
    if user_message:
        lower_msg = user_message.lower()
        if "generate an image" in lower_msg:
            custom_image_path = generate_custom_image(user_message)
            if not custom_image_path.startswith("Error"):
                bot_response = (f"<div class='image-container'><div class='input-text'>Input: {user_message}</div>"
                                f"<img src='/{custom_image_path}' alt='Custom Generated Image'/>"
                                f"<a class='download-btn' href='/{custom_image_path}' download>Download Image</a></div>")
            else:
                bot_response = custom_image_path
            user_chat = ChatMessage(role="user", message=user_message)
            bot_chat = ChatMessage(role="bot", message=bot_response)
            db.session.add_all([user_chat, bot_chat])
            db.session.commit()
            return jsonify({"user": user_message, "bot": bot_response})
        elif "text as image" in lower_msg:
            image_from_text = generate_text_as_image(user_message)
            if not image_from_text.startswith("Error"):
                bot_response = (f"<div class='image-container'><div class='input-text'>Input: {user_message}</div>"
                                f"<img src='/{image_from_text}' alt='Text as Image'/>"
                                f"<a class='download-btn' href='/{image_from_text}' download>Download Image</a></div>")
            else:
                bot_response = image_from_text
            user_chat = ChatMessage(role="user", message=user_message)
            bot_chat = ChatMessage(role="bot", message=bot_response)
            db.session.add_all([user_chat, bot_chat])
            db.session.commit()
            return jsonify({"user": user_message, "bot": bot_response})
        
        if not is_valid_question(user_message):
            bot_response = "I specialize in Agriculture."
        else:
            bot_response = generate_text_response(user_message)
        
        user_chat = ChatMessage(role="user", message=user_message)
        bot_chat = ChatMessage(role="bot", message=bot_response)
        db.session.add_all([user_chat, bot_chat])
        db.session.commit()
        return jsonify({"user": user_message, "bot": bot_response})
    return jsonify({"error": "No message provided"}), 400

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            relative_path = file_path.replace("\\", "/")
            user_image_msg = ChatMessage(
                role="user", 
                message=(f"<div class='image-container'><div class='input-text'>Uploaded: {file.filename}</div>"
                         f"<img src='/{relative_path}' alt='Uploaded Image'/>"
                         f"<a class='download-btn' href='/{relative_path}' download>Download Image</a></div>")
            )
            db.session.add(user_image_msg)
            db.session.commit()

            analysis = analyze_and_store_image(file_path)
            session['image_analysis'] = analysis
            status_msg = "Image uploaded and analyzed. Please use 'Generate Solution' to continue."
            bot_status = ChatMessage(role="bot", message=status_msg)
            db.session.add(bot_status)
            db.session.commit()
    return redirect(url_for('index'))

@app.route("/generate_solution", methods=["POST"])
def generate_solution():
    analysis = session.get("image_analysis")
    if not analysis:
        bot_msg = ChatMessage(role="bot", message="No image context available. Please upload an image first.")
        db.session.add(bot_msg)
        db.session.commit()
        return redirect(url_for("index"))
    
    solution_text, solution_image_path = generate_solution_response(analysis)
    chat_text = ChatMessage(role="bot", message=f"Solution: {solution_text}")
    db.session.add(chat_text)
    if solution_image_path and not solution_image_path.startswith("Error"):
        try:
            img = Image.open(solution_image_path)
            width, height = img.size
            new_width, new_height = 1024, 955
            left = (width - new_width) / 2 if width > new_width else 0
            top = (height - new_height) / 2 if height > new_height else 0
            right = left + new_width
            bottom = top + new_height
            cropped = img.crop((left, top, right, bottom))
            cropped.save(solution_image_path)
        except Exception as e:
            print(f"Error cropping image: {e}")
        chat_image = ChatMessage(
            role="bot",
            message=(f"<div class='image-container'><div class='input-text'>Context: {analysis}</div>"
                     f"<img src='/{solution_image_path}' alt='Solution Image'/>"
                     f"<a class='download-btn' href='/{solution_image_path}' download>Download Image</a></div>")
        )
        db.session.add(chat_image)
    db.session.commit()
    session.pop("image_analysis", None)
    return redirect(url_for("index"))

@app.route("/followup_image", methods=["POST"])
def followup_image():
    followup_query = request.form.get("followup_query")
    context = request.form.get("context") or session.get("image_analysis", "")
    if followup_query and context:
        followup_response = generate_follow_up_response(context, followup_query)
        chat = ChatMessage(role="bot", message=f"Follow-up Response: {followup_response}")
        db.session.add(chat)
        db.session.commit()
    else:
        chat = ChatMessage(role="bot", message="Please upload an image and/or provide a valid follow-up query.")
        db.session.add(chat)
        db.session.commit()
    return redirect(url_for('index'))

@app.route("/video", methods=["POST"])
def video():
    if 'video' in request.files:
        video_file = request.files['video']
        if video_file:
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(video_file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            video_file.save(file_path)
            status_message = "Processing video analysis..."
            bot_status = ChatMessage(role="bot", message=status_message)
            db.session.add(bot_status)
            db.session.commit()
            video_result = process_video(file_path)
            video_result = generate_text_response(video_result)
            chat = ChatMessage(role="bot", message=f"Video result: {video_result}")
            db.session.add(chat)
            db.session.commit()
    return redirect(url_for('index'))

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    ChatMessage.query.delete()
    db.session.commit()
    return redirect(url_for('index'))

@app.route("/set_language", methods=["POST"])
def set_language():
    selected_language = request.form.get("language")
    if selected_language:
        session['default_language'] = selected_language
    return redirect(url_for("index"))

@app.route("/generate_audio", methods=["POST"])
def generate_audio():
    data = request.get_json()
    text = data.get("text", "")
    if text:
        lang = detect_language(text)
        audio_file = save_audio(text, lang)
        if audio_file:
            audio_url = "/" + audio_file  # relative URL
            print(f"Audio generated at: {audio_file} using language: {lang}")
            return jsonify({"audio_url": audio_url})
        else:
            return jsonify({"error": "Error generating audio"}), 500
    return jsonify({"error": "No text provided"}), 400

@app.route("/audio_input", methods=["POST"])
def audio_input():
    if 'audio' in request.files:
        status_message = "Analyzing audio file..."
        bot_status = ChatMessage(role="bot", message=status_message)
        db.session.add(bot_status)
        db.session.commit()

        audio_file = request.files['audio']
        if audio_file:
            result = process_audio(audio_file)
            if isinstance(result, tuple):
                recognized_text, gemini_response, audio_file_path = result
            else:
                recognized_text = result
                gemini_response = ""
                audio_file_path = None

            user_chat = ChatMessage(role="user", message="(Audio message received)")
            bot_chat = ChatMessage(role="bot", message=gemini_response)
            db.session.add_all([user_chat, bot_chat])
            if audio_file_path:
                audio_message = ChatMessage(
                    role="bot",
                    message=f"<audio controls><source src='/{audio_file_path}' type='audio/mpeg'></audio>"
                )
                db.session.add(audio_message)
            db.session.commit()
            return redirect(url_for("index"))
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
