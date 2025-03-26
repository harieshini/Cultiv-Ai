import os  
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, session
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
from audio import process_audio
from video import process_video
from werkzeug.utils import secure_filename
from googletrans import Translator
from gtts import gTTS
from PIL import Image

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # e.g. "static/uploads"
app.secret_key = 'your_secret_key_here'  # Ensure you set a strong secret key

db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
translator = Translator()

def detect_language(text):
    if 'default_language' in session:
        return session['default_language']
    detected_lang = translator.detect(text).lang
    return detected_lang if detected_lang in ['ta', 'en'] else 'en'

def save_audio(text, lang):
    try:
        audio = gTTS(text=text, lang=lang)
        unique_name = f"response_{uuid.uuid4().hex}.mp3"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        audio.save(file_path)
        return f"{app.config['UPLOAD_FOLDER']}/{unique_name}"
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle text input
        if 'message' in request.form:
            user_message = request.form.get("message")
            if user_message:
                status_message = "Processing text input..."
                bot_status = ChatMessage(role="bot", message=status_message)
                db.session.add(bot_status)
                db.session.commit()
                
                time.sleep(1)  # Simulated processing delay

                lower_msg = user_message.lower()
                # Check for custom image generation commands
                if "generate an image" in lower_msg:
                    # Generate custom image and display with input text
                    custom_image_path = generate_custom_image(user_message)
                    if not custom_image_path.startswith("Error"):
                        bot_msg = ChatMessage(
                            role="bot", 
                            message=f"<div class='image-container'><div class='input-text'>Input: {user_message}</div><img src='/{custom_image_path}' alt='Custom Generated Image'/><a class='download-btn' href='/{custom_image_path}' download>Download Image</a></div>"
                        )
                    else:
                        bot_msg = ChatMessage(role="bot", message=custom_image_path)
                    db.session.add(bot_msg)
                    db.session.commit()
                    return redirect(url_for("index"))
                elif "text as image" in lower_msg:
                    image_from_text = generate_text_as_image(user_message)
                    if not image_from_text.startswith("Error"):
                        bot_msg = ChatMessage(
                            role="bot", 
                            message=f"<div class='image-container'><div class='input-text'>Input: {user_message}</div><img src='/{image_from_text}' alt='Text as Image'/><a class='download-btn' href='/{image_from_text}' download>Download Image</a></div>"
                        )
                    else:
                        bot_msg = ChatMessage(role="bot", message=image_from_text)
                    db.session.add(bot_msg)
                    db.session.commit()
                    return redirect(url_for("index"))
                
                # Process as a regular text question
                if not is_valid_question(user_message):
                    bot_response = "I specialize in Agriculture."
                else:
                    bot_response = generate_text_response(user_message)
                
                user_chat = ChatMessage(role="user", message=user_message)
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add_all([user_chat, bot_chat])
                db.session.commit()

                # For text/audio inputs, store context for solution generation
                session["show_generate_solution"] = True
                session["solution_context"] = user_message

                # Convert bot response to audio using detected language
                audio_file = save_audio(bot_response, detect_language(user_message))
                if audio_file:
                    audio_message = ChatMessage(
                        role="bot",
                        message=f"<audio controls><source src='/{audio_file}' type='audio/mpeg'></audio>"
                    )
                    db.session.add(audio_message)
                    db.session.commit()

                return redirect(url_for("index"))

        # Handle audio input
        elif 'audio' in request.files:
            status_message = "Analyzing audio file..."
            bot_status = ChatMessage(role="bot", message=status_message)
            db.session.add(bot_status)
            db.session.commit()

            audio_file = request.files['audio']
            if audio_file:
                result = process_audio(audio_file)
                if isinstance(result, tuple):
                    gemini_response, audio_file_path = result
                else:
                    gemini_response = result
                    audio_file_path = None

                gemini_response = generate_text_response(gemini_response)
                
                user_chat = ChatMessage(role="user", message="(Audio message received)")
                bot_chat = ChatMessage(role="bot", message=gemini_response)
                db.session.add_all([user_chat, bot_chat])
                session["show_generate_solution"] = True
                session["solution_context"] = gemini_response

                if audio_file_path:
                    audio_message = ChatMessage(
                        role="bot",
                        message=f"<audio controls><source src='/{audio_file_path}' type='audio/mpeg'></audio>"
                    )
                    db.session.add(audio_message)

                db.session.commit()
                return redirect(url_for("index"))
    
    chats = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    return render_template("index.html", chats=chats)

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            # Generate unique filename for the uploaded image
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Display the uploaded image along with the original filename
            relative_path = file_path.replace("\\", "/")
            user_image_msg = ChatMessage(role="user", message=f"<div class='image-container'><div class='input-text'>Uploaded: {file.filename}</div><img src='/{relative_path}' alt='Uploaded Image'/><a class='download-btn' href='/{relative_path}' download>Download Image</a></div>")
            db.session.add(user_image_msg)
            db.session.commit()

            # Analyze image and store the analysis in session
            analysis = analyze_and_store_image(file_path)
            session['image_analysis'] = analysis
            session["show_generate_solution"] = True

            status_msg = "Image uploaded and analyzed. Please use 'Generate Solution' to continue."
            bot_status = ChatMessage(role="bot", message=status_msg)
            db.session.add(bot_status)
            db.session.commit()
    return redirect(url_for('index'))

@app.route("/generate_solution", methods=["POST"])
def generate_solution():
    # Use image analysis if available; otherwise, fall back to text/audio context
    analysis = session.get("image_analysis") or session.get("solution_context")
    if not analysis:
        bot_msg = ChatMessage(role="bot", message="No context available. Please upload an image or submit a query first.")
        db.session.add(bot_msg)
        db.session.commit()
        return redirect(url_for("index"))
    
    solution_text, solution_image_path = generate_solution_response(analysis)
    chat_text = ChatMessage(role="bot", message=f"Solution: {solution_text}")
    db.session.add(chat_text)
    if solution_image_path and not solution_image_path.startswith("Error"):
        # Crop the generated solution image to 1024x955 (center crop)
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
            message=f"<div class='image-container'><div class='input-text'>Context: {analysis}</div><img src='/{solution_image_path}' alt='Solution Image'/><a class='download-btn' href='/{solution_image_path}' download>Download Image</a></div>"
        )
        db.session.add(chat_image)
    db.session.commit()
    # Clear the solution generation flags after use
    session.pop("show_generate_solution", None)
    session.pop("solution_context", None)
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

@app.route("/clear", methods=["POST"])
def clear():
    ChatMessage.query.delete()
    db.session.commit()
    return redirect(url_for('index'))

@app.route("/set_language", methods=["POST"])
def set_language():
    selected_language = request.form.get("language")
    if selected_language:
        session['default_language'] = selected_language
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
