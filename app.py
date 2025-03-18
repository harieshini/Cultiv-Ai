import os
from flask import Flask, render_template, request, redirect, url_for
from config import UPLOAD_FOLDER, DATABASE_URI
from models import db, ChatMessage
from text import is_valid_question, generate_text_response
from image import analyze_and_generate_solution
from audio import process_audio
from video import process_video
from werkzeug.utils import secure_filename
from googletrans import Translator
from gtts import gTTS

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global translator instance for language detection (future use)
translator = Translator()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle text input
        if 'message' in request.form:
            user_message = request.form.get("message")
            if user_message:
                if not is_valid_question(user_message):
                    bot_response = "I specialize in Agriculture."
                else:
                    bot_response = generate_text_response(user_message)

                # Save user message and bot response to database
                user_chat = ChatMessage(role="user", message=user_message)
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add(user_chat)
                db.session.add(bot_chat)
                db.session.commit()

                # Convert bot response to audio
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
            audio_file = request.files['audio']
            if audio_file:
                # Process audio and convert to text
                text = process_audio(audio_file)
                user_chat = ChatMessage(role="user", message=text)
                db.session.add(user_chat)

                # Generate response based on audio text
                if is_valid_question(text):
                    bot_response = generate_text_response(text)
                else:
                    bot_response = "I specialize in Agriculture."

                # Save bot response to database
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add(bot_chat)
                db.session.commit()

                # Convert bot response to audio
                audio_file = save_audio(bot_response, detect_language(text))
                if audio_file:
                    audio_message = ChatMessage(
                        role="bot",
                        message=f"<audio controls><source src='/{audio_file}' type='audio/mpeg'></audio>"
                    )
                    db.session.add(audio_message)
                    db.session.commit()

                return redirect(url_for("index"))

    # GET request: Show chat history
    chats = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    return render_template("index.html", chats=chats)

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles image uploads: analyzes the image, generates a solution text and a solution image,
    then stores the solution as bot messages in the chat.
    """
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Output path for the solution image (prepend 'solution_' to the filename)
            solution_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"solution_{filename}")

            solution_text, generated_image_path = analyze_and_generate_solution(
                file_path, solution_image_output=solution_image_path
            )

            # Save solution text to chat
            bot_text_message = ChatMessage(role="bot", message=f"Solution: {solution_text}")
            db.session.add(bot_text_message)

            # Save generated image to chat
            if generated_image_path and not generated_image_path.startswith("Error"):
                bot_image_message = ChatMessage(
                    role="bot",
                    message=f"<img src='/{generated_image_path}' alt='Solution Image' />"
                )
                db.session.add(bot_image_message)

            db.session.commit()

    return redirect(url_for('index'))

@app.route("/video", methods=["POST"])
def video():
    """
    Handles video uploads and processes them.
    """
    if 'video' in request.files:
        video_file = request.files['video']
        if video_file:
            filename = secure_filename(video_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(file_path)

            video_result = process_video(file_path)
            chat = ChatMessage(role="bot", message=f"Video result: {video_result}")
            db.session.add(chat)
            db.session.commit()

    return redirect(url_for('index'))

@app.route("/clear", methods=["POST"])
def clear():
    """
    Clears all chat messages from the database.
    """
    ChatMessage.query.delete()
    db.session.commit()
    return redirect(url_for('index'))

def detect_language(text):
    """
    Detect language using googletrans.
    """
    translator = Translator()
    detected_lang = translator.detect(text).lang
    return 'ta' if detected_lang == 'ta' else 'en'

def save_audio(text, lang):
    """
    Converts text to speech and saves it as an audio file.
    """
    try:
        audio = gTTS(text=text, lang=lang)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'response.mp3')
        audio.save(file_path)
        return f"static/uploads/response.mp3"
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

if __name__ == "__main__":
    app.run(debug=True)
