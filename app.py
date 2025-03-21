import os
from flask import Flask, render_template, request, redirect, url_for, session
from config import UPLOAD_FOLDER, DATABASE_URI
from models import db, ChatMessage
from text import is_valid_question, generate_text_response
from image import analyze_and_generate_solution, generate_follow_up_response
from audio import process_audio
from video import process_video
from werkzeug.utils import secure_filename
from googletrans import Translator
from gtts import gTTS

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'  # Ensure you set a strong secret key

db.init_app(app)
with app.app_context():
    db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Global translator instance for language detection
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

                # Convert bot response to audio using the detected language
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
                result = process_audio(audio_file)
                if isinstance(result, tuple):
                    gemini_response, audio_file_path = result
                else:
                    gemini_response = result
                    audio_file_path = None

                user_chat = ChatMessage(role="user", message="(audio message received)")
                db.session.add(user_chat)
                bot_chat = ChatMessage(role="bot", message=gemini_response)
                db.session.add(bot_chat)

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
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            solution_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"solution_{filename}")

            solution_text, generated_image_path = analyze_and_generate_solution(
                file_path, solution_image_output=solution_image_path
            )

            bot_text_message = ChatMessage(role="bot", message=f"Solution: {solution_text}")
            db.session.add(bot_text_message)

            if generated_image_path and not generated_image_path.startswith("Error"):
                bot_image_message = ChatMessage(
                    role="bot",
                    message=f"<img src='/{generated_image_path}' alt='Solution Image' />"
                )
                db.session.add(bot_image_message)

            db.session.commit()
    return redirect(url_for('index'))

@app.route("/followup_image", methods=["POST"])
def followup_image():
    followup_query = request.form.get("followup_query")
    context = request.form.get("context")
    if followup_query and context:
        followup_response = generate_follow_up_response(context, followup_query)
        chat = ChatMessage(role="bot", message=f"Follow-up Response: {followup_response}")
        db.session.add(chat)
        db.session.commit()
    return redirect(url_for('index'))

@app.route("/video", methods=["POST"])
def video():
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
    ChatMessage.query.delete()
    db.session.commit()
    return redirect(url_for('index'))

@app.route("/set_language", methods=["POST"])
def set_language():
    selected_language = request.form.get("language")
    if selected_language:
        session['default_language'] = selected_language
    return redirect(url_for("index"))

def detect_language(text):
    # Use the default language from session if it exists.
    if 'default_language' in session:
        return session['default_language']
    detected_lang = translator.detect(text).lang
    return 'ta' if detected_lang == 'ta' else 'en'

def save_audio(text, lang):
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
