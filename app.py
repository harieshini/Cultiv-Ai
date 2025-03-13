import os
from flask import Flask, render_template, request, redirect, url_for
from config import UPLOAD_FOLDER, DATABASE_URI
from models import db, ChatMessage
from text import is_valid_question, generate_text_response
from image import analyze_image_with_gemini
from audio import process_audio
from video import process_video
from werkzeug.utils import secure_filename
from googletrans import Translator

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
                # Save chat messages
                user_chat = ChatMessage(role="user", message=user_message)
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add(user_chat)
                db.session.add(bot_chat)
                db.session.commit()
                return redirect(url_for("index"))
        # Handle audio input
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file:
                text = process_audio(audio_file)
                user_chat = ChatMessage(role="user", message=text)
                db.session.add(user_chat)
                if is_valid_question(text):
                    bot_response = generate_text_response(text)
                else:
                    bot_response = "I specialize in Agriculture."
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add(bot_chat)
                db.session.commit()
                return redirect(url_for("index"))
    # GET request: show chat history
    chats = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    return render_template("index.html", chats=chats)

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles image uploads. Uses the Gemini Vision API to analyze images.
    """
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_analysis = analyze_image_with_gemini(file_path)
            message = f"Image analysis result: {image_analysis}"
            chat = ChatMessage(role="bot", message=message)
            db.session.add(chat)
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

if __name__ == "__main__":
    app.run(debug=True)
