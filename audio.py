import os
import tempfile
import asyncio
from pydub import AudioSegment
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai
import nest_asyncio

# Allow nested async calls
nest_asyncio.apply()

# Load environment variables and configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set ffmpeg path if specified via environment variable.
ffmpeg_path = os.environ.get("FFMPEG_PATH")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path

# Initialize the Gemini text model
gemini_text = genai.GenerativeModel('gemini-1.5-pro')

def convert_audio_to_text(audio_file):
    """
    Convert the provided audio file (any common format) to text using speech_recognition.
    """
    recognizer = sr.Recognizer()
    try:
        # Save uploaded/dynamic audio to a temporary file
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        audio_file.save(temp_input)
        temp_input.close()

        # Create a temporary file for the WAV conversion
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        # Convert to WAV format using pydub
        audio = AudioSegment.from_file(temp_input.name)
        audio.export(temp_wav.name, format="wav")
        # IMPORTANT: Close the temp_wav file handle so that the file is not locked.
        temp_wav.close()

        # Use speech recognition to convert audio to text
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data, language="en-IN")
            print(f"Recognized Text: {recognized_text}")

        # Clean up temporary files
        os.remove(temp_input.name)
        os.remove(temp_wav.name)

        return recognized_text

    except sr.UnknownValueError:
        return "Audio not recognized. Please try again."
    except sr.RequestError as e:
        return f"Error during audio processing: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_gemini_response(input_text):
    """
    Generate a response using the Gemini text model.
    The prompt is tailored to agricultural advice.
    """
    try:
        prompt = (
            f"Based on the following audio input, provide a detailed agricultural solution or recommendation:\n\n"
            f"{input_text}\n\nResponse:"
        )

        async def async_generate():
            response = await gemini_text.generate_content_async(prompt)
            return response.text

        response_text = asyncio.run(async_generate())
        return response_text

    except Exception as e:
        return f"Error in generating Gemini response: {str(e)}"

def convert_text_to_audio(text, output_filename="response_audio.mp3"):
    """
    Convert the provided text to speech using gTTS and save as an MP3 file.
    """
    try:
        tts = gTTS(text=text, lang="en")
        # Save audio file to the 'static/uploads' directory (adjust as needed)
        output_path = os.path.join("static", "uploads", output_filename)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error converting text to audio: {e}")
        return None

def process_audio(file):
    """
    Process the audio file:
    1. Convert the audio to text.
    2. Generate an agricultural solution/response using Gemini.
    3. Convert the generated text to audio.
    Returns a tuple: (recognized_text, gemini_response, audio_file_path).
    """
    recognized_text = convert_audio_to_text(file)
    # Check if we got a valid recognized text.
    if recognized_text.startswith("Error") or recognized_text in ["Audio not recognized. Please try again."]:
        return recognized_text

    gemini_response = generate_gemini_response(recognized_text)
    audio_file_path = convert_text_to_audio(gemini_response)
    return (recognized_text, gemini_response, audio_file_path)
