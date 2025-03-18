import os
import tempfile
from pydub import AudioSegment
import speech_recognition as sr
from text import generate_text_response  # Import from text.py

def process_audio(file):
    """
    Process the audio file, convert speech to text, generate solution using Gemini API.
    """
    recognizer = sr.Recognizer()

    try:
        # Save the file to a temporary location
        temp_input = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_input)
        temp_input.close()

        # Convert WebM/OGG to WAV using pydub
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio = AudioSegment.from_file(temp_input.name)
        audio.export(temp_wav.name, format="wav")

        # Process with speech recognition
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-IN")
            print(f"Recognized Text: {text}")

        # Clean up temporary files
        os.remove(temp_input.name)
        os.remove(temp_wav.name)

        if text:
            # Generate solution using Gemini API
            solution = generate_text_response(text)  # Call the Gemini API
            print(f"Gemini Solution: {solution}")
            return solution

        else:
            return "Audio recognized, but no meaningful text extracted."

    except sr.UnknownValueError:
        return "Audio not recognized. Please try again."
    except sr.RequestError as e:
        return f"Error during audio processing: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
