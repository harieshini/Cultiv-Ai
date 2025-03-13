import speech_recognition as sr

def process_audio(file):
    """
    Process the audio file, converting speech to text.
    Provides clear feedback if recognition fails.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file) as source:
            audio_data = recognizer.record(source)
            # Use Google's speech recognition (or any lightweight model)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Audio not recognized. Please try again."
    except sr.RequestError as e:
        return f"Error during audio processing: {str(e)}"
