import google.generativeai as genai
from config import GEMINI_API_KEY
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from googletrans import Translator
import matplotlib.pyplot as plt
from datasets import load_dataset
import matplotlib.animation as animation
import numpy as np
import uuid

# Create an instance of the Translator.
translator = Translator()

# Load the agriculture QA dataset from Hugging Face.
dataset = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")

# Extract text data: Combine 'question' and 'answer' for a richer context.
corpus = []
for record in dataset:
    # Make sure the keys exist in the record.
    if "question" in record and "answer" in record:
        # Combine question and answer to capture broader agricultural context.
        corpus.append(record["question"] + " " + record["answer"])

# Optionally, sample a subset for efficiency if the dataset is very large.
# Here, we use the first 1000 entries if there are more than 1000.
sampled_corpus = corpus[:1000] if len(corpus) > 1000 else corpus

# Combine the sampled text into a single large reference string.
combined_text = " ".join(sampled_corpus)

# Fit the TF‑IDF vectorizer on the combined agricultural text.
vectorizer = TfidfVectorizer().fit([combined_text])

def is_valid_question(question, threshold=0.1):
    """
    Determines if the question is agriculture-related using TF‑IDF vectorization 
    and cosine similarity.
    
    ML Algorithm: TF‑IDF Vectorization with Cosine Similarity.
    """
    question_vec = vectorizer.transform([question.lower()])
    # Use combined_text instead of undefined allowed_text.
    corpus_vec = vectorizer.transform([combined_text])
    sim = cosine_similarity(question_vec, corpus_vec)[0][0]
    return sim > threshold

def translate_to_english(text):
    """
    Detects the language of the input text and translates it to English if necessary.
    Returns a tuple of (translated_text, original_language).
    """
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translation = translator.translate(text, dest='en')
            return translation.text, detected_lang
    except Exception:
        # If language detection or translation fails, assume text is English.
        pass
    return text, 'en'

def translate_from_english(text, target_lang):
    """
    Translates text from English to the target language.
    """
    try:
        if target_lang != 'en':
            translation = translator.translate(text, dest=target_lang)
            return translation.text
    except Exception:
        pass
    return text

def generate_text_response(prompt):
    """
    Generates a text response using the Gemini text model.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error in text generation: {str(e)}"

def process_query(query):
    """
    Processes the user's query with multi-language support.
    
    Steps:
    1. Detect and translate the query to English if necessary.
    2. Validate if the query is agriculture-related.
    3. Generate a text response using the Gemini text model.
    4. Translate the response back to the original language if needed.
    """
    # Translate query to English if needed.
    translated_query, original_lang = translate_to_english(query)
    
    # Validate if the query is agriculture-related.
    if not is_valid_question(translated_query):
        return "The query does not seem to be related to agriculture. Please ask an agriculture-related question."
    
    # Generate the text response.
    response = generate_text_response(translated_query)
    
    # Translate response back to original language if necessary.
    final_response = translate_from_english(response, original_lang)
    return final_response

def generate_animation_response(query):
    """
    Generates an animation based on the query.
    The animation is saved as a GIF file and the function returns the file path.
    
    This demonstration creates a simple animated sine wave using matplotlib.
    In a real-world agricultural chatbot, you might replace this with animation logic
    that illustrates soil testing, pest detection, or irrigation practices.
    """
    # Generate a unique filename for the animation.
    filename = f"animation_{uuid.uuid4().hex}.gif"
    
    # Create a figure and axis.
    fig, ax = plt.subplots()
    
    # Create an example animation: an animated sine wave.
    x = np.linspace(0, 2 * np.pi, 128)
    line, = ax.plot(x, np.sin(x))
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Animation for query: {query}")
    
    def update(frame):
        line.set_ydata(np.sin(x + frame / 10.0))
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=100, blit=True)
    
    # Save the animation as a GIF file using PillowWriter.
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=10)
    ani.save(filename, writer=writer)
    plt.close(fig)
    return filename
