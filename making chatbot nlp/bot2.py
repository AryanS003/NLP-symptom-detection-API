
import nltk
import warnings
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# Ensure NLTK packages are downloaded
nltk.download('punkt')  # For tokenization
nltk.download('wordnet')  # For lemmatization

# Load data files
with open('symptom.txt', 'r', errors='ignore') as f:
    symptoms_data = f.read().lower()

with open('pincodes.txt', 'r', errors='ignore') as m:
    pincodes_data = m.read().lower()

# Tokenization
symptom_sentences = nltk.sent_tokenize(symptoms_data)
symptom_words = nltk.word_tokenize(symptoms_data)
pincode_sentences = nltk.sent_tokenize(pincodes_data)
pincode_words = nltk.word_tokenize(pincodes_data)

lemmatizer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    """Lemmatizes tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]

# Remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    """Normalizes text by tokenizing, converting to lowercase, and removing punctuation."""
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Responses and keywords
GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = [
    "Hi! Are you experiencing any health issues? (Y/N)",
    "Hello! Do you have any health concerns? (Y/N)",
    "Hey there! Are you feeling unwell? (Y/N)"
]

POSITIVE_RESPONSES = ("yes", "y")
NEGATIVE_RESPONSES = ("no", "n")
FEVER_SYMPTOMS = ("i am suffering from fever", "i have fever", "fever")
FEVER_RESPONSE = "Which type of fever do you have? Please mention your symptoms so I can assist you further."

# Helper Functions
def greeting(sentence):
    """Returns a greeting response if a greeting keyword is detected."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def basic_response(sentence):
    """Returns response based on user confirmation about symptoms."""
    if sentence.lower() in POSITIVE_RESPONSES:
        return "Okay, please describe your symptoms."
    elif sentence.lower() in NEGATIVE_RESPONSES:
        return "Thank you for visiting. Take care!"

def fever_response(sentence):
    """Checks for fever-related keywords and provides a response."""
    if sentence.lower() in FEVER_SYMPTOMS:
        return FEVER_RESPONSE

def generate_response(user_response, sentences):
    """Generates a chatbot response using TF-IDF and cosine similarity."""
    response = ''
    sentences.append(user_response)
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    cosine_vals = cosine_similarity(tfidf[-1], tfidf)
    index = cosine_vals.argsort()[0][-2]
    flat = cosine_vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        response = "I'm sorry, I didn't understand that."
    else:
        response = sentences[index]
    
    sentences.pop()
    return response

# Main Chat Function
def chat(user_response):
    """Handles user interaction and routes responses."""
    user_response = user_response.lower()

    if user_response in ["bye", "exit", "quit"]:
        return "Bye! Take care."

    if user_response in ["thanks", "thank you"]:
        return "You're welcome!"

    if greeting(user_response):
        return greeting(user_response)

    if basic_response(user_response):
        return basic_response(user_response)

    if fever_response(user_response):
        return fever_response(user_response)

    if "module" in user_response:
        return generate_response(user_response, pincode_sentences)
    
    return generate_response(user_response, symptom_sentences)
