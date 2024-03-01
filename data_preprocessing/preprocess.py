import pandas as pd
import re
from nltk.corpus import stopwords
import spacy
import nltk

# Only need to import pandas once and you don't use TextBlob, word_tokenize, or the NLTK WordNetLemmatizer in this snippet
nltk.download('punkt')  # Necessary if you're using NLTK tokenization elsewhere
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your DataFrame
df = pd.read_csv('df.csv')
df.dropna(axis=0, inplace=True)

def remove_masked_terms(text):
    # Function to remove special characters, digits, and masked terms
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    cleaned_text = re.sub(r'\bx[^\s]*\b', '', cleaned_text)  # Remove words starting with 'x' which might not be needed after the first regex
    cleaned_text = re.sub(r'\b[Xx]+\b', '', cleaned_text)  # Remove placeholders like 'X', 'XX', etc.
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()  # Remove extra spaces and strip the text
    return cleaned_text

def remove_stop_words(text):
    # Function to remove stopwords using spaCy
    doc = nlp(text)
    filtered_sentence = [token.text for token in doc if not token.is_stop]
    return ' '.join(filtered_sentence)

# Assuming 'cleaned_grievance' is a column in your DataFrame that needs preprocessing
df['text_topic_modeling'] = df['cleaned_grievance'].apply(remove_masked_terms)  # Apply the function to remove masked terms first
df['text_topic_modeling'] = df['text_topic_modeling'].apply(remove_stop_words)  # Then remove stop words

def lemmatizer(text):
    # Function to lemmatize the text using spaCy
    doc = nlp(text)
    sent = [token.lemma_ for token in doc if token.text not in nlp.Defaults.stop_words]  # Use spaCy's stop words list directly
    return ' '.join(sent)

df['cleaned_lemma'] = df['text_topic_modeling'].apply(lemmatizer)  # Apply lemmatization on the text after stop words removal
