import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import warnings

# Import Spacy for NLP tasks
import spacy
nlp = spacy.load("en_core_web_sm")

# To hide warnings, especially those regarding future changes in libraries
warnings.filterwarnings('ignore')

# Setup Matplotlib for inline display
%matplotlib inline

def generate_word_cloud_for_org_code(df, org_code):
    """
    Generate a word cloud for a specific organization code.
    """
    filtered_df = df[df['org_code'] == org_code]
    text = ' '.join(filtered_df['cleaned_lemma'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def get_popular_terms_by_org_code(df, org_code, n_terms=10):
    """
    Find popular terms for a specific organization code.
    """
    filtered_df = df[df['org_code'] == org_code]
    words = ' '.join(filtered_df['cleaned_lemma'].tolist()).split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(n_terms)
    return pd.DataFrame(most_common_words, columns=['Term', 'Frequency'])

def get_popular_terms_for_all_org_codes(df, n_terms=10):
    """
    Aggregate popular terms across all organization codes.
    """
    popular_terms_df = pd.concat([get_popular_terms_by_org_code(df, org_code, n_terms) 
                                  for org_code in df['org_code'].unique()], ignore_index=True)
    popular_terms_df['org_code'] = org_code  # Adding org_code column might need adjustment based on your logic
    return popular_terms_df[['org_code', 'Term', 'Frequency']]

def get_top_n_gram(text, ngram=1, top=None):
    """
    Extract top N-grams from text.
    """
    vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]

def perform_nmf_topic_modeling(text_data, num_topics=145, num_words=15):
    """
    Perform NMF topic modeling.
    """
    # Assuming 'dtm' is your document-term matrix and 'tfidf' is your TfidfVectorizer fitted instance
    nmf_model = NMF(n_components=num_topics, random_state=40)
    W1 = nmf_model.fit_transform(text_data)  # You need to define 'text_data'
    H1 = nmf_model.components_
    vocab = np.array(tfidf.get_feature_names_out())  # You need to define 'tfidf'
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
    topic_words = [top_words(t) for t in H1]
    return [' '.join(t) for t in topic_words]

