import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import numpy as np

# Initialize the model globally if it's lightweight; otherwise, consider loading per request or using other strategies.
model = Doc2Vec.load('C:/Users/Ilay/Desktop/django_proj_2/doc2vec_model')

def preprocess_text(text):
    return word_tokenize(text.lower())
        # needs to add more constrains to improve

def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


def infer_embedding_preprocessed(model, text):
    processed_text = preprocess_text(text)
    embedding = model.infer_vector(processed_text)
    return normalize_vector(embedding)


def find_similar_articles(new_text_embedding, articles, top_n=10):
    similarities = []
    for article in articles:
        article_embedding = article['embedding']
        sim_score = 1 - cosine(new_text_embedding, article_embedding)
        similarities.append((article, sim_score))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top_n articles, excluding the similarity score
    top_articles = [tup[0] for tup in similarities[:top_n]]

    return top_articles

# Load your dataset and create embeddings on app initialization
def load_articles_and_embeddings():
    file_path = 'C:/Users/Ilay/Desktop/django_proj_2/Articles.csv'  # Update this path
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # Try 'utf-8' encoding first
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')  # Fallback to 'latin1' encoding if 'utf-8' fails

    # Preprocess and create embeddings
    df['tokenized_text'] = df['Article'].apply(preprocess_text)
    df['embedding'] = df['tokenized_text'].apply(lambda x: normalize_vector(model.infer_vector(x)))
    articles = df.to_dict(orient='records')

    # Add a key for article search
    for i in range(len(articles)):
        articles[i]['key'] = i

    return articles


# This will hold all articles and their embeddings in memory
articles_data = load_articles_and_embeddings()
