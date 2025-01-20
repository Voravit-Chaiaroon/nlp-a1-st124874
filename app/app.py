from flask import Flask, request, render_template
import torch
from gensim.models import KeyedVectors
import numpy as np

# Load your pre-trained model
model_gensim = torch.load('model_gensim.gensim')

# Initialize Flask app
app = Flask(__name__)

# Function to compute embeddings for a word
def get_word_embedding(word):
    try:
        return model_gensim[word]
    except KeyError:
        return None

# Compute the dot product between the query embedding and corpus embeddings
def compute_similarity(query):
    query_embedding = get_word_embedding(query)
    if query_embedding is None:
        return []

    similarities = []
    for word in model_gensim.index_to_key:
        word_embedding = get_word_embedding(word)
        dot_product = np.dot(query_embedding, word_embedding)
        similarities.append((word, dot_product))

    # Sort by similarity and return top 10 results
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:10]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = compute_similarity(query)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
