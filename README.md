# Word Similarity Search Flask Application

## Overview

This Flask web application allows users to search for a word and find the top 10 most similar words based on their dot product similarity using a pre-trained word embedding model.

## Features

- Input a query word through a simple web interface.
- Compute similarities between the query word and the entire corpus.
- Display the top 10 most similar words along with their similarity scores.

## Prerequisites

- Python 3.8+
- The following Python libraries:
  - Flask
  - Gensim
  - PyTorch
  - NumPy
  - tabulate

## File Structure


app/
│
├── app.py                 # Flask application code
├── templates/
│   └── index.html         # HTML template for the web interface
├── model_gensim.gensim    # Pre-trained word embedding model (to big to push to git)
└── README.md              # Project documentation

datasets/
│
├── word-test.v1.txt
└── wordsim_similarity_goldstandard.txt


