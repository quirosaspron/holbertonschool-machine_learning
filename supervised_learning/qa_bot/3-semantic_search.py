#!/usr/bin/env python3
"""Performs semantic search"""
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents stored in a directory.

    Parameters:
    - corpus_path (str): Path to the directory containing reference documents.
    - sentence (str): Sentence from which to perform semantic search.

    Returns:
    - The reference text of the document most similar to the sentence.
    """
    # Load the Universal Sentence Encoder model
    embed = hub.load("https://tfhub.dev\
/google/universal-sentence-encoder-large/5")

    # Specify the list of file extensions considered as text files
    valid_extensions = {".txt", ".md", ".csv", ".json"}

    # Read all documents from the specified directory
    documents = []
    for filename in os.listdir(corpus_path):
        # Build the complete file path
        file_path = os.path.join(corpus_path, filename)

        # Check if the file has a valid text-based extension and is a file
        valid_path = os.path.isfile(file_path)
        file_ext = os.path.splitext(filename)[1]
        if valid_path and file_ext in valid_extensions:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read and store file content
                documents.append(file.read().strip())

    # Check if any documents were read
    if not documents:
        raise None

    # Step 1: Embed the documents and the query sentence
    doc_embeddings = embed(documents)  # Encode the documents
    query_embedding = embed([sentence])  # Encode the query sentence

    # Step 2: Compute the similarity scores using inner product
    similarity_scores = np.inner(query_embedding, doc_embeddings)[0]

    # Step 3: Find the index of the most similar document
    most_similar_index = np.argmax(similarity_scores)

    # Step 4: Return the document with the highest similarity score
    return documents[most_similar_index]
