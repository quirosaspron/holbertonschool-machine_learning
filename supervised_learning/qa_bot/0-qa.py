#!/usr/bin/env python3
"""Question answering bot"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Parameters:
        - question (str): The question to answer.
        - reference (str): The reference document
                           from which to find the answer.

    Returns:
        A string containing the answer, or None if no answer is found.
    """
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")

    # Load BERT model for question answering from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the question and reference text
    quest_tokens = tokenizer.tokenize(question)
    refer_tokens = tokenizer.tokenize(reference)

    # Combine tokens and add special tokens [CLS] and [SEP]
    tokens = ['[CLS]'] + quest_tokens + ['[SEP]'] + refer_tokens + ['[SEP]']

    # Convert tokens to input IDs
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Create input masks (1 for real tokens, 0 for padding)
    input_mask = [1] * len(input_word_ids)

    # Create segment IDs (0 for question, 1 for reference)
    input_type_ids = [0] * (1 + len(quest_tokens) + 1) + \
        [1] * (len(refer_tokens) + 1)

    # Convert lists to tensors and add batch dimension
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    # Run the model
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # outputs[0][0][0] is the ignored '[CLS]' token logit
    # Ignores a token but adds +1 to preserve the same index
    # Get start and end positions for the answer
    start = tf.argmax(outputs[0][0][1:]) + 1
    end = tf.argmax(outputs[1][0][1:]) + 1

    # Extract and convert answer tokens to string
    answer_tokens = tokens[start: end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if not answer.strip():
        return None
    return answer
