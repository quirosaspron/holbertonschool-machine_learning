#!/usr/bin/env python3
"""Answers questions from multiple reference texts"""
qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """Answers questions from a corpus of texts"""
    while True:
        question = input("Q: ").lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        ref = semantic_search(coprus_path, question)
        answer = qa(question, ref)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
