#!/usr/bin/env python3
"""Simple loop"""
while True:
    question = input("Q: ").lower()
    if question in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    print("A: ")
