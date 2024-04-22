#!/usr/bin/env python3
"""Plots a histogram"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    "Plots student scores for a project"
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    # Plot histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Set labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Show plot
    plt.show()
