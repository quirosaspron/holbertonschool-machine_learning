#!/usr/bin/env python3
"""Plots a histogram"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    "Plots student scores for a project"
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins, edgecolor='black')
    plt.xticks(np.arange(0, 110, 10))
    plt.show()
