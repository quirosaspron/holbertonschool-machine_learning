#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def prune(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    Remove any entries where Close has NaN values.
    Return: the modified pd.DataFrame.
    """
    return df.dropna(subset=['Close'])
