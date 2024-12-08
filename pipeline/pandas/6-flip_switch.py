#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def flip_switch(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    Sort the data in reverse chronological order.
    Transpose the sorted dataframe.
    Return: the transformed pd.DataFrame
    """
    return df.sort_values(by='Timestamp', ascending=False).T
