#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def high(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    Sort it by the High price in descending order.
    Returns: the sorted pd.DataFrame
    """
    return df.sort_values(by='High', ascending=False)
