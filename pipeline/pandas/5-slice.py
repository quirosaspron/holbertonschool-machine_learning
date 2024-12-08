#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def slice(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    Extracts the columns High, Low, Close, and Volume_BTC
    Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame
    """
    return df.loc[::60, ['High', 'Low', 'Close', 'Volume_(BTC)']]
