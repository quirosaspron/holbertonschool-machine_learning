#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def index(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    - Set the Timestamp column as the index of the dataframe
    - Returns: the modified pd.DataFrame
    """
    return df.set_index('Timestamp')
