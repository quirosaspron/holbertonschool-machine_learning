#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def rename(df):
    """
    df: pd.DataFrame containing a column named Timestamp
    We will:
    - Rename the Timestamp column to Datetime.
    - Convert the timestamp values to datatime values
    - Display only the Datetime and Close column
    Returns: the modified pd.DataFrame
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df.loc[:, ['Datetime', 'Close']]
    return df
