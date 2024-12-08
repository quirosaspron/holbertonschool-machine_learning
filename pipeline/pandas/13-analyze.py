#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""


def analyze(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    - Compute descriptive statistics for all
      columns except the Timestamp column.
    - Return: a new pd.DataFrame containing these statistics
    """
    return df.drop(columns=['Timestamp']).describe()
