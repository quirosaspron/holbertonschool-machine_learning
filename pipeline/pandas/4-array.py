#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def array(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    - select the last 10 rows of the High and Close columns
    - Convert these selected values into a numpy.ndarray
    Returns: the modified numpy array
    """
    return df.loc[:, ['High', 'Close']].tail(10).to_numpy()
