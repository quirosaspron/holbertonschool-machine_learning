#!/usr/bin/env python3
"""
takes two DataFrames as input and performs some actions
"""
import pandas as pd

def hierarchy(df1, df2):
    """
    We will:
    - Rearrange the MultiIndex so that
      Timestamp is the first level.
    - Concatenate the bitstamp and coinbase
      tables from timestamps 1417411980 to
      1417417980, inclusive.
    - Add keys to the data, labeling rows
      from df2 as bitstamp and rows from df1 as coinbase.
    - Ensure the data is displayed in chronological order.
    - Return: the concatenated pd.DataFrame
    """
    df1 = df1.loc[
        (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
    df2 = df2.loc[
        (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.reorder_levels([1, 0], axis=0)
    df = df.sort_index()
    return df
