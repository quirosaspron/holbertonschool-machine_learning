#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""


def concat(df1, df2):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    - Index both dataframes on their Timestamp columns.
    - Include all timestamps from df2 (bitstamp) up to
      and including timestamp 1417411920.
    - Concatenate the selected rows from df2 to the top
      of df1 (coinbase).
    - Add keys to the concatenated data, labeling the rows
      from df2 as bitstamp and the rows from df1 as coinbase.
    - Return the concatenated pd.DataFrame.
    """
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
