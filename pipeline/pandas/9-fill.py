#!/usr/bin/env python3
"""
takes a pd.DataFrame as input and performs some actions
"""
import pandas as pd


def fill(df):
    """
    df: pd.DataFrame containing columns named High and Close
    We will:
    - Remove the Weighted_Price column
    - Fill missing values in the Close
      column with the previous row’s value
    - Fill missing values in the High, Low,
      and Open columns with the corresponding
      Close value in the same row.
    - Sets missing values in Volume_(BTC)
      and Volume_(Currency) to 0.
    - Return: the modified pd.DataFrame.
    """
    # Remove column Weighted_Price
    df = df.drop(columns=['Weighted_Price'])
    # Fill missing values
    df['Close'].fillna(method='pad', inplace=True)
    df['High'].fillna(df.Close, inplace=True)
    df['Low'].fillna(df.Close, inplace=True)
    df['Open'].fillna(df.Close, inplace=True)
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    return df
