#!/usr/bin/env python3
"""
Loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    filename is the file to load from
    delimiter is the column separator
    Returns: the loaded pd.DataFrame
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
