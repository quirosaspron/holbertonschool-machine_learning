#!/usr/bin/env python3
""" Concatenates two arrays """


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    new_arr = arr1
    for element in arr2:
        new_arr.append(element)
    return new_arr
