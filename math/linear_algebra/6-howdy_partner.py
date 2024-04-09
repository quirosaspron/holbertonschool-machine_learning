#!/usr/bin/env python3
def cat_arrays(arr1, arr2):
    new_arr = arr1
    for element in arr2:
        new_arr.append(element)
    return new_arr
