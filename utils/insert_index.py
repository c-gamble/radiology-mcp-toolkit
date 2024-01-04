def find_insert_index(arr: list, ele) -> int:
    for i in range(len(arr)):
        if ele < arr[i]: # use < if we want to place the test sample below all equal calibration samples
            return i
    return len(arr)