import numpy as np
import pandas as pd
import math
import csv
import time # Import the time module
from typeguard import typechecked

@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list[list[float]]: A list of lists, where each inner list represents a row
                           of the dataset with float values.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: If you are stuck on this, I recommend looking through the python csv library.
    """
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        data = [[float(value) for value in row] for row in csv_reader]
    return data.tolist()

@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    """
    Loads a dataset from a CSV file into a NumPy array.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array representing the dataset.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Numpy has a very useful csv file reader called genfromtxt.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    """
    Normalizes the input array (list of lists) using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (list[list[float]]): The input dataset as a list of lists of floats.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Probably the most complicated function to write because you cant use numpy.
    I would spend some time on this to make sure that all the equations for metrics are correct.
    """
    if not arr or not arr[0]:
        return 0  # empty input

    col_count = len(arr[0])

    # first column is just an id, don’t touch it
    skip = 1

    # store stats for each column
    avg_list, min_list, max_list, std_list = [], [], [], []
    for col in range(skip, col_count):
        col_vals = [row[col] for row in arr]
        avg = sum(col_vals) / len(col_vals)
        var = sum((x - avg) ** 2 for x in col_vals) / len(col_vals)
        stdev = math.sqrt(var)
        avg_list.append(avg)
        min_list.append(min(col_vals))
        max_list.append(max(col_vals))
        std_list.append(stdev)

    # throw out outliers
    kept_rows = []
    for row in arr:
        ok = True
        for i, col in enumerate(range(skip, col_count)):
            if std_list[i] > 0 and abs(row[col] - avg_list[i]) > 2 * std_list[i]:
                ok = False
                break
        if ok:
            kept_rows.append(row)

    # normalize
    final_rows = []
    for row in kept_rows:
        new_row = [row[0]]  # keep id
        for i, col in enumerate(range(skip, col_count)):
            if max_list[i] == min_list[i]:
                new_row.append(0.0)
            else:
                new_row.append((row[col] - min_list[i]) / (max_list[i] - min_list[i]))
        final_rows.append(new_row)

    # write file if needed
    if out_file:
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(final_rows)

    return len(final_rows)
    
@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    """
    Normalizes the input NumPy array using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (np.ndarray): The input dataset as a NumPy array.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: The same function as normalize_array but using numpy to calculate the metrics.
    This function should be almost copy and paste with numpy functions.
    """
    if arr.size == 0:
        return 0  # no data

    cols = arr.shape[1]
    start = 1  # skip first column (id)

    # stats for each column
    avg_list, min_list, max_list, std_list = [], [], [], []
    for col in range(start, cols):
        col_data = arr[:, col]
        avg = np.mean(col_data)
        stdev = np.std(col_data)
        min_val = np.min(col_data)
        max_val = np.max(col_data)

        avg_list.append(avg)
        min_list.append(min_val)
        max_list.append(max_val)
        std_list.append(stdev)

    # remove outliers
    good_rows = []
    for row in arr:
        ok = True
        for i, col in enumerate(range(start, cols)):
            if std_list[i] > 0 and abs(row[col] - avg_list[i]) > 2 * std_list[i]:
                ok = False
                break
        if ok:
            good_rows.append(row)

    if not good_rows:
        return 0

    good_arr = np.array(good_rows)

    # normalize values
    norm_rows = []
    for row in good_arr:
        new_row = [row[0]]  # keep id
        for i, col in enumerate(range(start, cols)):
            if max_list[i] == min_list[i]:
                new_row.append(0.0)
            else:
                new_row.append((row[col] - min_list[i]) / (max_list[i] - min_list[i]))
        norm_rows.append(new_row)

    norm_arr = np.array(norm_rows)

    # save file
    if out_file:
        np.savetxt(out_file, norm_arr, delimiter=",", fmt="%.6f")

    return norm_arr.shape[0]