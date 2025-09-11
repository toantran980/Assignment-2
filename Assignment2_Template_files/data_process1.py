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
    return data

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
    return data.tolist()

@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    if not arr or not arr[0]:
        return 0 #empty input
    
    
    num_cols =len(arr[0]) 

    #Assume first column = idx (do nothing, copy it over)
    start_col = 1

    #Calculate mean, min, max, std for each feature (exclude idx)
    means, mins, maxs, stds = [], [], [], []
    for c in range(start_col, num_cols):
        values = [row[c] for row in arr]
        mean = sum(values) / len(values)
        variance = sum((x-mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        means.append(min(values))
        maxs.append(max(values))
        stds.append(std)

    #filter out rows with any outliers
    filterd = []
    for row in arr:
        keep = True
        for i, c in enumerate(range(start_col, num_cols)):
            if stds[i] > 0 and abs(row[c] - means[i]) > * stds[i]:
                keep = False
                break
            if keep:
                filtered.append(row)
    
    #normalize values
    normalized = []
    for row in filterd:
        new_row = [row[0]] #keep idx unchanged
        for i, c in enumerate(range(start_col,num_cols)):
            if maxs[i] == mins[i]:
                new_row.append(0.0)
            else:
                new_row.append((row[c] - means[i]) / (maxs[i] - mins[i]))
        normalized.append(new_row)
    
    #save CSV
    if out_file:
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(normalized)

    return len(normalized)

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