import numpy as np
import pandas as pd
import torch as t
import time
''' '''
from data_loader import (
    load_dataset, load_dataset_np, normalize_array, normalize_array_np,
    load_dataset_pd, split_xy, split_training_test
)
from vector_product import (
    dot_product, find_largest_dot_product_py, dot_product_np, find_largest_dot_product_np,
    mat_mul_np, mat_mul_t, dot_product_t
)


if __name__ == "__main__":
    pass
    # Test data loading
    print('Testing data loading...')
    try:
        data_list = load_dataset('Assignment2_Template_files/GasProperties.csv')
        print(f'Loaded {len(data_list)} rows (list of lists)')
    except Exception as e:
        print('load_dataset failed:', e)
    try:
        data_np = load_dataset_np('Assignment2_Template_files/GasProperties.csv')
        print(f'Loaded {data_np.shape[0]} rows (NumPy array)')
    except Exception as e:
        print('load_dataset_np failed:', e)
    try:
        data_pd = load_dataset_pd('Assignment2_Template_files/GasProperties.csv')
        print(f'Loaded {data_pd.shape[0]} rows (Pandas DataFrame)')
    except Exception as e:
        print('load_dataset_pd failed:', e)

    # Test normalization
    print('\nTesting normalization...')
    try:
        nrows = normalize_array(data_list)
        print(f'normalize_array processed {nrows} rows')
    except Exception as e:
        print('normalize_array failed:', e)
    try:
        nrows_np = normalize_array_np(data_np)
        print(f'normalize_array_np processed {nrows_np} rows')
    except Exception as e:
        print('normalize_array_np failed:', e)

    # Test splitting
    print('\nTesting split_xy and split_training_test...')
    try:
        X, Y = split_xy(data_pd)
        print(f'split_xy: X shape {X.shape}, Y shape {Y.shape}')
        X_train, Y_train, X_test, Y_test = split_training_test(X, Y)
        print(f'split_training_test: X_train {X_train.shape}, X_test {X_test.shape}')
    except Exception as e:
        print('Splitting failed:', e)

    # Test vector and matrix operations
    print('\nTesting vector and matrix operations...')
    try:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        print('dot_product:', dot_product(a, b))
        print('find_largest_dot_product_py:', find_largest_dot_product_py([a, b], b))
        a_np = np.array(a)
        b_np = np.array(b)
        print('dot_product_np:', dot_product_np(a_np, b_np))
        print('find_largest_dot_product_np:', find_largest_dot_product_np(np.array([a, b]), b_np))
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 3)
        print('mat_mul_np:', mat_mul_np(A, B))
        A_t = t.tensor(A, dtype=t.float64)
        B_t = t.tensor(B, dtype=t.float64)
        print('mat_mul_t:', mat_mul_t(A_t, B_t))
        print('dot_product_t:', dot_product_t(A_t[0], B_t[0]))
    except Exception as e:
        print('Vector/matrix operation failed:', e)
