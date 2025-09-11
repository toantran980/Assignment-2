import torch as t
import numpy as np
import time
import platform
import os

def get_hardware_specs():
    specs = {}
    specs['CPU'] = platform.processor()
    specs['CPU cores'] = os.cpu_count()
    try:
        specs['System RAM (GB)'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2)
    except Exception:
        specs['System RAM (GB)'] = 'N/A'
    if t.cuda.is_available():
        gpu_idx = 0
        specs['GPU'] = t.cuda.get_device_name(gpu_idx)
        props = t.cuda.get_device_properties(gpu_idx)
        specs['GPU cores'] = props.multi_processor_count * 128
        # Try both possible attribute names
        clock = getattr(props, 'clockRate', getattr(props, 'clock_rate', None))
        mem_clock = getattr(props, 'memoryClockRate', getattr(props, 'memory_clock_rate', None))
        mem_bus = getattr(props, 'memoryBusWidth', getattr(props, 'memory_bus_width', None))
        specs['GPU clock (MHz)'] = clock // 1000 if clock else 'N/A'
        specs['GPU memory (GB)'] = round(props.total_memory / (1024**3), 2)
        if mem_clock and mem_bus:
            specs['GPU memory bandwidth (GB/s)'] = round(mem_bus / 8 * mem_clock * 2 / 1e6, 2)
        else:
            specs['GPU memory bandwidth (GB/s)'] = 'N/A'
    else:
        specs['GPU'] = 'None'
    return specs

def estimate_flops(specs):
    cpu_ghz = 3.0  # Update to your actual CPU clock speed if known
    cpu_flops = specs['CPU cores'] * cpu_ghz * 1e9 * 16  # 16 FLOPs/cycle for modern CPUs (AVX-512)
    gpu_flops = 0
    if specs['GPU'] != 'None':
        gpu_clock = specs['GPU clock (MHz)']
        if isinstance(gpu_clock, (int, float)):
            gpu_ghz = gpu_clock / 1000
            gpu_flops = specs['GPU cores'] * gpu_ghz * 1e9 * 2  # 2 FLOPs/cycle (FMA)
        else:
            gpu_flops = 0
    return cpu_flops, gpu_flops

def benchmark_dot_product(X, Y):
    # CPU
    start = time.time()
    cpu_result = np.dot(X, Y)
    cpu_time = time.time() - start
    # GPU
    if t.cuda.is_available():
        X_gpu = t.tensor(X, dtype=t.float64, device='cuda')
        Y_gpu = t.tensor(Y, dtype=t.float64, device='cuda')
        t.cuda.synchronize()
        start = time.time()
        gpu_result = t.dot(X_gpu, Y_gpu)
        t.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None
    return cpu_time, gpu_time

def benchmark_matmul(X):
    # CPU
    start = time.time()
    cpu_result = np.matmul(X, X)
    cpu_time = time.time() - start
    # GPU
    if t.cuda.is_available():
        X_gpu = t.tensor(X, dtype=t.float64, device='cuda')
        t.cuda.synchronize()
        start = time.time()
        gpu_result = t.matmul(X_gpu, X_gpu)
        t.cuda.synchronize()
        gpu_time = time.time() - start
    else:
        gpu_time = None
    return cpu_time, gpu_time

if __name__ == '__main__':
    print('Collecting hardware specs...')
    specs = get_hardware_specs()
    for k, v in specs.items():
        print(f'{k}: {v}')
    cpu_flops, gpu_flops = estimate_flops(specs)
    print(f'Estimated CPU peak: {cpu_flops/1e12:.2f} TFLOPS')
    if gpu_flops:
        print(f'Estimated GPU peak: {gpu_flops/1e12:.2f} TFLOPS')
    print('\nGenerating random data for benchmarking...')
    size = 1000
    X = np.random.randn(size, size)
    Y = np.random.randn(size)
    print('\nBenchmarking dot product...')
    cpu_dot, gpu_dot = benchmark_dot_product(X[0], Y)
    print(f'Dot product CPU time: {cpu_dot:.6f} s')
    if gpu_dot:
        print(f'Dot product GPU time: {gpu_dot:.6f} s')
    else:
        print('Dot product GPU not available.')
    print('\nBenchmarking matrix multiplication...')
    cpu_mat, gpu_mat = benchmark_matmul(X)
    print(f'Matrix multiplication CPU time: {cpu_mat:.6f} s')
    if gpu_mat:
        print(f'Matrix multiplication GPU time: {gpu_mat:.6f} s')
    else:
        print('Matrix multiplication GPU not available.')
    print('\nNote: Theoretical peak performance is rarely achieved in practice due to memory bandwidth, parallelization limits, and other system bottlenecks.')
