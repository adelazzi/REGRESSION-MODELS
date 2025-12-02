



import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 3. MPI SIMULATION (Multiprocessing Pool)
# =============================================================================
def mpi_worker(args):
    i_start, i_end, A, B = args
    n = A.shape[0]
    local_C = np.zeros((i_end - i_start, n), dtype=np.float32)
    for i in range(i_start, i_end):
        for j in range(n):
            tmp = 0.0
            for k in range(n):
                tmp += A[i, k] * B[k, j]
            local_C[i - i_start, j] = tmp
    return local_C

def mpi_matmul(A, B, num_processes=4):
    n = A.shape[0]
    chunk = n // num_processes
    args = [(i * chunk, (i + 1) * chunk if i < num_processes - 1 else n, A, B) 
            for i in range(num_processes)]
    
    with mp.Pool(processes=num_processes) as pool:
        chunks = pool.map(mpi_worker, args)
    return np.vstack(chunks)





n = 100
A = np.random.rand(n, n)
B = np.random.rand(n, n)

times = []
for _ in range(10):
    start = time.time()
    C = mpi_matmul(A, B)
    end = time.time()
    times.append(end - start)

print("mpi times:", times)
print("mpi average:", np.mean(times))
