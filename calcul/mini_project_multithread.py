from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time


def threaded_matmul(A, B, num_threads=8):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    
    def worker(start_row, end_row):
        for i in range(start_row, end_row):
            for j in range(n):
                for k in range(n):
                    C[i,j] += A[i,k] * B[k,j]
    
    chunk = n // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = [exe.submit(worker, i*chunk, (i+1)*chunk if i<num_threads-1 else n) 
                  for i in range(num_threads)]
        for f in futures: f.result()
    return C



n = 100
A = np.random.rand(n, n)
B = np.random.rand(n, n)

times = []
for _ in range(10):
    start = time.time()
    C = threaded_matmul(A, B)
    end = time.time()
    times.append(end - start)

print("multithread times:", times)
print("multithread average:", np.mean(times))
