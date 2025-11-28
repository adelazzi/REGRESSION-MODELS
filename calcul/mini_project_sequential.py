import numpy as np
import time

def sequential_matmul(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

n = 10 
A = np.random.rand(n, n)
B = np.random.rand(n, n)

times = []
for _ in range(10):
    start = time.time()
    C = sequential_matmul(A, B)
    end = time.time()
    times.append(end - start)

print("Sequential times:", times)
print("Sequential average:", np.mean(times))
