import numpy as np
import time

def numpy_blas_matmul(A, B):
    return A @ B  # or np.dot(A, B)

n = 100  
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)

times = []
for _ in range(10):
    start = time.time()
    C = numpy_blas_matmul(A, B)
    end = time.time()
    times.append(end - start)

print("CUDA GPU times:", times )
print(f"NumPy BLAS average: {np.mean(times):.2f}s")

