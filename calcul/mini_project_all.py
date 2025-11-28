import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings('ignore')

print(f"CPU Cores: {mp.cpu_count()}")
print(f"RAM: {os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / (1024**3):.1f} GB")

# =============================================================================
# 1. SEQUENTIAL (Pure Python loops)
# =============================================================================
def sequential_matmul(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# =============================================================================
# 2. MULTITHREAD (ThreadPoolExecutor)
# =============================================================================
def threaded_worker(start, end, A, B, C):
    n = A.shape[0]
    for i in range(start, end):
        for j in range(n):
            tmp = 0.0
            for k in range(n):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp

def threaded_matmul(A, B, num_threads=mp.cpu_count()):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    chunk = n // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for t in range(num_threads):
            start = t * chunk
            end = (t + 1) * chunk if t < num_threads - 1 else n
            futures.append(executor.submit(threaded_worker, start, end, A, B, C))
        for f in futures:
            f.result()
    return C

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

# =============================================================================
# 4. GPU SIMULATION (NumPy BLAS - OpenBLAS optimized)
# =============================================================================
def gpu_matmul(A, B):
    """NumPy @ uses optimized BLAS (your 'GPU equivalent')"""
    return A @ B

# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================
def benchmark_all(sizes=[256], runs=10):
    all_results = {}
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {n}x{n} matrices")
        print(f"{'='*60}")
        
        # Create matrices (float32 to save memory)
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
        
        methods = {
            "Sequential": sequential_matmul,
            f"Threads({mp.cpu_count()})": lambda a,b: threaded_matmul(a,b,mp.cpu_count()),
            "MPI(MultiProc-4)": lambda a,b: mpi_matmul(a,b,4),
            "GPU(BLAS)": gpu_matmul
        }
        
        results = {}
        for name, func in methods.items():
            print(f"Testing {name}...", end=" ")
            times = []
            
            for run in range(1, runs + 1):
                start = time.perf_counter()
                C = func(A, B)
                end = time.perf_counter()
                t = end - start
                times.append(t)
                if run % 5 == 0:
                    print(".", end="")
            
            avg_time = np.mean(times)
            results[name] = {
                "times": times,
                "average": avg_time,
                "std": np.std(times)
            }
            print(f"{avg_time:.3f}s ± {np.std(times):.3f}s")
        
        all_results[n] = results
        
        # Calculate Speedup & Efficiency
        seq_time = results["Sequential"]["average"]
        speedup = {}
        efficiency = {}
        for method in methods:
            if method != "Sequential":
                s = seq_time / results[method]["average"]
                p = mp.cpu_count() if "Threads" in method else 4 if "MPI" in method else 1
                e = s / p * 100  # Efficiency in %
                speedup[method] = s
                efficiency[method] = e
        
        print(f"\nSPEEDUP vs Sequential:")
        for method, s in speedup.items():
            print(f"  {method:15}: {s:6.2f}x (Efficiency: {efficiency[method]:5.1f}%)")
    
    return all_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    results = benchmark_all(sizes=[256, 512, 1024, 2048], runs=10)
    
    data = []
    speedup_data = []
    
    for n, methods in results.items():
        seq_time = methods["Sequential"]["average"]
        row = {"Size": f"{n}x{n}"}
        
        for method in methods:
            row[method] = f"{methods[method]['average']:.3f}"
            row[f"{method}_std"] = f"{methods[method]['std']:.3f}"
            
            if method != "Sequential":
                s = seq_time / methods[method]["average"]
                speedup_data.append({"Size": n, "Method": method, "Speedup": s})
        
        data.append(row)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(data)
    df.to_csv("matrix_benchmark_results.csv", index=False)
    
    speedup_df = pd.DataFrame(speedup_data)
    speedup_df.to_csv("speedup_results.csv", index=False)
    
    print(f"\n✅ Results saved:")
    print(f"   📊 matrix_benchmark_results.csv")
    print(f"   📈 speedup_results.csv")
    
    # PLOT 1: Execution Times
    plt.figure(figsize=(12, 8))
    
    sizes = list(results.keys())
    methods = list(results[sizes[0]].keys())
    
    x = np.arange(len(sizes))
    width = 0.2
    
    for i, method in enumerate(methods):
        times = [results[n][method]["average"] for n in sizes]
        plt.bar(x + i * width, times, width, label=method)
    
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (seconds)")
    plt.title("Matrix Multiplication Performance Comparison")
    plt.xticks(x + width * 1.5, [f"{n}x{n}" for n in sizes])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("performance_times.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # PLOT 2: Speedup
    plt.figure(figsize=(10, 6))
    for method in speedup_df["Method"].unique():
        method_data = speedup_df[speedup_df["Method"] == method]
        plt.plot(method_data["Size"], method_data["Speedup"], 
                marker='o', label=method, linewidth=2, markersize=8)
    
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup (vs Sequential)")
    plt.title("Speedup Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("speedup_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Plots saved:")
    print(f"   📈 performance_times.png")
    print(f"   🚀 speedup_plot.png")
    print("\n🎉 MINI-PROJECT COMPLETE! Copy CSV files and PNG plots to your Word report.")
