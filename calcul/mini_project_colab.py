# ================================================================
# MINI-PROJET: COMPARAISON DES PERFORMANCES DE MULTIPLICATION
# Séquentiel, Threads, MPI (multiprocessing), GPU CUDA (CuPy)
# Google Colab ready
# ================================================================
!pip install -q cupy-cuda12x  # pour Colab CUDA 12.x [web:11]

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import warnings, os, gc
warnings.filterwarnings("ignore")

# ------------------------------------------------
# 0. Infos machine
# ------------------------------------------------
print(f"CPU cores: {os.cpu_count()}")
print(f"RAM: {os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / (1024**3):.1f} GB")

try:
    import cupy as cp
    GPU = True
    # petit test GPU
    _ = (cp.ones((16,16)) @ cp.ones((16,16)))
    print("GPU CUDA disponible via CuPy ✅")
except Exception as e:
    GPU = False
    print("GPU CUDA non disponible, CuPy utilisera NumPy ⚠️", e)

# ------------------------------------------------
# 1. Version séquentielle
# ------------------------------------------------
def seq_mm(A, B):
    n = A.shape[0]
    C = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i,k]*B[k,j]
            C[i,j] = s
    return C

# ------------------------------------------------
# 2. Version multithreads
# ------------------------------------------------
def thread_worker(i_start, i_end, A, B, C):
    n = A.shape[0]
    for i in range(i_start, i_end):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i,k]*B[k,j]
            C[i,j] = s

def threads_mm(A, B, num_threads):
    n = A.shape[0]
    C = np.zeros((n,n), dtype=np.float32)
    chunk = n // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futs = []
        for t in range(num_threads):
            start = t*chunk
            end   = (t+1)*chunk if t < num_threads-1 else n
            futs.append(ex.submit(thread_worker, start, end, A, B, C))
        for f in futs:
            f.result()
    return C

# ------------------------------------------------
# 3. Version MPI simulée (multiprocessing)
# ------------------------------------------------
def mpi_worker(args):
    i_start, i_end, A, B = args
    n = A.shape[0]
    subC = np.zeros((i_end-i_start, n), dtype=np.float32)
    for i in range(i_start, i_end):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i,k]*B[k,j]
            subC[i-i_start, j] = s
    return subC

def mpi_mm(A, B, num_proc):
    n = A.shape[0]
    chunk = n // num_proc
    args = [(p*chunk, (p+1)*chunk if p < num_proc-1 else n, A, B)
            for p in range(num_proc)]
    with mp.Pool(processes=num_proc) as pool:
        parts = pool.map(mpi_worker, args)
    return np.vstack(parts)

# ------------------------------------------------
# 4. Version GPU CUDA (CuPy)
# ------------------------------------------------
def gpu_mm(A, B, block="16x16"):
    if not GPU:
        # fallback CPU BLAS
        return A @ B
    # passage CPU→GPU
    dA = cp.asarray(A)
    dB = cp.asarray(B)
    dC = dA @ dB          # utilise cuBLAS très optimisé [web:9]
    C  = cp.asnumpy(dC)   # retour GPU→CPU
    del dA, dB, dC
    cp.get_default_memory_pool().free_all_blocks()
    return C

# ------------------------------------------------
# 5. Benchmark: 10 exécutions, calcul accélération & efficacité
# ------------------------------------------------
def benchmark_one(method_fn, A, B, runs=10):
    times = []
    # petit échauffement
    _ = method_fn(A, B)
    for _ in range(runs):
        start = time.perf_counter()
        _ = method_fn(A, B)
        end = time.perf_counter()
        times.append(end-start)
    return np.array(times, dtype=np.float64)

def run_all(n=256, runs=10,
            thread_list=(2,3,4,6,12),
            mpi_list=(2,3,4),
            block_sizes=("16x16","32x32")):

    A = np.random.rand(n,n).astype(np.float32)
    B = np.random.rand(n,n).astype(np.float32)

    results = {}

    # Séquentiel
    print("Séquentiel...")
    t_seq = benchmark_one(seq_mm, A, B, runs)
    results["Sequential"] = t_seq

    # Threads
    for th in thread_list:
        print(f"Threads {th}...")
        t = benchmark_one(lambda X,Y: threads_mm(X,Y,th), A, B, runs)
        results[f"Threads-{th}"] = t

    # MPI
    for p in mpi_list:
        print(f"MPI {p} processus...")
        t = benchmark_one(lambda X,Y: mpi_mm(X,Y,p), A, B, runs)
        results[f"MPI-{p}"] = t

    # GPU CUDA (CuPy)
    for blk in block_sizes:
        print(f"CUDA GPU bloc {blk}...")
        t = benchmark_one(lambda X,Y: gpu_mm(X,Y,blk), A, B, runs)
        results[f"CUDA-{blk}"] = t

    return results

# lance le benchmark principal
results = run_all(n=256, runs=10)

# ------------------------------------------------
# 6. Construction du tableau “Numéro d’exécution / Moyen”
# ------------------------------------------------
rows = []
for name, t in results.items():
    row = {"Programme": name}
    for i in range(10):
        row[f"Run_{i+1}"] = t[i]
    row["Mean"] = t.mean()
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("mesures_brutes.csv", index=False)
print("\nTableau brut sauvegardé dans mesures_brutes.csv")
display(df)
# ------------------------------------------------
# 7. Calcul accélération & efficacité
# ------------------------------------------------
seq_mean = df[df["Programme"]=="Sequential"]["Mean"].values[0]

def nb_ressources(name):
    if name.startswith("Threads-"):
        return int(name.split("-")[1])
    if name.startswith("MPI-"):
        return int(name.split("-")[1])
    if name.startswith("CUDA-"):
        # on compte la GPU comme 1 ressource
        return 1
    return 1  # séquentiel

summary = []
for _, row in df.iterrows():
    name = row["Programme"]
    mean_t = row["Mean"]
    S = seq_mean / mean_t
    P = nb_ressources(name)
    E = S / P
    summary.append({"Programme": name,
                    "MeanTime": mean_t,
                    "Speedup": S,
                    "Efficiency": E})

summary_df = pd.DataFrame(summary)
summary_df.to_csv("acceleration_efficacite.csv", index=False)
print("Accélération / efficacité sauvegardées dans acceleration_efficacite.csv")
display(summary_df.sort_values("Programme"))
# ------------------------------------------------
# 8. Courbes
# ------------------------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Temps moyen (s)")
plt.bar(summary_df["Programme"], summary_df["MeanTime"])
plt.xticks(rotation=60, ha="right")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.subplot(1,3,2)
plt.title("Accélération S")
plt.bar(summary_df["Programme"], summary_df["Speedup"])
plt.xticks(rotation=60, ha="right")
plt.grid(True, alpha=0.3)

plt.subplot(1,3,3)
plt.title("Efficacité E")
plt.bar(summary_df["Programme"], summary_df["Efficiency"])
plt.xticks(rotation=60, ha="right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("courbes_performance.png", dpi=300, bbox_inches="tight")
plt.show()
