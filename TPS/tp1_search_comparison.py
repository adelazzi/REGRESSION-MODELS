import random
import time
import numpy as np
import matplotlib.pyplot as plt

def recherche_sequentielle_simple(tab, x):
    comparisons = 0
    for val in tab:
        comparisons += 1
        if val == x:
            return True, comparisons
    return False, comparisons


def recherche_sequentielle_optimisee(tab, x):
    comparisons = 0
    for val in tab:
        comparisons += 1
        if val == x:
            return True, comparisons
        if val > x:
            return False, comparisons
    return False, comparisons


def recherche_binaire_iterative(tab, x):
    comparisons = 0
    left, right = 0, len(tab) - 1
    while left <= right:
        mid = (left + right) // 2
        comparisons += 1
        if tab[mid] == x:
            return True, comparisons
        elif tab[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return False, comparisons


def recherche_binaire_recursive(tab, x, left=0, right=None, comparisons=0):
    if right is None:
        right = len(tab) - 1
    if left > right:
        return False, comparisons
    mid = (left + right) // 2
    comparisons += 1
    if tab[mid] == x:
        return True, comparisons
    elif tab[mid] < x:
        return recherche_binaire_recursive(tab, x, mid + 1, right, comparisons)
    else:
        return recherche_binaire_recursive(tab, x, left, mid - 1, comparisons)


def test_algorithms(sizes, n_tests=30):
    results = {
        "seq_simple": {"time": [], "comparisons": []},
        "seq_opt": {"time": [], "comparisons": []},
        "bin_iter": {"time": [], "comparisons": []},
        "bin_rec": {"time": [], "comparisons": []}
    }

    for size in sizes:
        print(f"\n Testing for size = {size}")
        for _ in range(n_tests):
            tab = sorted(random.uniform(0, 10000) for _ in range(size))
            x = random.uniform(0, 10000)

            start = time.time()
            _, c = recherche_sequentielle_simple(tab, x)
            results["seq_simple"]["time"].append(time.time() - start)
            results["seq_simple"]["comparisons"].append(c)

            start = time.time()
            _, c = recherche_sequentielle_optimisee(tab, x)
            results["seq_opt"]["time"].append(time.time() - start)
            results["seq_opt"]["comparisons"].append(c)

            start = time.time()
            _, c = recherche_binaire_iterative(tab, x)
            results["bin_iter"]["time"].append(time.time() - start)
            results["bin_iter"]["comparisons"].append(c)

            start = time.time()
            _, c = recherche_binaire_recursive(tab, x)
            results["bin_rec"]["time"].append(time.time() - start)
            results["bin_rec"]["comparisons"].append(c)

    return results

 
sizes = [1000, 10000, 100000, 1000000]
n_tests = 30

results = test_algorithms(sizes, n_tests)

def print_table(results, sizes, n_tests):
    algos = {
        "seq_simple": "Séquentielle simple",
        "seq_opt": "Séquentielle optimisée",
        "bin_iter": "Binaire itérative",
        "bin_rec": "Binaire récursive"
    }

    print("\n" + "="*80)
    print(f"{'Taille':<10}{'Algorithme':<25}{'Moy. Comparaisons':<20}{'Moy. Temps (s)':<15}")
    print("="*80)

    for size in sizes:
        start = (sizes.index(size)) * n_tests
        end = start + n_tests
        for key, name in algos.items():
            avg_c = np.mean(results[key]["comparisons"][start:end])
            avg_t = np.mean(results[key]["time"][start:end])
            print(f"{str(size):<10}{name:<25}{avg_c:<20.2f}{avg_t:<15.6f}")
        print("-"*80)

print_table(results, sizes, n_tests)

def plot_results(results, sizes, metric):
    plt.figure(figsize=(8, 6))
    for name, label in [
        ("seq_simple", "Séquentielle simple"),
        ("seq_opt", "Séquentielle optimisée"),
        ("bin_iter", "Binaire itérative"),
        ("bin_rec", "Binaire récursive")
    ]:
        means = []
        for size in sizes:
            start = (sizes.index(size)) * n_tests
            end = start + n_tests
            means.append(np.mean(results[name][metric][start:end]))
        plt.plot(sizes, means, marker='o', label=label)

    plt.xscale('log')
    plt.xlabel("Taille du tableau (n)")
    plt.ylabel("Temps (s)" if metric == "time" else "Nombre de comparaisons")
    plt.title("Comparaison des algorithmes de recherche")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(results, sizes, "time")
plot_results(results, sizes, "comparisons")
