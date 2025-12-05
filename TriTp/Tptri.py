import random
import time
import matplotlib.pyplot as plt
import sys

# Augmenter la limite de recursion
sys.setrecursionlimit(10000)

# Generation des tableaux
def generate_array(size, type):
    if type == 'random':
        return [random.randint(1, 1000) for i in range(size)]
    elif type == 'sorted':
        return list(range(size))
    elif type == 'reverse':
        return list(range(size, 0, -1))

# Tri fusion
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Tri rapide
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Choisir un pivot aleatoire pour eviter le pire cas
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Test de performance
def test_sorting(algorithm, arr):
    arr_copy = arr.copy()
    start = time.time()
    
    if algorithm == 'merge':
        sorted_arr = merge_sort(arr_copy)
    else:  # quick
        quick_sort(arr_copy)
        sorted_arr = arr_copy
    
    end = time.time()
    return end - start

# Execution des tests
def run_tests():
    sizes = [1000, 5000, 10000]
    types = ['random', 'sorted', 'reverse']
    
    results = {}
    
    for size in sizes:
        print(f"Testing size: {size}")
        results[size] = {}
        
        for arr_type in types:
            print(f"  Type: {arr_type}")
            arr = generate_array(size, arr_type)
            
            # Test merge sort
            merge_time = test_sorting('merge', arr)
            
            # Test quick sort
            quick_time = test_sorting('quick', arr)
            
            results[size][arr_type] = {
                'merge': merge_time,
                'quick': quick_time
            }
            
            print(f"    Merge: {merge_time:.4f}s, Quick: {quick_time:.4f}s")
    
    return results

# Graphiques
def plot_results(results):
    sizes = list(results.keys())
    types = ['random', 'sorted', 'reverse']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, arr_type in enumerate(types):
        merge_times = [results[size][arr_type]['merge'] for size in sizes]
        quick_times = [results[size][arr_type]['quick'] for size in sizes]
        
        axes[i].plot(sizes, merge_times, 'b-o', label='Merge Sort')
        axes[i].plot(sizes, quick_times, 'r-s', label='Quick Sort')
        axes[i].set_title(f'{arr_type.capitalize()} Array')
        axes[i].set_xlabel('Size')
        axes[i].set_ylabel('Time (s)')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Programme principal
if __name__ == "__main__":
    print("Comparaison Merge Sort vs Quick Sort")
    print("-" * 40)
    
    results = run_tests()
    
    print("\nResultats:")
    for size in results:
        print(f"\nTaille {size}:")
        for arr_type in results[size]:
            merge_time = results[size][arr_type]['merge']
            quick_time = results[size][arr_type]['quick']
            print(f"  {arr_type}: Merge={merge_time:.4f}s, Quick={quick_time:.4f}s")
    
    plot_results(results)