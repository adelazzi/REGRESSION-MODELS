import random
import time


class SortMetrics:
    def __init__(self):
        self.comparisons = 0
        self.movements = 0
        self.cpu_time = 0


def generate_array(size, order='random'):
    arr = list(range(size))
    if order == 'random':
        random.shuffle(arr)
    elif order == 'descending':
        arr.reverse()
    return arr


def selection_sort(arr):
    metrics = SortMetrics()
    start = time.process_time()

    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            metrics.comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            metrics.movements += 1

    metrics.cpu_time = time.process_time() - start
    return metrics


def bubble_sort(arr):
    metrics = SortMetrics()
    start = time.process_time()

    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - 1 - i):
            metrics.comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                metrics.movements += 1
                swapped = True
        if not swapped:
            break

    metrics.cpu_time = time.process_time() - start
    return metrics


def insertion_sort_exchanges(arr):
    metrics = SortMetrics()
    start = time.process_time()

    n = len(arr)
    for i in range(1, n):
        j = i
        while j > 0:
            metrics.comparisons += 1
            if arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                metrics.movements += 1
                j -= 1
            else:
                break

    metrics.cpu_time = time.process_time() - start
    return metrics


def insertion_sort_shifts(arr):
    metrics = SortMetrics()
    start = time.process_time()

    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1

        while j >= 0:
            metrics.comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                metrics.movements += 1
                j -= 1
            else:
                break
        arr[j + 1] = key
        metrics.movements += 1

    metrics.cpu_time = time.process_time() - start
    return metrics


def run_experiment(sizes, orders, repetitions=30):
    report = {}
    sort_algorithms = {
        'Selection Sort': selection_sort,
        'Bubble Sort': bubble_sort,
        'Insertion Sort (Exchanges)': insertion_sort_exchanges,
        'Insertion Sort (Shifts)': insertion_sort_shifts,
    }

    for size in sizes:
        print(f"Testing array size: {size}")
        for order in orders:
            print(f"  Initial order: {order}")
            key = (size, order)
            report[key] = {}
            for name, func in sort_algorithms.items():
                total_comparisons = 0
                total_movements = 0
                total_cpu_time = 0

                print(f"    Algorithm: {name}")
                for i in range(1, repetitions + 1):
                    arr = generate_array(size, order)
                    metrics = func(arr.copy())
                    total_comparisons += metrics.comparisons
                    total_movements += metrics.movements
                    total_cpu_time += metrics.cpu_time

                    if i % (repetitions // 5) == 0 or i == repetitions:
                        print(f"      Completed {i}/{repetitions} runs")

                report[key][name] = {
                    'Avg Comparisons': total_comparisons / repetitions,
                    'Avg Movements': total_movements / repetitions,
                    'Avg CPU Time (sec)': total_cpu_time / repetitions
                }

                print(f"    Completed {name}: "
                      f"Avg Comparisons={report[key][name]['Avg Comparisons']:.2f}, "
                      f"Avg Movements={report[key][name]['Avg Movements']:.2f}, "
                      f"Avg CPU Time={report[key][name]['Avg CPU Time (sec)']:.4f}s")

    return report


if __name__ == "__main__":
    array_sizes = [1000, 10000]  # Adjust sizes as needed
    initial_orders = ['random', 'ascending', 'descending']
    repetitions = 30

    results = run_experiment(array_sizes, initial_orders, repetitions)

    print("\nSummary of results:")
    for (size, order), data in results.items():
        print(f"Array size: {size}, Initial order: {order}")
        for algorithm, metrics in data.items():
            print(f"  {algorithm} -> Comparisons: {metrics['Avg Comparisons']:.2f}, "
                  f"Movements: {metrics['Avg Movements']:.2f}, "
                  f"CPU Time: {metrics['Avg CPU Time (sec)']:.4f} seconds")



'''
The code is an experimental framework in Python to study the performance of simple sorting algorithms. It focuses on four key algorithms: selection sort, bubble sort, insertion sort by exchanges, and insertion sort by shifting. The program evaluates these algorithms on arrays of different sizes and initial orderings, repeatedly running each algorithm to gather meaningful average metrics related to comparisons, movements, and CPU execution time.

### Main functionalities:
- Data generation for sorting tests: creates arrays of defined sizes and orders (random, ascending, descending).
- Implementation of four sorting algorithms with counting of:
  - Comparisons: how many times elements are compared.
  - Movements: how many times elements are swapped or shifted.
- Timing the CPU time taken by each sort.
- Running multiple repetitions (e.g., 30) for statistical reliability.
- Printing progress and summarizing average results.

***

### Detailed description of each method:

**1. `generate_array(size, order)`**
- Produces an integer list of the given size.
- Orders the list as specified: shuffled/random, sorted ascending, or sorted descending.

**2. `selection_sort(arr)`**
- Sorts the array by repeatedly selecting the minimum value from the unsorted portion and swapping it to the beginning.
- Counts comparisons for finding the minimum.
- Counts movements for each swap.
- Measures CPU time spent in sorting.

**3. `bubble_sort(arr)`**
- Repeatedly compares adjacent elements, swapping if they are out of order.
- Continues passes until no swaps occur, indicating array is sorted.
- Counts each adjacent comparison and swap (movement).
- Measures CPU runtime.

**4. `insertion_sort_exchanges(arr)`**
- For each element, moves it leftwards by pairwise swapping with the previous element until reaching the correct sorted position.
- Counts each comparison and each swap.
- Measures sorting time.

**5. `insertion_sort_shifts(arr)`**
- Similar to the previous insertion sort, but instead of swapping, shifts all larger elements to the right and inserts the key element once at the correct location.
- Counts comparisons and movements (shifts plus final insert).
- Measures execution time.

**6. `run_experiment(sizes, orders, repetitions)`**
- Coordinates the overall testing.
- For each array size and initial order, runs each sorting algorithm multiple times.
- Tracks cumulative comparisons, movements, and CPU time.
- Prints progress for each algorithm and repetition.
- Returns averaged metrics for all combinations.


'''