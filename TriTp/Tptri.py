import random
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import statistics

# Augmenter la limite de récursion pour les gros tableaux
sys.setrecursionlimit(100000)

def generate_array(size, distribution='random'):
    """Génère un tableau selon différentes distributions"""
    if distribution == 'random':
        return [random.randint(0, 10000) for _ in range(size)]
    elif distribution == 'sorted':
        return list(range(size))
    elif distribution == 'reverse':
        return list(range(size, 0, -1))
    elif distribution == 'nearly_sorted':
        arr = list(range(size))
        # Introduire quelques inversions aléatoires
        num_swaps = max(1, size // 10)
        for _ in range(num_swaps):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    return []

def merge_sort(arr, comparisons, swaps):
    """Tri fusion avec comptage des opérations"""
    if len(arr) <= 1:
        return arr
    
    # Diviser le tableau en deux moitiés
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], comparisons, swaps)
    right = merge_sort(arr[mid:], comparisons, swaps)
    
    # Fusionner les deux moitiés triées
    return merge(left, right, comparisons, swaps)

def merge(left, right, comparisons, swaps):
    """Fusionne deux tableaux triés"""
    result = []
    i, j = 0, 0
    
    # Comparer et fusionner les éléments
    while i < len(left) and j < len(right):
        comparisons[0] += 1
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
        swaps[0] += 1
    
    # Ajouter les éléments restants
    while i < len(left):
        result.append(left[i])
        i += 1
        swaps[0] += 1
    
    while j < len(right):
        result.append(right[j])
        j += 1
        swaps[0] += 1
    
    return result

def quick_sort(arr, comparisons, swaps, low=0, high=None, depth=0, pivot_strategy='median_of_three'):
    """Tri rapide avec comptage des opérations et différentes stratégies de pivot"""
    if high is None:
        high = len(arr) - 1
    
    # Protection contre la récursion trop profonde
    if depth > 2 * len(arr).bit_length():
        heap_sort_simple(arr, comparisons, swaps, low, high)
        return
    
    if low < high:
        # Choisir la stratégie de pivot
        if high - low > 10:
            if pivot_strategy == 'median_of_three':
                median_of_three(arr, low, high, comparisons, swaps)
            elif pivot_strategy == 'random':
                random_pivot(arr, low, high, swaps)
        
        # Partitionner le tableau
        pivot_index = partition(arr, comparisons, swaps, low, high)
        
        # Trier récursivement les deux parties
        quick_sort(arr, comparisons, swaps, low, pivot_index - 1, depth + 1, pivot_strategy)
        quick_sort(arr, comparisons, swaps, pivot_index + 1, high, depth + 1, pivot_strategy)

def median_of_three(arr, low, high, comparisons, swaps):
    """Sélectionne le pivot selon la méthode médiane de trois"""
    mid = (low + high) // 2
    
    # Comparer et organiser low, mid, high
    comparisons[0] += 2
    if arr[mid] < arr[low]:
        arr[low], arr[mid] = arr[mid], arr[low]
        swaps[0] += 1
    
    if arr[high] < arr[low]:
        arr[low], arr[high] = arr[high], arr[low]
        swaps[0] += 1
    
    if arr[high] < arr[mid]:
        arr[mid], arr[high] = arr[high], arr[mid]
        swaps[0] += 1
    
    # Placer la médiane à la fin
    arr[mid], arr[high] = arr[high], arr[mid]
    swaps[0] += 1

def random_pivot(arr, low, high, swaps):
    """Sélection aléatoire du pivot"""
    random_index = random.randint(low, high)
    arr[random_index], arr[high] = arr[high], arr[random_index]
    swaps[0] += 1

def heap_sort_simple(arr, comparisons, swaps, low, high):
    """Tri par tas simple comme solution de secours"""
    # Tri par insertion simple pour éviter la complexité du heap sort
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        
        while j >= low:
            comparisons[0] += 1
            if arr[j] <= key:
                break
            arr[j + 1] = arr[j]
            swaps[0] += 1
            j -= 1
        
        arr[j + 1] = key
        if j + 1 != i:
            swaps[0] += 1

def partition(arr, comparisons, swaps, low, high):
    """Partitionne le tableau pour le tri rapide"""
    # Le pivot est maintenant à la fin grâce à median_of_three
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        comparisons[0] += 1
        if arr[j] <= pivot:
            i += 1
            if i != j:  # Éviter les échanges inutiles
                arr[i], arr[j] = arr[j], arr[i]
                swaps[0] += 1
    
    # Placer le pivot à sa position finale
    if i + 1 != high:
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        swaps[0] += 1
    
    return i + 1

def test_algorithm(algorithm, arr):
    """Teste un algorithme de tri et retourne les métriques"""
    arr_copy = arr.copy()
    comparisons = [0]
    swaps = [0]
    
    start_time = time.perf_counter()
    
    if algorithm == "merge":
        sorted_arr = merge_sort(arr_copy, comparisons, swaps)
        end_time = time.perf_counter()
        # Vérifier que le tableau est bien trié
        is_sorted = sorted_arr == sorted(arr)
    else:  # quick sort
        quick_sort(arr_copy, comparisons, swaps)
        sorted_arr = arr_copy
        end_time = time.perf_counter()
        # Vérifier que le tableau est bien trié
        is_sorted = arr_copy == sorted(arr)
    
    if not is_sorted:
        raise ValueError(f"L'algorithme {algorithm} n'a pas correctement trié le tableau")
    
    return {
        'time': end_time - start_time,
        'comparisons': comparisons[0],
        'swaps': swaps[0]
    }

def test_algorithm_multiple(algorithm, arr, iterations=30, pivot_strategy='median_of_three'):
    """Teste un algorithme plusieurs fois et retourne les statistiques"""
    times = []
    comparisons_list = []
    swaps_list = []
    
    for _ in range(iterations):
        arr_copy = arr.copy()
        comparisons = [0]
        swaps = [0]
        
        start_time = time.perf_counter()
        
        if algorithm == "merge":
            sorted_arr = merge_sort(arr_copy, comparisons, swaps)
            end_time = time.perf_counter()
            is_sorted = sorted_arr == sorted(arr)
        else:  # quick sort
            quick_sort(arr_copy, comparisons, swaps, pivot_strategy=pivot_strategy)
            sorted_arr = arr_copy
            end_time = time.perf_counter()
            is_sorted = arr_copy == sorted(arr)
        
        if not is_sorted:
            raise ValueError(f"L'algorithme {algorithm} n'a pas correctement trié le tableau")
        
        times.append(end_time - start_time)
        comparisons_list.append(comparisons[0])
        swaps_list.append(swaps[0])
    
    return {
        'time_mean': statistics.mean(times),
        'time_std': statistics.stdev(times) if len(times) > 1 else 0,
        'comparisons_mean': statistics.mean(comparisons_list),
        'comparisons_std': statistics.stdev(comparisons_list) if len(comparisons_list) > 1 else 0,
        'swaps_mean': statistics.mean(swaps_list),
        'swaps_std': statistics.stdev(swaps_list) if len(swaps_list) > 1 else 0,
        'times_raw': times,
        'comparisons_raw': comparisons_list,
        'swaps_raw': swaps_list
    }

def run_experiments(sizes, distributions):
    """Exécute les expérimentations pour différentes tailles et distributions"""
    results = {
        'merge': {dist: {'time': [], 'comparisons': [], 'swaps': []} for dist in distributions},
        'quick': {dist: {'time': [], 'comparisons': [], 'swaps': []} for dist in distributions}
    }
    
    for size in sizes:
        print(f"📊 Testing size: {size}")
        for distribution in distributions:
            print(f"  - Distribution: {distribution}")
            arr = generate_array(size, distribution)
            
            try:
                # Test tri fusion
                merge_results = test_algorithm("merge", arr)
                results['merge'][distribution]['time'].append(merge_results['time'])
                results['merge'][distribution]['comparisons'].append(merge_results['comparisons'])
                results['merge'][distribution]['swaps'].append(merge_results['swaps'])
                
                # Test tri rapide
                quick_results = test_algorithm("quick", arr)
                results['quick'][distribution]['time'].append(quick_results['time'])
                results['quick'][distribution]['comparisons'].append(quick_results['comparisons'])
                results['quick'][distribution]['swaps'].append(quick_results['swaps'])
                
            except Exception as e:
                print(f"    ⚠️  Erreur pour taille {size}, distribution {distribution}: {e}")
                # Ajouter des valeurs par défaut en cas d'erreur
                results['merge'][distribution]['time'].append(0)
                results['merge'][distribution]['comparisons'].append(0)
                results['merge'][distribution]['swaps'].append(0)
                results['quick'][distribution]['time'].append(0)
                results['quick'][distribution]['comparisons'].append(0)
                results['quick'][distribution]['swaps'].append(0)
    
    return results

def run_comprehensive_experiments(sizes, distributions, iterations=30):
    """Exécute les expérimentations complètes selon le protocole du TP"""
    results = {
        'merge': {dist: {'time_mean': [], 'time_std': [], 'comparisons_mean': [], 
                        'comparisons_std': [], 'swaps_mean': [], 'swaps_std': []} 
                 for dist in distributions},
        'quick_median': {dist: {'time_mean': [], 'time_std': [], 'comparisons_mean': [], 
                               'comparisons_std': [], 'swaps_mean': [], 'swaps_std': []} 
                        for dist in distributions},
        'quick_random': {dist: {'time_mean': [], 'time_std': [], 'comparisons_mean': [], 
                               'comparisons_std': [], 'swaps_mean': [], 'swaps_std': []} 
                        for dist in distributions}
    }
    
    total_tests = len(sizes) * len(distributions) * 3  # 3 algorithmes
    current_test = 0
    
    for size in sizes:
        print(f"\n📊 Testing size: {size:,} éléments")
        print("=" * 50)
        
        for distribution in distributions:
            current_test += 3
            print(f"  🔄 Distribution: {distribution} ({current_test}/{total_tests} tests)")
            
            # Générer le même tableau pour tous les algorithmes
            arr = generate_array(size, distribution)
            
            try:
                # Test Merge Sort
                print("    ⏱️  Merge Sort...", end="")
                merge_stats = test_algorithm_multiple("merge", arr, iterations)
                results['merge'][distribution]['time_mean'].append(merge_stats['time_mean'])
                results['merge'][distribution]['time_std'].append(merge_stats['time_std'])
                results['merge'][distribution]['comparisons_mean'].append(merge_stats['comparisons_mean'])
                results['merge'][distribution]['comparisons_std'].append(merge_stats['comparisons_std'])
                results['merge'][distribution]['swaps_mean'].append(merge_stats['swaps_mean'])
                results['merge'][distribution]['swaps_std'].append(merge_stats['swaps_std'])
                print(f" ✅ ({merge_stats['time_mean']:.4f}s)")
                
                # Test Quick Sort (median of three)
                print("    ⏱️  Quick Sort (median)...", end="")
                quick_median_stats = test_algorithm_multiple("quick", arr, iterations, 'median_of_three')
                results['quick_median'][distribution]['time_mean'].append(quick_median_stats['time_mean'])
                results['quick_median'][distribution]['time_std'].append(quick_median_stats['time_std'])
                results['quick_median'][distribution]['comparisons_mean'].append(quick_median_stats['comparisons_mean'])
                results['quick_median'][distribution]['comparisons_std'].append(quick_median_stats['comparisons_std'])
                results['quick_median'][distribution]['swaps_mean'].append(quick_median_stats['swaps_mean'])
                results['quick_median'][distribution]['swaps_std'].append(quick_median_stats['swaps_std'])
                print(f" ✅ ({quick_median_stats['time_mean']:.4f}s)")
                
                # Test Quick Sort (random pivot)
                print("    ⏱️  Quick Sort (random)...", end="")
                quick_random_stats = test_algorithm_multiple("quick", arr, iterations, 'random')
                results['quick_random'][distribution]['time_mean'].append(quick_random_stats['time_mean'])
                results['quick_random'][distribution]['time_std'].append(quick_random_stats['time_std'])
                results['quick_random'][distribution]['comparisons_mean'].append(quick_random_stats['comparisons_mean'])
                results['quick_random'][distribution]['comparisons_std'].append(quick_random_stats['comparisons_std'])
                results['quick_random'][distribution]['swaps_mean'].append(quick_random_stats['swaps_mean'])
                results['quick_random'][distribution]['swaps_std'].append(quick_random_stats['swaps_std'])
                print(f" ✅ ({quick_random_stats['time_mean']:.4f}s)")
                
            except Exception as e:
                print(f"\n    ❌ Erreur: {e}")
                # Ajouter des valeurs par défaut
                for algo in ['merge', 'quick_median', 'quick_random']:
                    for metric in ['time_mean', 'time_std', 'comparisons_mean', 
                                  'comparisons_std', 'swaps_mean', 'swaps_std']:
                        results[algo][distribution][metric].append(0)
    
    return results

def plot_results(sizes, results, metric='time'):
    """Trace les résultats des expérimentations"""
    distributions = ['random', 'sorted', 'reverse', 'nearly_sorted']
    colors = {'merge': 'blue', 'quick': 'red'}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, distribution in enumerate(distributions):
        ax = axes[idx]
        
        # Tracer pour merge sort
        ax.plot(sizes, results['merge'][distribution][metric], 
                label='Tri Fusion', color=colors['merge'], marker='o', linewidth=2)
        
        # Tracer pour quick sort
        ax.plot(sizes, results['quick'][distribution][metric], 
                label='Tri Rapide', color=colors['quick'], marker='s', linewidth=2)
        
        ax.set_title(f'Distribution: {distribution.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Taille du tableau', fontsize=10)
        
        if metric == 'time':
            ax.set_ylabel('Temps d\'exécution (secondes)', fontsize=10)
        elif metric == 'comparisons':
            ax.set_ylabel('Nombre de comparaisons', fontsize=10)
        else:
            ax.set_ylabel('Nombre d\'échanges', fontsize=10)
            
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Échelle logarithmique pour mieux voir les différences
    
    plt.suptitle(f'Comparaison des algorithmes de tri - {metric.capitalize()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_comprehensive_results(sizes, results):
    """Trace des graphiques complets selon les exigences du TP"""
    distributions = ['random', 'sorted', 'reverse', 'nearly_sorted']
    
    # Graphique 1: Temps d'exécution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, distribution in enumerate(distributions):
        ax = axes[idx]
        
        ax.errorbar(sizes, results['merge'][distribution]['time_mean'], 
                   yerr=results['merge'][distribution]['time_std'],
                   label='Merge Sort', marker='o', linewidth=2, capsize=5)
        ax.errorbar(sizes, results['quick_median'][distribution]['time_mean'], 
                   yerr=results['quick_median'][distribution]['time_std'],
                   label='Quick Sort (median)', marker='s', linewidth=2, capsize=5)
        ax.errorbar(sizes, results['quick_random'][distribution]['time_mean'], 
                   yerr=results['quick_random'][distribution]['time_std'],
                   label='Quick Sort (random)', marker='^', linewidth=2, capsize=5)
        
        ax.set_title(f'Temps d\'exécution - {distribution.capitalize()}', fontweight='bold')
        ax.set_xlabel('Taille du tableau')
        ax.set_ylabel('Temps moyen (secondes)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle('Comparaison des temps d\'exécution (moyenne ± écart-type)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Graphique 2: Nombre de comparaisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, distribution in enumerate(distributions):
        ax = axes[idx]
        
        ax.plot(sizes, results['merge'][distribution]['comparisons_mean'], 
                label='Merge Sort', marker='o', linewidth=2)
        ax.plot(sizes, results['quick_median'][distribution]['comparisons_mean'], 
                label='Quick Sort (median)', marker='s', linewidth=2)
        ax.plot(sizes, results['quick_random'][distribution]['comparisons_mean'], 
                label='Quick Sort (random)', marker='^', linewidth=2)
        
        ax.set_title(f'Nombre de comparaisons - {distribution.capitalize()}', fontweight='bold')
        ax.set_xlabel('Taille du tableau')
        ax.set_ylabel('Nombre moyen de comparaisons')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle('Comparaison du nombre de comparaisons', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_complexity(sizes, results):
    """Analyse comparative avec les complexités théoriques"""
    print("\n" + "="*100)
    print("ANALYSE COMPARATIVE DES COMPLEXITÉS")
    print("="*100)
    
    # Calcul des complexités théoriques
    theoretical_nlogn = [n * np.log2(n) for n in sizes]
    theoretical_n2 = [n * n for n in sizes]
    
    print("\n📊 Analyse des performances par distribution:")
    print("-" * 80)
    
    distributions = ['random', 'sorted', 'reverse', 'nearly_sorted']
    
    for distribution in distributions:
        print(f"\n🔍 Distribution: {distribution.upper()}")
        print(f"{'Taille':<8} {'Merge':<12} {'Quick(med)':<12} {'Quick(rand)':<12} {'n·log(n)':<12} {'n²':<12}")
        print("-" * 80)
        
        for i, size in enumerate(sizes):
            merge_comp = results['merge'][distribution]['comparisons_mean'][i]
            quick_med_comp = results['quick_median'][distribution]['comparisons_mean'][i]
            quick_rand_comp = results['quick_random'][distribution]['comparisons_mean'][i]
            
            print(f"{size:<8} {merge_comp:<12.0f} {quick_med_comp:<12.0f} "
                  f"{quick_rand_comp:<12.0f} {theoretical_nlogn[i]:<12.0f} {theoretical_n2[i]:<12.0f}")

# Paramètres conformes au TP
sizes = [1000, 5000, 10000, 50000]  # Tailles spécifiées dans le TP
distributions = ['random', 'sorted', 'reverse', 'nearly_sorted']
iterations = 30  # Minimum 30 itérations comme demandé

try:
    print("🚀 ÉTUDE COMPARATIVE DES ALGORITHMES DE TRI")
    print("   Fusion vs Rapide - Protocole expérimental complet")
    print(f"   Tailles: {sizes}")
    print(f"   Distributions: {distributions}")
    print(f"   Itérations par test: {iterations}")
    print("-" * 80)
    
    results = run_comprehensive_experiments(sizes, distributions, iterations)
    print("\n✅ Expérimentations terminées avec succès!")
    
    # Analyse des résultats
    analyze_complexity(sizes, results)
    
    # Génération des graphiques
    print("\n📊 Génération des graphiques d'analyse...")
    plot_comprehensive_results(sizes, results)
    print("✅ Analyse graphique terminée!")
    
    # Conclusions
    print("\n" + "="*100)
    print("CONCLUSIONS DE L'ÉTUDE")
    print("="*100)
    print("✅ Algorithmes implémentés: Merge Sort et Quick Sort (2 variantes de pivot)")
    print("✅ Mesures effectuées: Comparaisons, échanges, temps d'exécution")
    print("✅ Distributions testées: Aléatoire, triée, inverse, presque triée")
    print("✅ Analyse statistique: Moyenne et écart-type sur 30 itérations")
    print("✅ Comparaison avec complexités théoriques: O(n log n) et O(n²)")

except Exception as e:
    print(f"❌ Erreur lors de l'exécution: {e}")
    import traceback
    traceback.print_exc()