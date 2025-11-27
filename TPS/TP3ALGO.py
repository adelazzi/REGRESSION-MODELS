import random
import time
import statistics

# =====================================================
#                 TRI PAR SÉLECTION
# =====================================================
# Algorithme : trouve le minimum dans la partie non triée
# puis l'échange avec la position actuelle.
def selection_sort(arr):
    n = len(arr)
    comp = 0       # compteur de comparaisons
    moves = 0      # compteur de déplacements / échanges

    for i in range(n):
        min_idx = i
        # Recherche du plus petit élément dans la partie non triée
        for j in range(i+1, n):
            comp += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Échange avec l’élément en position i
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        moves += 1

    return comp, moves


# =====================================================
#                     TRI À BULLES
# =====================================================
# Compare les éléments adjacents et les échange s’ils sont
# dans le mauvais ordre. Répété jusqu'à tri complet.
def bubble_sort(arr):
    n = len(arr)
    comp = 0
    moves = 0

    for i in range(n):
        for j in range(0, n-i-1):
            comp += 1
            # Si la paire est mal ordonnée → échange
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                moves += 1

    return comp, moves


# =====================================================
#     TRI PAR INSERTION — VERSION PAR ÉCHANGES
# =====================================================
# Insertion utilisant des échanges successifs.
# Moins efficace que la version avec déplacements.
def insertion_sort_swap(arr):
    comp = 0
    moves = 0
    n = len(arr)

    for i in range(1, n):
        j = i
        # Tant que l’élément est plus petit que le précédent → échanger
        while j > 0:
            comp += 1
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
                moves += 1
                j -= 1
            else:
                break

    return comp, moves


# =====================================================
#    TRI PAR INSERTION — VERSION PAR DÉPLACEMENTS
# =====================================================
# Version optimisée : on décale les éléments plus grands,
# puis on insère la clé directement à sa bonne position.
def insertion_sort_shift(arr):
    comp = 0
    moves = 0
    n = len(arr)

    for i in range(1, n):
        key = arr[i]      # élément à insérer
        j = i - 1

        # Déplacement des éléments plus grands vers la droite
        while j >= 0:
            comp += 1
            if arr[j] > key:
                arr[j+1] = arr[j]
                moves += 1
                j -= 1
            else:
                break

        # Insertion de l'élément à sa position exacte
        arr[j+1] = key
        moves += 1

    return comp, moves


# =====================================================
#        GÉNÉRATION DES TABLEAUX DE TEST
# =====================================================
# Trois types :
# - random    : valeurs aléatoires
# - sorted    : déjà trié croissant
# - reversed  : trié décroissant (pire cas)
def generate_array(n, typ):
    if typ == "random":
        return [random.randint(0, 1000000) for _ in range(n)]
    if typ == "sorted":
        return list(range(n))
    if typ == "reversed":
        return list(range(n, 0, -1))


# =====================================================
#   EXÉCUTER UN ALGORITHME 30 FOIS (MOYENNE)
# =====================================================
# Permet d’obtenir des résultats plus fiables.
def test_algo(algo, n, typ):
    comp_list = []
    moves_list = []
    time_list = []

    for _ in range(30):
        arr = generate_array(n, typ)
        start = time.perf_counter()   # début chronomètre
        comp, moves = algo(arr)       # exécution
        end = time.perf_counter()     # fin chronomètre

        comp_list.append(comp)
        moves_list.append(moves)
        time_list.append(end - start)

    # Retourne la moyenne des 30 exécutions
    return (
        statistics.mean(comp_list),
        statistics.mean(moves_list),
        statistics.mean(time_list),
    )


# =====================================================
#                   EXPÉRIMENTATION
# =====================================================
# Liste des algorithmes
algorithms = {
    "Selection": selection_sort,
    "Bubble": bubble_sort,
    "Insertion-swaps": insertion_sort_swap,
    "Insertion-shifts": insertion_sort_shift,
}

# Tailles à tester
sizes = [10, 100, 500, 700]

# Types de tableaux
types = ["random", "sorted", "reversed"]

results = []

# Exécution de toutes les combinaisons avec progress printing
for name, algo in algorithms.items():
    for n in sizes:
        for typ in types:
            comp, moves, t = test_algo(algo, n, typ)
            results.append((name, n, typ, comp, moves, t))
            print(f"Completed: {name}, Size: {n}, Type: {typ} -> Comparisons: {comp:.0f}, Moves: {moves:.0f}, Time: {t:.4f}s")

# =====================================================
#                AFFICHAGE DES RÉSULTATS
# =====================================================
from tabulate import tabulate

# Define headers for the table columns
headers = ["Algorithm", "Size", "Type", "Comparisons", "Moves", "Time (s)"]

# Use tabulate to format the results list into a nice table
print("\n=== RÉSULTATS ===")
print(tabulate(results, headers=headers, floatfmt=".2f", tablefmt="grid"))


# import csv

# # Define headers to write in CSV file
# headers = ["Algorithm", "Size", "Type", "Comparisons", "Moves", "Time (s)"]

# with open("results.csv", mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)  # Write the header row
#     writer.writerows(results)  # Write all results rows

# print("Exported results to results.csv")




import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig)

# First column, top plot (row 0, col 0)
ax1 = fig.add_subplot(gs[0, 0])
# First column, bottom plot (row 1, col 0)
ax2 = fig.add_subplot(gs[1, 0])
# Second column, spans both rows (rows 0 and 1, col 1)
ax3 = fig.add_subplot(gs[:, 1])

types_list = ["random", "sorted", "reversed"]
algos = list(algorithms.keys())

# Plot top-left graph (random arrays)
for name in algos:
    x_vals, y_vals = [], []
    for r in results:
        if r[0] == name and r[2] == types_list[0]:
            x_vals.append(r[1])
            y_vals.append(r[5])
    ax1.plot(x_vals, y_vals, marker='o', label=name)
ax1.set_title(f"Time by Input Size ({types_list[0]} arrays)")
ax1.set_ylabel("Average Time (s)")
ax1.grid(True)
ax1.legend(loc='upper left')

# Plot bottom-left graph (sorted arrays)
for name in algos:
    x_vals, y_vals = [], []
    for r in results:
        if r[0] == name and r[2] == types_list[1]:
            x_vals.append(r[1])
            y_vals.append(r[5])
    ax2.plot(x_vals, y_vals, marker='o', label=name)
ax2.set_title(f"Time by Input Size ({types_list[1]} arrays)")
ax2.set_xlabel("Input Size")
ax2.set_ylabel("Average Time (s)")
ax2.grid(True)
ax2.legend(loc='upper left')

# Plot right column graph spanning two rows (reversed arrays)
for name in algos:
    x_vals, y_vals = [], []
    for r in results:
        if r[0] == name and r[2] == types_list[2]:
            x_vals.append(r[1])
            y_vals.append(r[5])
    ax3.plot(x_vals, y_vals, marker='o', label=name)
ax3.set_title(f"Time by Input Size ({types_list[2]} arrays)")
ax3.set_xlabel("Input Size")
ax3.set_ylabel("Average Time (s)")
ax3.grid(True)
ax3.legend(loc='upper left')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), constrained_layout=True)

types_list = ["random", "sorted", "reversed"]
algos = list(algorithms.keys())
metrics = {"Comparisons": 3, "Moves": 4, "Time": 5}  # indexes in results tuple

for col, typ in enumerate(types_list):
    for row, (metric, idx) in enumerate(metrics.items()):
        ax = axs[row, col]
        for name in algos:
            x_vals = [r[1] for r in results if r[0] == name and r[2] == typ]
            y_vals = [r[idx] for r in results if r[0] == name and r[2] == typ]
            ax.plot(x_vals, y_vals, marker='o', label=name)
        ax.set_title(f"{metric} on {typ} arrays")
        if row == 2:
            ax.set_xlabel("Input Size")
        ax.set_ylabel(f"Average {metric}" if col == 0 else "")
        ax.grid(True)
        if row == 0 and col == 2:
            ax.legend(loc='upper left', fontsize='small')

plt.show()
