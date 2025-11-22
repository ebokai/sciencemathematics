import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from tqdm import tqdm


@njit(parallel=True)
def reproduce_numba(bacteria_matrix, p_reproduce=0.002, p_mutate=0.01):

    for i in prange(simulation_size):
        for j in prange(simulation_size):
            occupied = bacteria_matrix[i, j, 0]
            mic = bacteria_matrix[i, j, 1]
            n_mutations = bacteria_matrix[i, j, 2]
            if not occupied:
                continue

            new_i = (i + np.random.randint(-1, 2)) % simulation_size
            new_j = (j + np.random.randint(0, 2)) % simulation_size

            if not (
                new_i >= 0
                and new_i < simulation_size
                and new_j >= 0
                and new_j < simulation_size
            ):
                continue

            new_occupied = bacteria_matrix[new_i, new_j, 0]

            if not new_occupied and np.random.random() < p_reproduce:
                bacteria_matrix[new_i, new_j, 0] = 1

                if np.random.random() < p_mutate:
                    mic *= 10
                    n_mutations += 1

                bacteria_matrix[new_i, new_j, 1] = mic
                bacteria_matrix[new_i, new_j, 2] = n_mutations

    return bacteria_matrix


@njit(parallel=True)
def kill_numba(bacteria_matrix, antibiotic_matrix):
    for i in prange(simulation_size):
        for j in prange(simulation_size):
            occupied = bacteria_matrix[i, j, 0]
            mic = bacteria_matrix[i, j, 1]
            if not occupied:
                continue
            if mic < antibiotic_matrix[i, j]:
                bacteria_matrix[i, j, 0] = 0
                bacteria_matrix[i, j, 1] = 0
                bacteria_matrix[i, j, 2] = 0
    return bacteria_matrix


def initialize_ab_matrix(regions=10, ab0=1):
    antibiotic_matrix = np.zeros((simulation_size, simulation_size))
    x = np.arange(simulation_size)
    for i in range(simulation_size):
        antibiotic_matrix[i, :] = ab0 * np.exp((x - 10) / 10)
    return antibiotic_matrix


def initialize_bacteria(n):
    if n > simulation_size:
        n = simulation_size

    bacteria_matrix = np.zeros((simulation_size, simulation_size, 3), dtype=float)

    i = np.random.choice(np.arange(simulation_size), n, replace=False)
    bacteria_matrix[i, 0, 0] = 1
    bacteria_matrix[i, 0, 1] = 1
    bacteria_matrix[i, 0, 2] = 0

    return bacteria_matrix

    pairs, bacteria = [], []
    for _ in range(n):
        i = np.random.randint(0, simulation_size)
        pair = (i, 0)
        if pair not in pairs:
            pairs.append(pair)
            bacteria.append(Bacterium(i, 0, 1))
    return pairs, bacteria


# --- INITIALIZE ---
simulation_size = 150
iterations = 10000

bacteria_matrix = initialize_bacteria(50)
antibiotic_matrix = initialize_ab_matrix()

# --- SIMULATION ---
for t in tqdm(range(iterations)):
    bacteria_matrix = reproduce_numba(bacteria_matrix)
    bacteria_matrix = kill_numba(bacteria_matrix, antibiotic_matrix)

# ------------------

y, x = np.where(bacteria_matrix[:, :, 0] > 0)
colors = bacteria_matrix[y, x, 2]


# --- PLOT ---
plt.matshow(np.log10(antibiotic_matrix), fignum=0, cmap="gray_r")
plt.colorbar(label="antibiotic concentration [log mg/L]")
plt.scatter(x, y, c=colors, cmap="autumn", edgecolor="None", s=10)
plt.colorbar(label="number of mutations")
plt.show()
# ------------------
