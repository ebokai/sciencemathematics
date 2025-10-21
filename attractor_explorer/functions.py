import numpy as np 
import iterator_naive
from config_vars import *


def normalize(x, low = 0.2, high = 0.8):

	x = np.array(x)

	x = (x - min(x))/(max(x) - min(x))

	return low + x * (high - low) 
	

def compute_grid_stats(Fx, Fy, grid_size=20, compute_ft=True):
    """
    Compute both entropy and Fourier magnitude from iterates.
    
    Parameters:
        Fx, Fy : array-like floats in [0,1]
        grid_size : int, size of the grid
        compute_ft : bool, whether to compute Fourier magnitude
    
    Returns:
        entropy : float
        ft_a : 1D array of Fourier magnitudes (or None if compute_ft=False)
    """
    # Map points to grid indices
    ix = np.floor(Fx[TRANSIENT:] * grid_size).astype(int) % grid_size
    iy = np.floor(Fy[TRANSIENT:] * grid_size).astype(int) % grid_size
    
    # Build grid using bincount for speed
    grid = np.zeros((grid_size, grid_size))
    np.add.at(grid, (ix, iy), 1)
    
    # Normalize
    grid = grid / np.sum(grid)
    
    # Entropy calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_grid = grid * np.log(grid)
        entropy_grid[np.isnan(entropy_grid)] = 0
        entropy_grid[np.isinf(entropy_grid)] = 0
    entropy = -np.sum(entropy_grid)
    
    # Fourier magnitude
    ft_a = None
    if compute_ft:
        grid_centered = grid - np.mean(grid)
        ft = np.fft.fft2(grid_centered)
        ft_a = np.abs(ft.flatten())
    
    return entropy, ft_a

def generate_and_normalize(parameters):
    # Generate iterates of the system given current parameters
    x_iterates, y_iterates = iterator_naive.generate_iterates(MAX_ITS, parameters)

    # Normalize coordinates to fit in the plotting area
    normalized_x = normalize(x_iterates)
    normalized_y = normalize(y_iterates)

    return normalized_x, normalized_y

def find_attractor():
    rasterization_entropy = 0
    while rasterization_entropy < 3:
        parameters = np.random.uniform(-1,1,12)
        x_iterates, y_iterates = iterator_naive.generate_iterates(MAX_ITS, parameters)
        if len(x_iterates) < MAX_ITS:
            continue
        normalized_x = normalize(x_iterates)
        normalized_y = normalize(y_iterates)
        rasterization_entropy, _ = compute_grid_stats(normalized_x, normalized_y, compute_ft = False)
    return parameters


