import numpy as np 
from att_exp_params import *

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


def intersect_slider(pos, pars, active_par):
	
	xmin = 32 
	xmax = xmin + SLIDER_WIDTH
	ymin = 160
	ymax = 160 + 12 * 32
	x, y = pos 

	is_click = False

	if x > xmin and x < xmax and y > ymin and y < ymax:
		k = int((y - ymin)/(ymax - ymin) * 12)
		active_par = k
		new_par = -1 + 2 * (x - xmin)/(xmax - xmin)
		pars[k] = new_par
		is_click = True

	return is_click, pars, active_par