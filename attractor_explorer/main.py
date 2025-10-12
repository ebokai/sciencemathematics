import pygame
import numpy as np 

import att_exp_functions
import att_exp_iterator_naive
from att_exp_params import *


# Function to draw a single parameter slider_id (both background and value)
def draw_slider(value, slider_id, active_slider_id):
    # Parameter names
    labels = ['a','x','y','x²','xy','y²','b','x','y','x²','xy','y²']
    
    # Geometry
    x = 32
    y = 160 + slider_id * 36  # slightly more spacing
    w = SLIDER_WIDTH
    h = 28
    mid_x = XMID
    
    # Background bar
    bg_color = (40, 40, 40)
    pygame.draw.rect(surface_left, bg_color, (x, y, w, h), border_radius=6)
    
    # Active highlight glow
    if slider_id == active_slider_id:
        glow_color = (80, 180, 80)
        pygame.draw.rect(surface_left, glow_color, (x-2, y-2, w+4, h+4), border_radius=8)
    
    # Fill based on value
    fill_w = int(abs(value) * w/2)
    if value >= 0:
        fill_color = (50, 150, 255)
        fill_rect = (mid_x, y, fill_w, h)
    else:
        fill_color = (255, 80, 80)
        fill_rect = (mid_x + value * w/2, y, fill_w, h)
    
    pygame.draw.rect(surface_left, fill_color, fill_rect, border_radius=6)
    
    # Draw the label (parameter symbol) on left
    label_text = font.render(labels[slider_id], True, (230, 230, 230))
    surface_left.blit(label_text, (x + 4, y + 4))
    
    # Draw the numeric value on right
    value_text = font.render(f"{value:.3f}", True, (230, 230, 230))
    surface_left.blit(value_text, (XRES - YRES - 70, y + 4))



# Function to clear the right surface and draw the new attractor
def clear_and_draw(parameters):

	surface_right.fill(SURFACE_RIGHT_BG)
	normalized_x, normalized_y = generate_and_normalize(parameters)

	# Draw points representing the attractor (skip transient iterations)
	for x,y in zip(normalized_x[TRANSIENT:], normalized_y[TRANSIENT:]):
		surface_right.set_at((int(YRES*x), int(YRES*y)), PIXEL_COLOR)

	# Return entropy and Fourier transform of attractor
	return att_exp_functions.compute_grid_stats(normalized_x, normalized_y)


def generate_and_normalize(parameters):
	# Generate iterates of the system given current parameters
	x_iterates, y_iterates = att_exp_iterator_naive.generate_iterates(MAX_ITS, parameters)

	# Normalize coordinates to fit in the plotting area
	normalized_x = att_exp_functions.normalize(x_iterates)
	normalized_y = att_exp_functions.normalize(y_iterates)

	return normalized_x, normalized_y

def find_attractor():
	rasterization_entropy = 0
	while rasterization_entropy < 3:
		parameters = np.random.uniform(-1,1,12)
		x_iterates, y_iterates = att_exp_iterator_naive.generate_iterates(MAX_ITS, parameters)
		if len(x_iterates) < MAX_ITS:
			continue
		normalized_x = att_exp_functions.normalize(x_iterates)
		normalized_y = att_exp_functions.normalize(y_iterates)
		rasterization_entropy, _ = att_exp_functions.compute_grid_stats(normalized_x, normalized_y, compute_ft = False)
	return parameters


# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((XRES, YRES))
pygame.display.set_caption('Iterated map explorer')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 18)

# Two surfaces: left (controls) and right (visualization)
surface_left = pygame.Surface((XRES-YRES, YRES), pygame.SRCALPHA)
surface_left.fill(SURFACE_LEFT_BG)
surface_right = pygame.Surface((YRES, YRES), pygame.SRCALPHA)
surface_right.fill(SURFACE_RIGHT_BG)

# Initialize random parameters
parameters = np.random.uniform(-1,1,12)
active_slider_id = 0
rasterization_entropy = 0

# Placeholder for Fourier transform data
fourier_transform_data = np.zeros(20)

# Keep track of parameter changes
parameter_change = True

# --- Main loop ---
while True:

	surface_left.fill(SURFACE_LEFT_BG)


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()

		# Handle mouse clicks (for slider_id adjustment)
		elif event.type == pygame.MOUSEBUTTONDOWN:
			mouse_position = pygame.mouse.get_pos()
			click, parameters, active_slider_id = att_exp_functions.intersect_slider(mouse_position, parameters, active_slider_id)
			if click:
				rasterization_entropy, fourier_transform_data = clear_and_draw(parameters)
				parameter_change = True

		# Handle keyboard input
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				pygame.quit()
				exit()

			# Navigate between sliders
			slider_delta = -1 if event.key == pygame.K_UP else (+1 if event.key == pygame.K_DOWN else 0)
			active_slider_id = (active_slider_id + slider_delta) % 12

			# Generate a new random attractor (until it doesn't diverge)
			if event.key == pygame.K_r:
				surface_right.fill(SURFACE_RIGHT_BG)
				parameters = find_attractor()
				rasterization_entropy, fourier_transform_data = clear_and_draw(parameters)
				parameter_change = True



	# Fine-tune active parameter with arrow keys
	keys = pygame.key.get_pressed()
	delta = 0.001 if keys[pygame.K_RIGHT] else (-0.001 if keys[pygame.K_LEFT] else 0)
	if delta != 0:
		parameters[active_slider_id] = np.clip(parameters[active_slider_id] + delta, -1, 1)
		rasterization_entropy, fourier_transform_data = clear_and_draw(parameters)
		parameter_change = True
		


	# --- UI text instructions ---
	message = 'PRESS R TO FIND NEW ATTRACTOR'
	surface_left.blit(font.render(message, True, (255, 255, 255)), (32, 32))
	message = 'CLICK SLIDERS TO CHANGE PARAMETERS'
	surface_left.blit(font.render(message, True, (255, 255, 255)), (32, 64))
	message = 'PRESS LEFT/RIGHT TO FINE-TUNE ACTIVE PARAMETER'
	surface_left.blit(font.render(message, True, (255, 255, 255)), (32, 96))
	message = 'PRESS UP/DOWN TO CHANGE ACTIVE PARAMETER'
	surface_left.blit(font.render(message, True, (255, 255, 255)), (32, 128))

	# Draw all sliders for current parameters
	for slider_id in range(12):
		draw_slider(parameters[slider_id], slider_id, active_slider_id)	

	# Display entropy and Fourier data (or divergence message)
	if rasterization_entropy > 0:
		message = f'RASTERIZATION ENTROPY: {rasterization_entropy:.2f}'
		fourier_line_spectrum = [(i, YRES - 64 - 200 * f) for i,f in zip(np.linspace(32, XRES-YRES-96, len(fourier_transform_data)), fourier_transform_data[:len(fourier_transform_data)])]
		pygame.draw.lines(surface_left, PIXEL_COLOR, False, fourier_line_spectrum)
	else:
		message = 'ORBIT DIVERGES'

	msg_text = font.render(message, True, (255, 255, 255))
	surface_left.blit(msg_text, (32, 160 + (slider_id + 3) * 32))

	# Combine control and visualization panels
	screen.blit(surface_left,(0,0))
	if parameter_change:
		screen.blit(surface_right,(XRES-YRES,0))
		parameter_change = False

	# Update display
	pygame.display.flip()
	clock.tick(60)
