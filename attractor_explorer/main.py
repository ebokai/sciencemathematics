import pygame
import numpy as np 

import functions 
import slider 
import iterator_naive
from config_vars import *




# Function to clear the right surface and draw the new attractor
def clear_and_draw(parameters):

	surface_right.fill(SURFACE_RIGHT_BG)
	normalized_x, normalized_y = functions.generate_and_normalize(parameters)

	# Draw points representing the attractor (skip transient iterations)
	for x,y in zip(normalized_x[TRANSIENT:], normalized_y[TRANSIENT:]):
		surface_right.set_at((int(YRES*x), int(YRES*y)), PIXEL_COLOR)

	# Return entropy and Fourier transform of attractor
	return functions.compute_grid_stats(normalized_x, normalized_y)



# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((XRES, YRES))
pygame.display.set_caption('Iterated map explorer')
clock = pygame.time.Clock()


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
			click, parameters, active_slider_id = slider.intersect_slider(mouse_position, parameters, active_slider_id)
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
				parameters = functions.find_attractor()
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
	surface_left.blit(PYGAME_FONT.render(message, True, (255, 255, 255)), (32, 32))
	message = 'CLICK SLIDERS TO CHANGE PARAMETERS'
	surface_left.blit(PYGAME_FONT.render(message, True, (255, 255, 255)), (32, 64))
	message = 'PRESS LEFT/RIGHT TO FINE-TUNE ACTIVE PARAMETER'
	surface_left.blit(PYGAME_FONT.render(message, True, (255, 255, 255)), (32, 96))
	message = 'PRESS UP/DOWN TO CHANGE ACTIVE PARAMETER'
	surface_left.blit(PYGAME_FONT.render(message, True, (255, 255, 255)), (32, 128))

	# Draw all sliders for current parameters
	for slider_id in range(12):
		slider.draw_slider(surface_left, parameters[slider_id], slider_id, active_slider_id)	

	# Display entropy and Fourier data (or divergence message)
	if rasterization_entropy > 0:
		message = f'RASTERIZATION ENTROPY: {rasterization_entropy:.2f}'
		fourier_line_spectrum = [(i, YRES - 64 - 200 * f) for i,f in zip(np.linspace(32, XRES-YRES-96, len(fourier_transform_data)), fourier_transform_data[:len(fourier_transform_data)])]
		pygame.draw.lines(surface_left, PIXEL_COLOR, False, fourier_line_spectrum)
	else:
		message = 'ORBIT DIVERGES'

	msg_text = PYGAME_FONT.render(message, True, (255, 255, 255))
	surface_left.blit(msg_text, (32, 160 + (slider_id + 3) * 32))

	# Combine control and visualization panels
	screen.blit(surface_left,(0,0))
	if parameter_change:
		screen.blit(surface_right,(XRES-YRES,0))
		parameter_change = False

	# Update display
	pygame.display.flip()
	clock.tick(60)
