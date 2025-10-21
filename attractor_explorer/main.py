import pygame
import numpy as np 

import app_state
import functions 
import text
import slider 
import iterator_naive
from config_vars import *

# --- Pygame setup ---
pygame.init()
screen = pygame.display.set_mode((XRES, YRES))
pygame.display.set_caption('Iterated map explorer')
clock = pygame.time.Clock()

current_state = app_state.AppState()

# --- Main loop ---
while True:

	current_state.init_frame()

	for event in pygame.event.get():
		current_state.handle_event(event)

	current_state.update_parameter()
	current_state.draw_instructions()
	current_state.draw_sliders()
	current_state.draw_fourier_spectrum()
	current_state.blit_surfaces(screen)

	pygame.display.flip()
	clock.tick(60)
