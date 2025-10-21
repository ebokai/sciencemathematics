import numpy as np 
import pygame
import functions
import slider
import text
from config_vars import *

class AppState():

	def __init__(self):
		self.parameters = np.random.uniform(-1, 1, 12)
		self.active_slider_id = 0
		self.rasterization_entropy = 0
		self.n_fourier = 400
		self.fourier_transform_data = np.zeros(self.n_fourier)

		self.parameter_change = True
		self.clicked = False

		self.surface_left = pygame.Surface((XRES-YRES, YRES), pygame.SRCALPHA)
		self.surface_right = pygame.Surface((YRES, YRES), pygame.SRCALPHA)

		self.surface_left.fill(SURFACE_LEFT_BG)
		self.surface_right.fill(SURFACE_RIGHT_BG)

	def init_frame(self):
		self.surface_left.fill(SURFACE_LEFT_BG)

	def handle_event(self, event):
		if event.type == pygame.QUIT:
			self.quit()

		elif event.type == pygame.MOUSEBUTTONDOWN:
			self.handle_mousedown()

		elif event.type == pygame.KEYDOWN:
			self.update_active_slider(event)

			if event.key == pygame.K_ESCAPE:
				self.quit()

			if event.key == pygame.K_r:
				self.generate_new_attractor()

	def quit(self):
		pygame.quit()
		exit()

	def handle_mousedown(self):
		mouse_position = pygame.mouse.get_pos()
		self.clicked, self.parameters, self.active_slider_id = slider.intersect_slider(mouse_position, self.parameters, self.active_slider_id)
		if self.clicked:
			self.clear_and_draw()

	def update_active_slider(self, event):
		slider_delta = -1 * (event.key == pygame.K_UP) + 1 * (event.key == pygame.K_DOWN)
		self.active_slider_id = (self.active_slider_id + slider_delta) % 12

	def update_parameter(self):
		keys = pygame.key.get_pressed()
		delta = -0.001 * (keys[pygame.K_LEFT]) + 0.001 * (keys[pygame.K_RIGHT])
		self.parameters[self.active_slider_id] = np.clip(self.parameters[self.active_slider_id] + delta, -1, 1)
		self.clear_and_draw()

	def generate_new_attractor(self):
		self.parameters = functions.find_attractor()
		self.clear_and_draw()

	def clear_and_draw(self):
		self.surface_right.fill(SURFACE_RIGHT_BG)
		normalized_x, normalized_y = functions.generate_and_normalize(self.parameters)
		for x,y in zip(normalized_x[TRANSIENT:], normalized_y[TRANSIENT:]):
			self.surface_right.set_at((int(YRES*x), int(YRES*y)), PIXEL_COLOR)
		self.rasterization_entropy, self.fourier_transform_data = functions.compute_grid_stats(normalized_x, normalized_y)
		self.parameter_change = True

	def draw_instructions(self):
		text.instruction_text(self.surface_left)

	def draw_sliders(self):
		for slider_id in range(12):
			slider.draw_slider(self.surface_left, self.parameters[slider_id], slider_id, self.active_slider_id)

	def draw_fourier_spectrum(self):
		message = 'ORBIT DIVERGES'
		if self.rasterization_entropy > 0:
			message = f'RASTERIZATION ENTROPY: {self.rasterization_entropy:.2f}'
			X = np.linspace(32, XRES - YRES - 96, self.n_fourier)
			Y = self.fourier_transform_data[:self.n_fourier]
			fourier_line_spectrum = [(x, YRES - 64 - 200 * y) for x,y in zip(X,Y)]
			pygame.draw.lines(self.surface_left, PIXEL_COLOR, False, fourier_line_spectrum)
		message_render = PYGAME_FONT.render(message, True, TEXT_COLOR)
		self.surface_left.blit(message_render, (32, 160 + 14 * 32))

	def blit_surfaces(self, screen):
		screen.blit(self.surface_left, (0,0))
		if self.parameter_change:
			screen.blit(self.surface_right, (XRES-YRES,0))
			self.parameter_change = False








