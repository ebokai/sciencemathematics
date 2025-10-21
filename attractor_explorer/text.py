from config_vars import *

def instruction_text(surface):
	# --- UI text instructions ---
	message = 'PRESS R TO FIND NEW ATTRACTOR'
	surface.blit(PYGAME_FONT.render(message, True, TEXT_COLOR), (32, 32))
	message = 'CLICK SLIDERS TO CHANGE PARAMETERS'
	surface.blit(PYGAME_FONT.render(message, True, TEXT_COLOR), (32, 64))
	message = 'PRESS LEFT/RIGHT TO FINE-TUNE ACTIVE PARAMETER'
	surface.blit(PYGAME_FONT.render(message, True, TEXT_COLOR), (32, 96))
	message = 'PRESS UP/DOWN TO CHANGE ACTIVE PARAMETER'
	surface.blit(PYGAME_FONT.render(message, True, TEXT_COLOR), (32, 128))