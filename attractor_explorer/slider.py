from config_vars import * 
import pygame

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

# Function to draw a single parameter slider_id (both background and value)
def draw_slider(slider_surface, slider_value, slider_id, active_slider_id):
    # Parameter names
    labels = ['a','x','y','x²','xy','y²','b','x','y','x²','xy','y²']
    
    # Geometry
    x = 32
    y = 160 + slider_id * 36  # slightly more spacing
    slider_width = SLIDER_WIDTH
    slider_height = 28
    mid_x = XMID
    
    # Background bar
    pygame.draw.rect(slider_surface, SLIDER_BG_COLOR, (x, y, slider_width, slider_height), border_radius=6)
    
    # Active highlight glow
    if slider_id == active_slider_id:
        pygame.draw.rect(slider_surface, ACTIVE_SLIDER_COLOR, (x-2, y-2, slider_width+4, slider_height+4), border_radius=8)
    
    # Fill based on slider_value
    fill_width = int(abs(slider_value) * slider_width/2)
    if slider_value >= 0:
        fill_color = SLIDER_POS_COLOR
        fill_rect = (mid_x, y, fill_width, slider_height)
    else:
        fill_color = SLIDER_NEG_COLOR
        fill_rect = (mid_x + slider_value * slider_width/2, y, fill_width, slider_height)
    
    pygame.draw.rect(slider_surface, fill_color, fill_rect, border_radius=6)
    
    # Draw the label (parameter symbol) on left
    label_text = PYGAME_FONT.render(labels[slider_id], True, SLIDER_FONT_COLOR)
    slider_surface.blit(label_text, (x + 4, y + 4))
    
    # Draw the numeric slider_value on right
    value_text = PYGAME_FONT.render(f"{slider_value:.3f}", True, SLIDER_FONT_COLOR)
    slider_surface.blit(value_text, (XRES - YRES - 70, y + 4))