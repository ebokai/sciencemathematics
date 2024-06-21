import pygame
import numpy as np 


def f(x, y, pars):

	v = np.array([1,x,y,x*x,x*y,y*y])

	fx = v @ pars[:6]
	fy = v @ pars[6:]

	return fx, fy

def generate_iterates(max_its, pars):

	x = 0
	y = 0
	r = x*x + y*y 
	i = 0 

	Fx, Fy = [], []

	while r < 4 and i < max_its:
		x,y = f(x,y,pars)
		r = x*x + y*y 
		i += 1
		Fx.append(x)
		Fy.append(y)

	return Fx, Fy

def normalize(x, low = 0.2, high = 0.8):

	x = np.array(x)

	x = (x - min(x))/(max(x) - min(x))

	return low + x * (high - low) 

def detect(Fx, Fy):

	grid_size = 10

	grid = np.zeros((grid_size,grid_size))
	for x,y in zip(Fx[TRANSIENT:], Fy[TRANSIENT:]):
		ix = int((x * grid_size) % grid_size)
		iy = int((y * grid_size) % grid_size)
		grid[ix,iy] += 1

	grid = grid / np.sum(grid)

	s = grid * np.log(grid, where = (grid != 0))
	s[np.isinf(s)] = 0
	s[np.isnan(s)] = 0

	return -np.sum(s)


def draw_slider(par, k, active_par):

	var = ['a','x','y','x^2','xy','y^2','b','x','y','x^2','xy','y^2']

	message = f'{par:.3f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (XRES-YRES-64, 160 + k * 32))

	x = 32
	y = 160 + k * 32
	w = abs(par) * SLIDER_WIDTH/2
	h = 28 

	# slider background
	if k == active_par:
		c = (55, 85, 55)
	else:
		c = (55, 55, 55)
	pygame.draw.rect(surface_left, c, (x, y, SLIDER_WIDTH, h))

	# slider value
	if par < 0:
		x = XMID + par * SLIDER_WIDTH/2
		c = (255, 55, 55)
	else:
		c = (55, 55, 255)
		x = XMID
	pygame.draw.rect(surface_left, c, (x, y, w ,h))

	message = var[k]
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, y))

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

def clear_and_draw(pars):
	surface_right.fill((20,20,20,255))
	Fx, Fy = generate_iterates(MAX_ITS, pars)
	Fnx = normalize(Fx)
	Fny = normalize(Fy)
	for x,y in zip(Fnx[TRANSIENT:], Fny[TRANSIENT:]):
		pygame.draw.circle(surface_right, (255, 255, 255, 128), (YRES*x,YRES*y), 1)
	return detect(Fnx, Fny), fourier_grid(Fnx, Fny)


def fourier_grid(Fx, Fy):

	grid_size = 20
	grid = np.zeros((grid_size,grid_size))
	for x,y in zip(Fx[TRANSIENT:], Fy[TRANSIENT:]):
		ix = int((x * grid_size) % grid_size)
		iy = int((y * grid_size) % grid_size)
		grid[ix,iy] += 1

	grid = grid / np.sum(grid)
	grid -= np.mean(grid)

	ft = np.fft.fft2(grid)
	ft_a = np.abs(ft.flatten())

	return ft_a

# GLOBAL
XRES, YRES = 1600, 900
MAX_ITS = 2500
TRANSIENT = 100

# GUI STUFF
SLIDER_WIDTH = (XRES-YRES)-128
XMID = 32 + SLIDER_WIDTH/2

pygame.init()
screen = pygame.display.set_mode((XRES, YRES))
pygame.display.set_caption('Iterated map explorer')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 18)

surface_left = pygame.Surface((XRES-YRES, YRES), pygame.SRCALPHA)
surface_left.fill((0,0,0,255))
surface_right = pygame.Surface((YRES, YRES), pygame.SRCALPHA)
surface_right.fill((20,20,20,255))

pars = np.random.uniform(-1,1,12)
active_par = 0
s = 0

ft = np.zeros(20)

while True:

	surface_left.fill((0,0,0,255))

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()

		elif event.type == pygame.MOUSEBUTTONDOWN:
			pos = pygame.mouse.get_pos()
			click, pars, active_par = intersect_slider(pos, pars, active_par)
			if click:
				s, ft = clear_and_draw(pars)

		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				pygame.quit()
				exit()
			if event.key == pygame.K_UP:
				active_par -= 1
				active_par %= 12
			elif event.key == pygame.K_DOWN:
				active_par += 1
				active_par %= 12

			if event.key == pygame.K_r:
				surface_right.fill((20,20,20,255))
				Fx = []
				s = 0
				while s < 2:
					pars = np.random.uniform(-1,1,12)
					Fx, Fy = generate_iterates(MAX_ITS, pars)
					if len(Fx) < MAX_ITS:
						continue
					Fnx = normalize(Fx)
					Fny = normalize(Fy)
					s = detect(Fnx, Fny)
				ft = fourier_grid(Fnx, Fny)
				
				for x,y in zip(Fnx[TRANSIENT:], Fny[TRANSIENT:]):

					pygame.draw.circle(surface_right, (255, 255, 255, 128), (YRES*x,YRES*y), 1)

	keys = pygame.key.get_pressed()
	if keys[pygame.K_RIGHT]:
		if pars[active_par] <= 0.999:
			pars[active_par] += 0.001
			s, ft = clear_and_draw(pars)
	if keys[pygame.K_LEFT]:
		if pars[active_par] >= -0.999:
			pars[active_par] -= 0.001
			s, ft = clear_and_draw(pars)

	message = f'PRESS R TO FIND NEW ATTRACTOR'
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, 32))

	message = f'CLICK SLIDERS TO CHANGE PARAMETERS'
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, 64))

	message = f'PRESS LEFT/RIGHT TO FINE-TUNE ACTIVE PARAMETER'
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, 96))

	message = f'PRESS UP/DOWN TO CHANGE ACTIVE PARAMETER'
	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, 128))

	for k in range(12):
		draw_slider(pars[k],k,active_par)	

	if s > 0:
		message = f'RASTERIZATION ENTROPY: {s:.2f}'
		ft_a = [(i, YRES - 64 - 200 * f) for i,f in zip(np.linspace(0,XRES-YRES,len(ft)),ft[:len(ft)])]
		pygame.draw.lines(surface_left, (155, 155, 155), False, ft_a)
	else:
		message = 'ORBIT DIVERGES'

	msg_text = font.render(message, False, (255, 255, 255))
	surface_left.blit(msg_text, (32, 160 + (k + 2) * 32))

	screen.blit(surface_left,(0,0))
	screen.blit(surface_right,(XRES-YRES,0))
	pygame.display.flip()
	clock.tick(60)