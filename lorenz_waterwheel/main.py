import pygame 
import numpy as np 
import time

from config import *

class Bucket():
	def __init__(self, angle):
		self.mass = 0 # np.random.uniform(MIN_MASS,MAX_MASS)
		self.angle = angle
		self.size = BUCKET_SIZE
		self.fa = 0
		self.fb = 0
		self.update_pos()

	def update_pos(self):
		self.x = YRES/2 + RADIUS * np.cos(self.angle)
		self.y = YRES/2 + RADIUS * np.sin(self.angle)

	def get_torque(self):
		f_gravity = self.mass * GRAVITY
		torque = -SMALL_RADIUS * f_gravity * np.cos(self.angle)
		return torque

	def draw_bucket(self, surface):
		x_corner = self.x - self.size/2
		y_corner = self.y - self.size/2

		bucket_surface = pygame.Surface((self.size, self.size))
		bucket_surface.fill((200,200,255))
		surface.blit(bucket_surface, (x_corner, y_corner))

		bucket_surface = pygame.Surface((self.size - 4, self.size - 4))
		bucket_surface.fill((2,2,20))
		surface.blit(bucket_surface, (x_corner + 2, y_corner + 2))

		bucket_surface = pygame.Surface((self.size - 4, max(1,self.mass/MAX_MASS * self.size - 4)))
		R = self.mass/MAX_MASS * 255
		bucket_surface.fill((int(R),38,255))
		surface.blit(bucket_surface, (x_corner + 2, y_corner + 2 + self.size * (1-self.mass/MAX_MASS)))

	def leak_and_fill(self, omega):

		if self.mass > MIN_MASS:
			self.mass -= LEAK_RATE * self.mass/MAX_MASS * DT

			if self.mass < MIN_MASS:
				self.mass = MIN_MASS

		c1 = self.x + self.size > YRES/2 - RADIUS/FILL_WIDTH
		c2 = self.x < YRES/2 + RADIUS/FILL_WIDTH
		c3 = self.y < YRES/2
		condition = c1 * c2 * c3

		if (condition):
			self.mass += FILL_RATE / (1 + abs(omega)) * DT
			if self.mass > MAX_MASS:
				self.mass = MAX_MASS

		self.fa = self.mass * np.cos(self.angle * MODE)
		self.fb = self.mass * np.sin(self.angle * MODE)

class Wheel():
	def __init__(self):

		self.moment = M0

		self.angle = 0
		self.omega = 0
		
		self.d_angle = 0
		self.d_omega = 0

		self.torque = 0
		self.friction = 0

		self.omega_history = np.zeros(HISTORY)

	def rotate(self, buckets):

		self.angle += self.d_angle

		self.torque = 0
		self.moment = M0

		for bucket in buckets:
			bucket.angle += self.d_angle
			bucket.update_pos()
			self.torque += bucket.get_torque()
			self.moment += bucket.mass

		self.moment *= SMALL_RADIUS * SMALL_RADIUS
		self.friction = -FRICTION * self.omega

		if FRAME % CAPTURE_INTERVAL == 0:
			self.omega_history = np.roll(self.omega_history, 1)
			self.omega_history[0] = self.omega

		self.d_omega = DT * (self.torque + self.friction) / self.moment
		self.omega += self.d_omega
		self.d_angle = DT * self.omega

def transform_x(X, mid):
	minX = min(X)
	maxX = max(X)
	Zx = (X-minX)/(maxX - minX)
	XX = mid + 200 * (-1 + 2 * Zx)
	return XX


FRAME = 0


# ------------


pygame.init()
screen = pygame.display.set_mode((XRES, YRES))
pygame.display.set_caption('Window')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)

surface = pygame.Surface((XRES, YRES), pygame.SRCALPHA)
surface.fill((0,0,0,255))

wheel = Wheel()
buckets = [Bucket(2*np.pi*i/N_BUCKETS) for i in range(N_BUCKETS)]

FOURIER = np.zeros((HISTORY, 2))

start = time.perf_counter()

while True:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			exit()

	surface.fill((0,0,0,255))

	x1 = YRES/2 - RADIUS/FILL_WIDTH
	pygame.draw.rect(surface, (44,44,99), (x1, 0, 2*RADIUS/FILL_WIDTH, YRES))

	wheel.rotate(buckets)
	for bucket in buckets:
		bucket.leak_and_fill(wheel.omega)
		bucket.draw_bucket(surface)

	if FRAME % CAPTURE_INTERVAL == 0:
		FA = sum([bucket.fa for bucket in buckets])
		FB = sum([bucket.fb for bucket in buckets])
		FOURIER = np.roll(FOURIER, 1, axis = 0)
		FOURIER[0,:] = FA, FB
		if FRAME == 0:
			FOURIER[:,0] = FA
			FOURIER[:,1] = FB


	X = wheel.omega_history
	Y = FOURIER[:,1]

	Xmid = XRES-YRES/2
	Ymid = YRES/2

	XX = transform_x(X,Xmid)
	YY = transform_x(-Y,Ymid)

	lines = [(x,y) for x,y in zip(XX,YY)]
	x0 = XX[0]
	y0 = YY[0]
	
	pygame.draw.aalines(surface, (255, 255, 255), False, lines)
	pygame.draw.circle(surface, (255, 0, 0), (x0,y0), 6)

	message = f'fps: {FRAME/(time.perf_counter()-start):.1f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,32))

	message = f'total mass: {sum([b.mass for b in buckets]):.1f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,64))

	message = f'omega: {wheel.omega:.2f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,96))

	message = f'd(omega): {wheel.d_omega/DT:.5f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,128))

	message = f'torque: {wheel.torque/wheel.moment/DT:.2f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,160))

	message = f'friction: {wheel.friction/wheel.moment/DT:.2f}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,192))

	message = f'frame: {FRAME}'
	msg_text = font.render(message, False, (255, 255, 255))
	surface.blit(msg_text, (32,224))

	screen.blit(surface,(0,0))
	pygame.display.flip()
	clock.tick(60)
	FRAME += 1