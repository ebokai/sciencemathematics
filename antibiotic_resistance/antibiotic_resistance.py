import numpy as np 
import matplotlib.pyplot as plt 


class Bacterium(object):
	def __init__(self, i, j, MIC, n_mutations = 0):

		# location on lattice
		self.i = i
		self.j = j
		
		# minimum inhibitory concentration
		self.MIC = MIC 

		# number of mutations
		self.n_mutations = n_mutations
	

def reproduce(bacteria, pairs, p_reproduce = 0.05, p_mutate = 0.01):

	if len(bacteria) > 500:
		sample = np.random.choice(np.arange(len(bacteria)), 500, replace = False)
	else:
		sample = np.arange(len(bacteria))

	for i in sample:

		b = bacteria[i]

		if np.random.random() < p_reproduce:
			new_i = (b.i + np.random.randint(-1,2)) % SIMSIZE
			new_j = (b.j + np.random.randint(0,2)) % SIMSIZE
			p1 = (new_i, new_j)
			if p1 not in pairs:
				if np.random.random() < p_mutate:
					new_MIC = b.MIC * 10 
					new_n_mutations = b.n_mutations + 1
				else:
					new_MIC = b.MIC 
					new_n_mutations = b.n_mutations
				bacteria.append(Bacterium(new_i, new_j, new_MIC, new_n_mutations))
				pairs.append(p1)	
	return bacteria, pairs

def kill(bacteria, pairs, antibiotic_matrix):
	for b in bacteria:
		if b.MIC < antibiotic_matrix[b.i,b.j]:
			bacteria.remove(b)
			pairs.remove((b.i,b.j))		
	return bacteria,pairs


def initialize_ab_matrix(regions = 10, ab0 = 1):
	antibiotic_matrix = np.zeros((SIMSIZE, SIMSIZE))
	x = np.arange(SIMSIZE)
	for i in range(SIMSIZE):
		antibiotic_matrix[i,:] = ab0 * np.exp((x-10)/10)
	return antibiotic_matrix

def initialize_bacteria(n):
	pairs, bacteria = [], []
	for _ in range(n):
		i = np.random.randint(0, SIMSIZE)
		pair = (i,0)
		if pair not in pairs:
			pairs.append(pair)
			bacteria.append(Bacterium(i, 0, 1))
	return pairs, bacteria



# --- INITIALIZE ---
SIMSIZE = 150
antibiotic_matrix = initialize_ab_matrix()
pairs, bacteria = initialize_bacteria(50)
# ------------------

			
# --- SIMULATION ---
for t in range(25000):
	bacteria, pairs = reproduce(bacteria, pairs)
	bacteria, pairs = kill(bacteria, pairs, antibiotic_matrix)
	if t % 500 == 0:
		print('time: %d, number of bacteria: %d' % (t,len(bacteria)))
# ------------------


# --- PLOT ---
plt.matshow(np.log10(antibiotic_matrix),fignum=0,cmap='gray_r')
plt.colorbar(label='antibiotic concentration [log mg/L]')
plt.scatter([b.j for b in bacteria],[b.i for b in bacteria],
	c=[b.n_mutations for b in bacteria], cmap='autumn', edgecolor='None', s = 10)
plt.colorbar(label = 'number of mutations')
plt.show()
# ------------------



