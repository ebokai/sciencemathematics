import matplotlib.pyplot as plt 
import matplotlib.animation as anim 
import numpy as np 
import colorsys
from matplotlib.collections import LineCollection 

# Function to compute a transformation on a complex number z
# Parameters:
# z: complex number to be transformed
# a, b: angles influencing the rotation and scaling
# c: coefficients for polynomial terms
# p: powers for polynomial terms
def f(z, a, b, c, p):
    return np.sum(c * np.power(z, p)) * np.exp(1j * a * b)

# Function to interpolate between two values a and b based on a parameter t and duration dt
# This creates a smooth transition using a cosine function
def morph(a, b, t, dt):
    return a + (b - a) * (1 + np.cos(2 * np.pi * t / dt)) / 2

# Initialization function for the animation
# This is called once at the beginning to set up the plot
def init():
    lc.set_segments(lines)
    return lc,

# Animation function, updates the lines for each frame
# This is called repeatedly to update the plot
def animate(t):
    lines = []  # List to hold the line segments for the current frame

    # Interpolating parameters for the current time t
    k = [morph(k1[j], k2[j], t, dt[j]) for j in range(n_parts)]
    ka_t = [morph(ka[j, 0], ka[j, 1], t, d1[j]) for j in range(n_z)]
    kb_t = [morph(kb[j, 0], kb[j, 1], t, d2[j]) for j in range(n_z)]
    kc_t = [morph(kc[j, 0], kc[j, 1], t, d3[j]) for j in range(n_z)]
    pa_t = [morph(pa[j, 0], pa[j, 1], t, d4[j]) for j in range(n_z)]
    pb_t = [morph(pb[j, 0], pb[j, 1], t, d5[j]) for j in range(n_z)]
    r_t = [morph(r[j, 0], r[j, 1], t, d6[j]) for j in range(n_z)]
    ro_t = [morph(ro[j, 0], ro[j, 1], t, d7[j]) for j in range(n_z)]

    # Generate lines based on the current interpolated parameters
    for i, a in enumerate(np.linspace(0, 2 * np.pi, n_lines)):
        line = []
        # Initial complex number z is a sum of rotational and radial components
        z = np.sum([ro_t[j] * np.exp(1j * a) + r_t[j] * np.cos(i / n_lines * 2 * np.pi * ka_t[j] + pa_t[j]) * np.exp(1j * (a * kb_t[j] + pb_t[j])) for j in range(n_z)])
        line.append((np.real(z), np.imag(z)))
        
        # Apply the transformation function iteratively
        for j in range(n_parts):
            z = f(z, a, k[j], coeffs[j], powers[j])
            line.append((np.real(z), np.imag(z)))

        lines.append(line)

    lc.set_segments(lines)  # Update the line collection with the new segments
    return lc,

# Animation parameters
dpi = 72
xres = 1024
yres = 768

# Set plot background color to black
plt.rcParams['axes.facecolor'] = '#000000'  
fig, ax = plt.subplots(figsize=(xres / dpi, yres / dpi), facecolor='k', dpi=dpi)
ax.set_aspect('equal')

# Number of lines and parts
n_lines = 500  # Number of lines to draw
n_parts = 1    # Number of parts in the transformation
n_z = 1        # Number of initial complex numbers

# Generate random parameters for the transformation functions
orders = np.random.randint(1, 4, n_parts)  # Polynomial orders for the transformation
coeffs = [np.random.uniform(-1, 1, order + 1) for order in orders]  # Coefficients for the polynomials
powers = [np.arange(order + 1) for order in orders]  # Powers for the polynomials

# Function parameters (interpolation targets)
k1 = np.random.randint(-10, 10, n_parts)
k2 = np.random.randint(-10, 10, n_parts)
dt = np.random.randint(750, 7500, n_parts)  # Durations for the interpolation

# Parameters for the initial complex number z
ka = np.random.randint(-10, 10, (n_z, 2))
d1 = np.random.randint(750, 7500, n_z)

kb = np.random.randint(-10, 10, (n_z, 2))
d2 = np.random.randint(750, 7500, n_z)

kc = np.random.randint(-10, 10, (n_z, 2))
d3 = np.random.randint(750, 7500, n_z)

r = np.random.uniform(0.1, 0.3, (n_z, 2))  # Radial components
d4 = np.random.randint(750, 7500, n_z)

ro = np.random.uniform(0.1, 0.3, (n_z, 2))  # Rotational components
d5 = np.random.randint(750, 7500, n_z)

pa = np.pi / 6 * np.random.randint(0, 2, (n_z, 2))  # Phase offsets
d6 = np.random.randint(750, 7500, n_z)

pb = np.pi / 6 * np.random.randint(0, 2, (n_z, 2))  # Phase offsets
d7 = np.random.randint(750, 7500, n_z)

# List to hold the lines for plotting
lines = []  
colors = [colorsys.hsv_to_rgb(i / n_lines, 1, 1) for i in range(n_lines)]  # Generate colors for the lines

# Create a LineCollection object to hold the lines and add it to the axes
lc = LineCollection(lines, alpha=0.45, lw=0.5, color=colors)
ax.add_collection(lc)
ax.set_xlim(-1.6, 1.6)  # Set x-axis limits
ax.set_ylim(-0.9, 0.9)  # Set y-axis limits

# Create the animation object
animation = anim.FuncAnimation(fig, animate, init_func=init, interval=0, blit=True)
plt.show()
