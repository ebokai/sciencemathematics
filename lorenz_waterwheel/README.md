# Wheel and Bucket Simulation

This project is an interactive **wheel-and-bucket physics simulation** implemented in Python using **Pygame** and **NumPy**. The simulation models a wheel with multiple buckets around its perimeter, where buckets can **fill and leak mass**, generating torque and affecting the wheel’s rotation.  

It provides real-time visualization of **angular velocity**, **torque**, **mass distribution**, and **Fourier analysis** of the dynamics.

---

## Features

- Real-time simulation of a **rotating wheel** with multiple buckets.
- **Buckets fill and leak** depending on position and wheel rotation.
- Calculates **torque**, **friction**, and **angular acceleration** dynamically.
- Visualizes:
  - Wheel rotation
  - Bucket mass distribution
  - Historical angular velocity
  - Fourier transform of mass-weighted components
- Real-time **statistics overlay**: FPS, total mass, omega, torque, friction, and frame number.

---

## Requirements

- Python 3.10+  
- Pygame  
- NumPy  

Install dependencies via pip:

```bash
pip install pygame numpy
````

---

## How to Use

1. Ensure a `config.py` file exists with the required simulation constants (e.g., `XRES`, `YRES`, `BUCKET_SIZE`, `N_BUCKETS`, `GRAVITY`, `DT`, etc.).
2. Run the simulation:

```bash
python wheel_simulation.py
```

3. The simulation window will show:

   * A central wheel with buckets
   * Real-time rotation and mass changes
   * A trace of historical omega vs. Fourier component

4. Close the window or press the quit button to exit.

---

## Key Classes and Functions

* **Bucket**

  * Represents a single bucket on the wheel.
  * Handles **mass, torque calculation, filling/leaking, and drawing**.
* **Wheel**

  * Represents the rotating wheel.
  * Computes **angular velocity**, **moment of inertia**, **torque**, and **friction**.
* **transform_x(X, mid)**

  * Scales data to screen coordinates for plotting.
* **Main Loop**

  * Updates wheel rotation
  * Updates bucket masses
  * Draws the wheel, buckets, and Fourier/omega trace
  * Displays overlay information

---

## Simulation Details

* `N_BUCKETS` defines the number of buckets around the wheel.
* `FILL_RATE` and `LEAK_RATE` control mass dynamics of buckets.
* `FRICTION` models damping proportional to angular velocity.
* `HISTORY` defines how many frames of omega are stored for visualization.
* Fourier transform tracks global bucket mass distribution over time.

---

## License

This project is open-source under the MIT License.

