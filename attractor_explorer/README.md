# Iterated Map Explorer

This project is an interactive **iterated map explorer** built with Python and Pygame. It allows you to explore the dynamics of a 2D quadratic map by adjusting its parameters in real-time and visualizing the resulting attractors.

---

## Features

- **Interactive sliders** to adjust 12 parameters of the quadratic map.
- Real-time visualization of iterated points.
- Calculation of **rasterization entropy** to measure orbit complexity.
- Fourier analysis of point distributions.
- Generate new random attractors with a key press.
- Smooth GUI with live updates using Pygame.

---

## Map Definition

The 2D quadratic map is defined as:

```

x_{n+1} = a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
y_{n+1} = b0 + b1*x + b2*y + b3*x^2 + b4*x*y + b5*y^2

````

Where `pars = [a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5]` are the 12 adjustable parameters.

---

## Requirements

- Python 3.10+
- Pygame
- NumPy
- Numba

Install dependencies via pip:

```bash
pip install pygame numpy numba
````

---

## How to Use

1. Run the script:

```bash
python main.py
```

2. Controls:

* **R** – Generate a new random attractor.
* **Arrow Up / Down** – Switch active parameter.
* **Arrow Left / Right** – Fine-tune active parameter.
* **Mouse Click** – Adjust parameter via slider.
* **ESC** – Exit the program.

3. The left panel contains sliders for all 12 parameters and displays entropy and Fourier analysis.
4. The right panel visualizes the iterated points of the current map.

---

## Visualization Details

* **Entropy** measures the uniformity of points in a 10x10 grid.
* **Fourier spectrum** is calculated on a 20x20 grid to analyze spatial patterns.
* Sliders are color-coded for positive (blue) and negative (red) values.

---

## Notes

* Initial conditions are fixed at `(x, y) = (0, 0)`.
* Transient iterations (`TRANSIENT = 100`) are ignored to avoid initial transients affecting measurements.
* Maximum iterations per orbit: `MAX_ITS = 35000`.

---

## License

This project is open-source and available under the MIT License.

---
