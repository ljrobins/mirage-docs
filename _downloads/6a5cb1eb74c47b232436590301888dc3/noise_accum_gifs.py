"""
Noise Sampling
==============

Plots the accumulation of noise in the CCD due to integration noise, readout noise, and truncation noise.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# %%
# Integration noise

d = 10
t = np.linspace(0, 1, 100)
grid = np.zeros((d, d))

fig, ax = plt.subplots(figsize=(5, 5))

actor = ax.imshow(grid, cmap='gray', vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])


def animate(i):
    global grid
    grid += np.random.poisson(1, (d, d))
    actor.set_data(grid)
    plt.tight_layout()
    actor.vmin = 0
    plt.title(f'Integrating... {i / t.size * 100:.1f}%')
    return (actor,)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=t.size, interval=50)
plt.show()

# %%
# Readout noise

t = np.linspace(0, 1, 10)
grid = np.zeros((d, d))

fig, ax = plt.subplots(figsize=(5, 5))

actor = ax.imshow(grid, cmap='gray', vmin=-2, vmax=2)
plt.xticks([])
plt.yticks([])


def animate(i):
    global grid
    grid = np.random.normal(0, 1, (d, d))
    actor.set_data(grid)
    plt.tight_layout()
    plt.title(f'Readout Sample {i+1}/{t.size}')
    return (actor,)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=t.size, interval=500)
plt.show()

# %%
# Truncation noise

t = np.linspace(0, 1, 10)
grid = np.zeros((d, d))

fig, ax = plt.subplots(figsize=(5, 5))

actor = ax.imshow(grid, cmap='gray', vmin=-0.5, vmax=0.5)
plt.xticks([])
plt.yticks([])


def animate(i):
    global grid
    grid = 0.5 - np.random.random((d, d))
    actor.set_data(grid)
    plt.tight_layout()
    plt.title(f'Truncation Sample {i+1}/{t.size}')
    return (actor,)


ani = animation.FuncAnimation(fig, animate, repeat=True, frames=t.size, interval=500)
plt.show()
