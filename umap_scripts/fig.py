# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure(figsize=(10, 5))

# First subplot: Unit cell with atoms
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_title("Unit Cell", pad=20)

# Draw unit cell cube
cube_edges = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)
for edge in [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]:
    ax1.plot3D(*cube_edges[edge].T, color="blue", lw=1)

# Add atoms with periodic connections
atoms = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]])
ax1.scatter(*atoms.T, s=100, c="red", depthshade=False)

# Draw periodic connections
for atom in atoms:
    for dim in range(3):
        if atom[dim] > 0.5:
            neighbor = atom.copy()
            neighbor[dim] -= 1.0
            ax1.plot(
                [atom[0], neighbor[0] + 1],
                [atom[1], neighbor[1]],
                [atom[2], neighbor[2]],
                color="gray",
                linestyle="--",
            )

# Second subplot: Periodic tiling
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_title("Periodic Tiling", pad=20)

# Draw 3x3x3 grid of cells
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            offset = np.array([i, j, k])
            edges = cube_edges + offset
            for edge in [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]:
                ax2.plot3D(*edges[edge].T, color="skyblue", alpha=0.3, lw=0.5)

# Plot atoms in central cell and their periodic neighbors
all_atoms = []
for atom in atoms:
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                all_atoms.append(atom + np.array([i, j, k]))

all_atoms = np.array(all_atoms)
ax2.scatter(*all_atoms.T, s=30, c="red", alpha=0.6, depthshade=False)

# Formatting
for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)
    ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.savefig("unit_cell.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
