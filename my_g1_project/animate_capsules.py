"""Animates the arm's two capsules moving through a logged trajectory,
color-coded by the real safety margin at each tick."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

data = np.load("trajectory_log.npz")
shoulder, elbow, palm, margin = data["shoulder"], data["elbow"], data["palm"], data["margin"]

STRIDE = max(1, len(margin) // 150)   # cap ~150 frames for a reasonable file size
shoulder, elbow, palm, margin = shoulder[::STRIDE], elbow[::STRIDE], palm[::STRIDE], margin[::STRIDE]

center = np.array([0.475, 0.0, 0.808])
half = np.array([0.35, 0.01, 0.075])

def box_corners(center, half):
    return np.array([center + np.array([sx, sy, sz]) * half
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

def box_edges():
    return [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),
            (2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

def set_axes_equal(ax):
    """matplotlib doesn't auto-equalize 3D axes -- without this, a thin
    wall visually looks like a cube, the same issue from the first test plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = limits.mean(axis=1)
    r = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([origin[0]-r, origin[0]+r])
    ax.set_ylim3d([origin[1]-r, origin[1]+r])
    ax.set_zlim3d([origin[2]-r, origin[2]+r])

corners = box_corners(center, half)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

def update(i):
    ax.cla()
    for a, b in box_edges():
        ax.plot(*zip(corners[a], corners[b]), color="gray", linewidth=1)

    safe = margin[i] >= 0
    color = "tab:green" if safe else "tab:red"

    ax.plot([shoulder[i,0], elbow[i,0]], [shoulder[i,1], elbow[i,1]], [shoulder[i,2], elbow[i,2]],
            color=color, linewidth=6, solid_capstyle="round", alpha=0.85)
    ax.plot([elbow[i,0], palm[i,0]], [elbow[i,1], palm[i,1]], [elbow[i,2], palm[i,2]],
            color=color, linewidth=6, solid_capstyle="round", alpha=0.85)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"tick {i*STRIDE}   margin = {margin[i]:.4f}   {'SAFE' if safe else 'VIOLATION'}",
                 color=color)
    set_axes_equal(ax)

ani = FuncAnimation(fig, update, frames=len(margin), interval=80)
ani.save("capsule_trajectory.gif", writer=PillowWriter(fps=12))
print("saved capsule_trajectory.gif")