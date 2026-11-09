"""
Triangular Reflection Depth
===========================
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.interpolate import interp1d

import mirage as mr

pts = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)
o = np.array([1, -0.4])
o_hat = o / np.linalg.norm(o)
depths = np.linspace(-0.3, 1.2, 200)
depth_pdf = mr.triangle_depth_pdf(pts, -o_hat)
p = interp1d(depth_pdf[:, 0], depth_pdf[:, 1], fill_value=0, bounds_error=False)(depths)

grid_n = 200
xg = np.linspace(-0.1, 1.1, grid_n)
yg = np.linspace(-0.1, 1.1, grid_n)
XG, YG = np.meshgrid(xg, yg)
pts_grid = np.stack([XG.ravel(), YG.ravel()], axis=1)
tri_path = Path(np.vstack([pts, pts[0]]))
inside = tri_path.contains_points(pts_grid)
depth_grid = pts_grid @ o_hat
depth_grid_2d = depth_grid.reshape(grid_n, grid_n)
depth_grid_2d[~inside.reshape(grid_n, grid_n)] = np.nan
v_depths = pts @ o_hat

r1, r2 = np.random.rand(1000), np.random.rand(1000)
mask = r1 + r2 > 1
r1[mask], r2[mask] = 1 - r1[mask], 1 - r2[mask]
samples = r1[:, None] * (pts[1] - pts[0]) + r2[:, None] * (pts[2] - pts[0]) + pts[0]
sample_depths = samples @ o_hat

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.add_patch(Polygon(pts, closed=True, alpha=0.15, color='steelblue', zorder=1))
ax.plot(*pts[[0, 1, 2, 0]].T, 'k-', lw=1.5, zorder=2)
levels = np.linspace(np.nanmin(depth_grid_2d), np.nanmax(depth_grid_2d), 8)
cs = ax.contour(
    XG, YG, depth_grid_2d, levels=levels, colors='black', linewidths=2, zorder=3
)
ax.clabel(cs, fmt='$d=%.2f$ [m]', fontsize=8)
arrow_scale = 0.4
ax.annotate(
    '',
    xy=-o_hat * arrow_scale,
    xytext=(0, 0),
    arrowprops=dict(arrowstyle='->', color='red', lw=2),
)
ax.text(
    *(-o_hat * arrow_scale * 1.15),
    '$\\hat{o}$',
    color='red',
    ha='center',
    va='center',
    fontsize=11,
)
ax.scatter(pts[:, 0], pts[:, 1], color='k', zorder=5, s=40)
for i, v in enumerate(pts, 1):
    ax.text(
        v[0] + 0.04,
        v[1] + 0.04,
        f'$v_{i}$',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'),
    )
ax.scatter(
    samples[:, 0],
    samples[:, 1],
    color='tab:orange',
    s=5,
    zorder=4,
    alpha=0.6,
    label='Random samples',
)
ax.legend()
ax.set_xlim(-0.5, 1.25)
ax.set_ylim(-0.1, 1.25)
ax.set_aspect('equal')
ax.set_title('Sampled Triangle and Depth Contours')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid()
ax2 = axes[1]
ax2.plot(depths, p, 'steelblue', lw=2, label='Analytic reflection depth PDF')
ax2.fill_between(depths, p, alpha=0.15, color='steelblue')
ax2.hist(
    sample_depths,
    bins=30,
    density=True,
    alpha=0.4,
    color='tab:orange',
    label='Monte Carlo reflection PDF',
)
colors = ['tab:orange', 'tab:green', 'tab:red']
for i, (vd, col) in enumerate(zip(v_depths, colors), 1):
    ax2.axvline(vd, color=col, ls='--', lw=1.5, label=f'$v_{i}$ depth = {vd:.2f}')
ax2.set_xlabel('Depth $d$ [m]')
ax2.set_ylabel('Probability Density')
ax2.set_title('Reflection Depth PDF along $\\hat{o}$')
ax2.legend(fontsize=9)
ax2.set_xlim(depths[0], depths[-1])
ax2.grid()
plt.tight_layout()
plt.show()
