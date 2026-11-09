# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
SDLR Noise Filtering
====================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pykdtree.kdtree import KDTree

import mirage as mr
import mirage.vis as mrv

n = 100000

x_sig = np.random.rand(n // 100)
y_sig = np.sin(x_sig)

x = np.random.rand(n, 2)
all_pts = np.vstack((x, np.vstack((x_sig, y_sig)).T))

nnd = KDTree(all_pts).query(all_pts, k=10)[0][:, 1:]

df = pl.read_parquet(
    os.path.join(
        os.environ['SRCDIR'],
        '..',
        'scratch',
        'digos',
        'data',
        'proc',
        'digos-dlr-midas.parquet',
    )
)
df = df.filter(pl.col('est_depth').is_not_nan())

epsecs = mr.date_to_epsec(np.array(df['date'].to_list()))
est_depth = df['est_depth'].to_numpy()

x = (epsecs - epsecs.min()) / np.ptp(epsecs)
y = (est_depth - est_depth.min()) / np.ptp(est_depth)
xy = np.vstack((x, y)).T

nnd = KDTree(xy).query(xy, k=11)[0]

bx = [0.5, 0.7]
dm = ((x > bx[0]) & (x < bx[1])) & ((y > bx[0]) & (y < bx[1]))
density = dm.sum() / (bx[1] - bx[0]) ** 2

xbr = np.linspace(0, 1, 1000)
xxbr, yybr = np.meshgrid(xbr, xbr)
br_pts = np.vstack((xxbr.flatten(), yybr.flatten())).T

br_nnd = KDTree(xy).query(br_pts, k=10)[0]
nn = 3

p = mr.nth_pp_neighbor_cdf(br_nnd[:, nn], nn, density)

plt.figure()
plt.scatter(*xy.T, c='k', s=10, label='Measurements')
plt.imshow(p.reshape(xxbr.shape) > 0.1, extent=(0, 1, 0, 1), origin='lower')
plt.colorbar(cax=mrv.get_cbar_ax(), label='Probability')
plt.show()

plt.figure(figsize=(8, 8))
for i, nn in enumerate([1, 3, 10]):
    for j, p_tol in enumerate([0.01, 0.02, 0.1]):
        plt.subplot(3, 3, i + 3 * j + 1)
        is_signal = mr.nth_pp_neighbor_cdf(nnd[:, nn], nn, density) < p_tol
        plt.scatter(*xy[is_signal].T, c='m', s=10, label='Signal')
        plt.scatter(*xy[~is_signal].T, c='k', s=10, label='Noise')
        plt.xlim(0.72, 0.82)
        plt.ylim(0.2, 0.66)
        plt.title(f'$p={p_tol}$, $n={nn}$')
        if j == 0 and i == 2:
            plt.legend()
        plt.xticks([])
        plt.yticks([])
        # plt.gca().set_facecolor('k')

plt.gcf().supxlabel('Time')
plt.gcf().supylabel('Residual range')

plt.tight_layout()
plt.show()
