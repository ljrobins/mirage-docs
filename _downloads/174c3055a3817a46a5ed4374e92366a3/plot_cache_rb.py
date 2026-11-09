# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mirage",
#     "taichi",
# ]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Shadow Caches
=============
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from taichi.lang.impl import get_runtime

import mirage.vis as mrv
from mirage import ticlip


def azel_to_ij(naz: int, nel: int, az: float, el: float):
    """Convert azimuth/elevation angles to cache-grid indices."""
    i = int(np.floor(az / (np.pi * 2) * (naz - 1)))
    j = int(np.floor((el + np.pi / 2) / np.pi * (nel - 1)))
    return i, j


def ij_to_az_el(naz: int, nel: int, i: int, j: int) -> np.ndarray:
    """Convert cache-grid indices to azimuth/elevation angles."""
    az = i / naz * ti.math.pi * 2
    el = j / nel * ti.math.pi - ti.math.pi / 2
    return np.array((az, el))


# The docs configuration initializes Taichi before importing gallery modules.
# Re-initializing it here invalidates fields declared by mirage.ticlip.
if get_runtime().prog is None:
    ti.init(
        arch=ti.cpu,
        cfg_optimization=False,
        opt_level=1,
        fast_math=False,
        advanced_optimization=False,
    )

obj_path = os.path.join(
    os.environ['SRCDIR'], '../examples/15-attitude-sdc-2025/models/star37e-quad.obj'
)
# obj_path = os.path.join(
#     os.environ['SRCDIR'],
#     '..',
#     'examples',
#     '15-attitude-sdc-2025',
#     'models',
#     'echostar-ii-quad.obj',
# )
naz, nel = 200, 100

finfo = ticlip.load_finfo(obj_path)

cache = ticlip.gen_cache(naz, nel)
cache_np = cache.to_numpy(dtype=np.uint32)

nf = finfo.fn.shape[0]
n = naz * nel * nf**2  # Number of points actually used for the cache
cache = ticlip.gen_cache(naz, nel)

cache_np = cache.to_numpy(dtype=np.uint8).flatten()[:n].reshape(naz, nel, nf, nf)

assert np.allclose(
    0.0, np.trace(cache_np, axis1=-1, axis2=-2).sum()
)  # Make sure that no face is ever shadowing itself

cache_sum = cache_np.sum(axis=2).sum(axis=2)

plt.figure(figsize=(6, 4))

im = ticlip.plot_cache(cache, naz, nel, nf)

plt.title('Shadow Cache Density')
plt.xlabel('Azimuth ($a$) [deg]')
plt.ylabel('Elevation ($h$) [deg]')
plt.colorbar(im, label='Number of Self-Shadowing Faces', cax=mrv.get_cbar_ax())
plt.tight_layout()
plt.show()
