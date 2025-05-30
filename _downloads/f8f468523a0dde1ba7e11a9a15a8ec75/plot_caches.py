"""
Shadow Caches
=============
"""

import sys

sys.path.append('/Users/liamrobinson/Documents/maintained-wip/taichi-polyclip')
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import ticlip

import mirage.vis as mrv


def azel_to_ij(naz: int, nel: int, az: float, el: float):
    i = int(np.floor(az / (np.pi * 2) * (naz - 1)))
    j = int(np.floor((el + np.pi / 2) / np.pi * (nel - 1)))
    return i, j


def ij_to_az_el(naz: int, nel: int, i: int, j: int) -> np.ndarray:
    az = i / naz * ti.math.pi * 2
    el = j / nel * ti.math.pi - ti.math.pi / 2
    return np.array((az, el))


ti.init(
    arch=ti.cpu,
    cfg_optimization=False,
    opt_level=1,
    fast_math=False,
    advanced_optimization=False,
)

obj_paths = [
    '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/models/star37e-quad.obj',
    '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/models/echostar-ii-quad.obj',
]

naz, nel = 300, 300

for obj_path in obj_paths:
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

    deg_start = (215, -32)
    az, el = np.deg2rad(deg_start)
    i, j = azel_to_ij(naz, nel, az, el)
    azel_start = np.rad2deg(ij_to_az_el(naz, nel, i, j))

    azmax = az + 2 * np.pi / naz
    elmax = el + np.pi / nel
    extent = np.rad2deg(([az, azmax, el, elmax]))

    plt.figure(figsize=(6, 4))

    im = ticlip.plot_cache(cache, naz, nel, nf)

    a1 = plt.gca()
    a1.add_patch(
        plt.Rectangle(
            azel_start,
            width=360 / naz,
            height=180 / nel,
            fill=False,
            edgecolor='red',
            linewidth=2,
        )
    )
    plt.title('Shadow Cache Density')
    plt.xlabel('Azimuth ($a$) [deg]')
    plt.ylabel('Elevation ($h$) [deg]')
    plt.colorbar(im, label='Number of Self-Shadowing Faces', cax=mrv.get_cbar_ax())
    plt.tight_layout()
