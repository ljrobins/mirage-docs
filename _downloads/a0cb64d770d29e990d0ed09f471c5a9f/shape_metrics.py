"""
Shape Comparison Metrics
========================
Using Frueh and Oliker's delta-neighborhood metric to compare shapes.
"""

from typing import Tuple

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

import mirage as mr
import mirage.vis as mrv

p = mr.SpaceObject('gem.obj')
v_vol1 = p.v / p.volume ** (1 / 3)
p = mr.SpaceObject(vertices_and_faces=(v_vol1 + 0.2, p.f.copy()))

# %%
# We need to find the in-sphere and out-sphere of the object. This optimization problem ends up boiling down to the
# location and radius of each sphere. Equivalently, we can just optimize the location of the origin to maximize the minimum support


def compute_in_sphere(p: mr.SpaceObject) -> Tuple[np.ndarray, float]:
    if not np.isclose(p.volume, 1):
        v_vol1 = p.v / p.volume ** (1 / 3)
        p = mr.SpaceObject(vertices_and_faces=(v_vol1, p.f.copy()))

    def in_sphere_objective(x):
        p2 = mr.SpaceObject(vertices_and_faces=(p.v.copy() - x, p.f.copy()))
        return -np.min(p2.supports)

    solver_kwargs = dict(jac='3-point', method='BFGS')
    in_sol = minimize(in_sphere_objective, np.zeros(3), **solver_kwargs)
    return in_sol.x, -in_sol.fun


def compute_out_sphere(p: mr.SpaceObject) -> Tuple[np.ndarray, float]:
    if not np.isclose(p.volume, 1):
        v_vol1 = p.v / p.volume ** (1 / 3)
        p = mr.SpaceObject(vertices_and_faces=(v_vol1, p.f.copy()))

    def out_sphere_objective(x):
        return np.max(mr.vecnorm(p.v - x))

    solver_kwargs = dict(jac='3-point', method='BFGS')
    out_sol = minimize(out_sphere_objective, np.zeros(3), **solver_kwargs)
    return out_sol.x, out_sol.fun


mr.tic('Optimizing in-sphere')
in_solx, in_solr = compute_in_sphere(p)
mr.toc()

mr.tic('Optimizing out-sphere')
out_solx, out_solr = compute_out_sphere(p)
mr.toc()

print(in_solx, in_solr)
print(out_solx, out_solr)


def delta_neighborhood(p1: mr.SpaceObject, p2: mr.SpaceObject) -> float:
    _, R1 = compute_out_sphere(p1)
    _, R2 = compute_out_sphere(p2)
    _, r1 = compute_in_sphere(p1)
    _, r2 = compute_in_sphere(p2)

    ctilde = 2 * (R1 / r1) ** 2 / (R1 / r1 + R2 / r2)
    return ctilde


pl = pv.Plotter()
mrv.render_spaceobject(pl, p, opacity=0.8)
mrv.two_sphere(pl, -in_solr, in_solx, color='r', opacity=0.3)
mrv.two_sphere(pl, out_solr, out_solx, color='b', opacity=0.3)
mrv.orbit_plotter(pl)
pl.show()
