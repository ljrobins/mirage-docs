"""
Voronoi Diagram to Supports
===========================

Investigating whether there is a connection between the area of a face, its support, and the area of its spherical projection on the Spherical Voronoi diagram
"""

import numpy as np
import pyvista as pv
from scipy.spatial import SphericalVoronoi

import mirage as mr
import mirage.vis as mrv

obj = mr.SpaceObject("gem.obj")
obj.shift_to_center_of_mass()


sv = SphericalVoronoi(obj.unique_normals)
# sv.sort_vertices_of_regions()

voronoi_areas = sv.calculate_areas()
voronoi_areas = voronoi_areas[obj.unique_to_all]

qp = mr.spiral_sample_sphere(int(1e5))
qv = mr.spherical_voronoi_weighting(obj.face_normals, voronoi_areas, qp)

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, scalars=voronoi_areas)
mrv.scatter3(pl, qp, scalars=qv, point_size=10, opacity=0.01)
pl.disable_anti_aliasing()
pl.show()
