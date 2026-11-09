# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Inertia Tensors for Meshes
==========================
"""

import numpy as np
import pyvista as pv

import mirage as mr


def tetrahedron_inertia_tensor(vertices: np.ndarray, density: np.ndarray) -> np.ndarray:
    """
    Compute the inertia tensor components of a tetrahedron given 3 vertices (4th is origin).

    Parameters:
    - vertices: (3, 3) numpy array, each row is (x, y, z) for vertex 1, 2, 3.
    - density: scalar mass density mu

    Returns:
    - I: 3x3 inertia tensor as numpy array
    """
    p1, p2, p3 = vertices
    x, y, z = vertices.T

    mat = np.array(
        [
            [p2[0] - p1[0], p3[0] - p1[0], -p1[0]],
            [p2[1] - p1[1], p3[1] - p1[1], -p1[1]],
            [p2[2] - p1[2], p3[2] - p1[2], -p1[2]],
        ]
    )
    volume = np.abs(np.linalg.det(mat)) / 6.0

    xx = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[0] * (x[1] + x[2]) + x[1] * x[2]
    yy = y[0] ** 2 + y[1] ** 2 + y[2] ** 2 + y[0] * (y[1] + y[2]) + y[1] * y[2]
    zz = z[0] ** 2 + z[1] ** 2 + z[2] ** 2 + z[0] * (z[1] + z[2]) + z[1] * z[2]

    yz = (
        2 * y[0] * z[0]
        + 2 * y[1] * z[1]
        + 2 * y[2] * z[2]
        + y[1] * z[0]
        + y[2] * z[0]
        + y[0] * z[1]
        + y[2] * z[1]
        + y[0] * z[2]
        + y[1] * z[2]
    )

    xy = (
        2 * x[0] * z[0]
        + 2 * x[1] * z[1]
        + 2 * x[2] * z[2]
        + x[1] * z[0]
        + x[2] * z[0]
        + x[0] * z[1]
        + x[2] * z[1]
        + x[0] * z[2]
        + x[1] * z[2]
    )

    xz = (
        2 * x[0] * y[0]
        + 2 * x[1] * y[1]
        + 2 * x[2] * y[2]
        + x[1] * y[0]
        + x[2] * y[0]
        + x[0] * y[1]
        + x[2] * y[1]
        + x[0] * y[2]
        + x[1] * y[2]
    )

    muV = density * volume
    coeff_diag = muV / 10
    coeff_offdiag = -muV / 20

    I = np.array(
        [
            [coeff_diag * (yy + zz), coeff_offdiag * xz, coeff_offdiag * xy],
            [coeff_offdiag * xz, coeff_diag * (xx + zz), coeff_offdiag * yz],
            [coeff_offdiag * xy, coeff_offdiag * yz, coeff_diag * (xx + yy)],
        ]
    )

    return I


def trimesh_itensor(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, densities: np.ndarray
) -> np.ndarray:
    """Compute a mesh inertia tensor by summing tetrahedron contributions."""
    if not isinstance(density, np.ndarray):
        densities = np.full_like(faces[:, 0], densities, dtype=np.float64)
    est_itensor = np.zeros((3, 3))
    for f, n, d in zip(faces, normals, densities):
        est_itensor += np.sign(mr.dot(vertices[f[0]], n)) * tetrahedron_inertia_tensor(
            vertices[f], density=d
        )
    return est_itensor


w = 2
h = 1
d = 2

density = 1
m = density * w * h * d

analytic_itensor = m / 12 * np.diag(((h**2 + d**2), (w**2 + d**2), (w**2 + h**2)))

mesh = pv.Cube(x_length=w, y_length=h, z_length=d).triangulate()

obj = mr.spaceobject_from_pyvista_mesh(mesh)

quad_sum_itensor = trimesh_itensor(obj.v, obj.f, obj.face_normals, density)
print(np.abs((quad_sum_itensor - analytic_itensor)).max())

# Now for a torus


def inertia_tensor_torus(m, R, rho):
    """
    Compute the inertia tensor for a torus object as shown in the image:
        I = [
            [m/8 * (5R^2 + 4rho^2),      0,                  0],
            [0,                 m/8 * (5R^2 + 4rho^2),       0],
            [0,                             0,        m/4 * (3R^2 + 4rho^2)]
        ]
    Parameters:
        m: mass
        R: major radius
        rho: minor radius
    Returns:
        3x3 numpy.ndarray representing the inertia tensor
    """
    I_xx = (m / 8) * (5 * R**2 + 4 * rho**2)
    I_yy = I_xx
    I_zz = (m / 4) * (3 * R**2 + 4 * rho**2)

    return np.diag((I_xx, I_yy, I_zz))


R = 5
rho = 20
m = 1
volume = 2 * np.pi**2 * R**2 * rho
density = m / volume

analytic_itensor = inertia_tensor_torus(m, R, rho)

mesh = pv.ParametricTorus(ringradius=rho, crosssectionradius=R).triangulate()
obj = mr.spaceobject_from_pyvista_mesh(mesh)

quad_sum_itensor = trimesh_itensor(obj.v, obj.f, obj.face_normals, density)

print(analytic_itensor)
print(quad_sum_itensor)
