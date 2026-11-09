# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Constant Offset Surfaces
========================

Uniformly sampling points from a surface in orientation space that puts a given body vector at a constant angular offset from a given vector
"""

import pyvista as pv

import mirage as mr
import mirage.vis as mrv

p = pv.Plotter()

n = int(1e4)

vb = mr.rand_unit_vectors(1)  # the body frame to offset from

for i in range(500):
    # The vector we want to be offset from by the given angle
    vi = mr.rand_unit_vectors(1)  # offset vector, in inertial
    # alpha = np.full((vi.shape[0]), 0.5)  # offset angle, rad
    alpha = 0.02

    quats_inertial_to_body = mr.random_offset_quaternions(
        vi, vb, alpha, size=n, boundary_only=False
    )
    dcm_inertial_to_body = mr.quat_to_dcm(quats_inertial_to_body)
    mrps = mr.dcm_to_mrp(dcm_inertial_to_body)

    # assert np.allclose(
    #     mr.angle_between_vecs(mr.stack_mat_mult_vec(dcm_inertial_to_body, vi), vb), alpha
    # )

    mrv.scatter3(p, mrps, point_size=5, scalars=mrps.shape[0] * [i], cmap='twilight')
p.set_background('k')

p.show()
