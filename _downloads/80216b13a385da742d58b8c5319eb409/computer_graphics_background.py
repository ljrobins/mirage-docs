"""
Graphics Background
===================

Plotting various transformations and concepts from computer graphics
"""

# isort: skip_file
# isort: off

import numpy as np
import vtk
import pyvista as pv

vtk.__version__

import mirage as mr
import mirage.vis as mrv

pv.set_plot_theme('document')


def orthographic_frustum_as_pyvista_mesh(
    origin, direction, up, width, near, far
) -> pv.PolyData:
    box = pv.Box()
    box.points[:, 0] *= width
    box.points[:, 1] *= width
    box.points[:, 2] *= far - near
    v3 = mr.hat(direction)
    v1 = mr.hat(np.cross(up, v3))
    v2 = mr.hat(np.cross(v3, v1))
    dcm = np.vstack((v1, v2, v3)).T
    box.points = mr.stack_mat_mult_vec(dcm, box.points)
    box.points += origin + direction * (far - near)
    return box


camera = pv.Camera()
near_range = 0.3
far_range = 0.9
camera.clipping_range = (near_range, far_range)
camera.position = (1.0, -1.0, 0.0)
camera.up = (0.0, 1.1, 1.0)
unit_vector = mr.hat(np.array(camera.direction))

perspective_frustum = camera.view_frustum(1.0)

position = camera.position
focal_point = camera.focal_point
line = pv.Line(position, focal_point)

bunny_obj = mr.SpaceObject('stanford_bunny.obj')
bunny = bunny_obj._mesh
xyz = camera.position + unit_vector * 0.6 - np.mean(bunny.points, axis=0)
bunny.points += np.array(xyz)

P = mr.perspective_projection_matrix(camera.view_angle, 1, near_range, far_range)
W = np.eye(4)
M = np.eye(4)
M[3, :3] = -xyz
V = mr.look_at_matrix(camera.position, camera.focal_point, camera.up)
MVP = P @ V @ M

ortho_frustum = orthographic_frustum_as_pyvista_mesh(
    camera.position, unit_vector, camera.up, 0.1, near_range, far_range
)

# %%
# Plotting the camera view frustum for perspective projection

pl = pv.Plotter(window_size=(2000, 1300), shape=(2, 1))
for i, frustum in enumerate([perspective_frustum, ortho_frustum]):
    clipping_plane_opacity = (
        np.abs(
            mr.dot(
                unit_vector.reshape(-1, 3), frustum.compute_normals()['Normals']
            ).flatten()
        )
        > 0.99
    ) * 0.2
    pl.subplot(i, 0)
    mrv.render_spaceobject(pl, bunny_obj)
    pl.add_mesh(frustum, style='wireframe', line_width=5, color='k')
    pl.add_mesh(frustum, color='orange', opacity=clipping_plane_opacity, line_width=5)
    pl.add_mesh(line, color='k', line_width=5)

    if i == 0:
        pl.add_text(
            'Perspective Projection',
            font_size=25,
            position='upper_edge',
            color='k',
            font='courier',
        )
    else:
        pl.add_text(
            'Orthographic Projection',
            font_size=25,
            position='upper_edge',
            color='k',
            font='courier',
        )

    if i == 0:
        pl.add_point_labels(
            [
                position,
                camera.position + unit_vector * near_range,
                camera.position + unit_vector * far_range,
                focal_point,
            ],
            ['$R_{cam}$', 'Near plane', 'Far plane', '$T_{cam}$'],
            margin=0,
            fill_shape=True,
            font_size=30,
            shape_color='white',
            point_color='red',
            text_color='black',
            always_visible=True,
        )
    else:
        pl.add_point_labels(
            [
                position,
                focal_point,
            ],
            ['$R_{cam}$', '$T_{cam}$'],
            margin=0,
            fill_shape=True,
            font_size=30,
            shape_color='white',
            point_color='red',
            text_color='black',
            always_visible=True,
        )
        near_range = 0.01

    bunny_on_near_plane = bunny.copy()
    x = mr.stack_mat_mult_vec(
        P @ V @ M,
        np.hstack(
            (bunny_on_near_plane.points, np.ones((bunny_on_near_plane.n_points, 1)))
        ),
    )
    bunny_on_near_plane.points = x[:, :3] / x[:, [3]]
    bunny_on_near_plane.points[:, -1] = 0
    bunny_on_near_plane.points *= near_range * mr.tand(camera.view_angle / 2) / 2
    bunny_on_near_plane.points = mr.stack_mat_mult_vec(
        V[:3, :3].T, bunny_on_near_plane.points
    )
    bunny_on_near_plane.rotate_vector(unit_vector, 180, inplace=True)
    bunny_on_near_plane.points += camera.position + unit_vector * near_range

    pl.add_mesh(bunny_on_near_plane, opacity=0.5)
    pl.add_mesh(bunny_on_near_plane, opacity=0.1, color='k', style='wireframe')

    mrv.plot_basis(
        pl, M[:3, :3].T, 'M', origin=np.mean(bunny.points, axis=0), scale=0.2
    )
    mrv.plot_basis(pl, W[:3, :3].T, 'W', origin=np.zeros(3), scale=0.2)
    mrv.plot_basis(pl, V[:3, :3].T, 'V', origin=camera.position, scale=0.2)
    mrv.plot_arrow(
        pl, camera.position, camera.up, label='$U_{cam}$', scale=0.15, color='c'
    )

pl.link_views()
pl.camera.position = (1.1, 1.1, -0.5)
pl.camera.focal_point = camera.position + unit_vector * far_range
pl.camera.up = (0.0, 0.0, 1.0)
pl.camera.zoom(1.3)
pl.show()
