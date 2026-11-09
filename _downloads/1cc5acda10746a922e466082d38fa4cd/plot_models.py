# /// script
# requires-python = ">=3.11"
# dependencies = ["mirage"]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Plotting models
===============

Plotting the 3D models used in the case studies
"""

import os

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

MODEL_DIR = os.path.join(
    os.environ['SRCDIR'], '..', 'examples', '15-attitude-sdc-2025', 'models'
)


def main():
    """Render the high-fidelity spacecraft models used in the SDC case study."""
    mr.set_model_directory(MODEL_DIR)
    obj_true = mr.SpaceObject('echostar-ii-tri.obj')

    obj_true_mat = mr.load_obj(obj_true.file_name)

    # obj_true_mat.material_vals = (
    #     0 * (obj_true_mat.material_names == 'starlink_panel')
    #     + 0 * (obj_true_mat.material_names == 'none')
    #     + 1 * (obj_true_mat.material_names == 'aluminum')
    #     + 2 * (obj_true_mat.material_names == 'mli')
    # )

    obj_true_mat.material_names[obj_true_mat.material_names == 'starlink_panel'] = (
        'solar_panel'
    )
    obj_true_mat.material_names[obj_true_mat.material_names == 'none'] = 'solar_panel'

    from matplotlib.colors import ListedColormap

    # Define the colors we want to use
    blue = np.array([0 / 256, 89 / 256, 159 / 256, 1.0])
    grey = np.array([189 / 256, 189 / 256, 189 / 256, 1.0])
    yellow = np.array([255 / 256, 200 / 256, 88 / 256, 1.0])
    newcolors = np.vstack((grey, yellow, blue))

    p = pv.Plotter(window_size=(2000, 1000))
    mrv.render_spaceobject(
        p,
        obj_true_mat,
        origin=(0, 0, 0),
        scalars=obj_true_mat.material_names,
        show_scalar_bar=True,
        scalar_bar_args=dict(label_font_size=50),
        cmap=ListedColormap(newcolors),
    )
    p.camera.position = (50, -30, 20)
    p.camera.up = (0, 0, 1)
    p.camera.zoom(2.7)
    p.show_bounds()
    p.show()

    obj_true = mr.SpaceObject('star37e.obj')

    p = pv.Plotter(window_size=(1000, 500))
    mrv.render_spaceobject(
        p,
        obj_true,
        origin=(0, 0, 0),
        cmap=ListedColormap(np.array([156 / 256, 132 / 256, 112 / 256, 1.0])),
        scalars=obj_true.f[:, 0],  # Dummy scalars to activate the colormap
        show_scalar_bar=False,
    )
    p.camera.position = (-6, 6, 3)
    p.camera.up = (0, 0, 1)
    p.camera.zoom(2.3)
    p.show_bounds()
    p.show()


if __name__ == '__main__':
    main()
