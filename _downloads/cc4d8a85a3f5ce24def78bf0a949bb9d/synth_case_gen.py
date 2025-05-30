"""
Generating R/B Light Curves
===========================

Creating and comparing light curves for two rocket body models
"""

import os

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.papers as mrp
import mirage.vis as mrv


def main():
    length = mr.minutes(5)
    cadence = mr.seconds(3)
    date0 = mr.utc(2025, 3, 8, 2, 0, 0)
    s0 = np.array([0.33333333, 0.33333333, -0.33333333])
    w0 = 0.03 * np.array([1.0, 2.0, 1.0])
    itensor = np.diag([1.0, 1.0, 0.25])

    model_dir = '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/models'
    station = mr.Station()
    mr.set_model_directory(model_dir)
    obj_true = mr.SpaceObject('matlib_saturn_v_sii.obj', identifier=20662)
    obj_simple = mr.SpaceObject('saturn_simple_tri.obj', identifier=20662)
    invert_obj_path = os.path.join(model_dir, 'saturn_simple.obj')

    obj_true_mat = mr.load_obj(obj_true.file_name)
    obj_true_mat.material_names[obj_true_mat.material_names == 'none'] = 'aluminum'

    obj_simple_mat = mr.load_obj(obj_simple.file_name)

    # Plotting the objects themselves
    from matplotlib.colors import ListedColormap

    # Define the colors we want to use
    white = np.array([245 / 256, 245 / 256, 245 / 256, 1.0])
    grey = np.array([120 / 256, 120 / 256, 120 / 256, 1.0])

    # Plotting the objects themselves
    p = pv.Plotter()
    mrv.render_spaceobject(
        p,
        obj_simple_mat,
        origin=(8, 0, 1),
        scalars=obj_simple_mat.material_names,
        cmap=ListedColormap(np.vstack((white, grey))),
    )
    mrv.render_spaceobject(
        p,
        obj_true_mat,
        origin=(-8, 0, 1),
        scalars=obj_true_mat.material_names,
        cmap=ListedColormap(np.vstack((grey, white))),
    )

    p.add_axes(shaft_length=1.0, label_size=(0.5, 0.2), line_width=3)
    p.camera.position = (0, 60, 0)
    p.camera.focal_point = (0, 0, 0)
    p.show()
    p.screenshot(
        '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/figs/rocket_bodies.png'
    )

    mrp.generate_light_curves(
        station,
        obj_true,
        obj_simple,
        invert_obj_path,
        lc_length=length,
        lc_cadence=cadence,
        integration_time_s=0.5,
        date0=date0,
        s0=s0,
        w0=w0,
        itensor=itensor,
        self_shadowing=True,
        trial_name='rocket_body_sdc',
        model_name='Rocket Body',
        make_plots=True,
        model_scale=0.3,
    )


if __name__ == '__main__':
    main()
