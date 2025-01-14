"""
Generating R/B Light Curves
===========================

Creating and comparing light curves for two rocket body models
"""

import numpy as np
import pyvista as pv

# noqa
import mirage as mr
import mirage.papers as mrp
import mirage.vis as mrv


def main():
    length = mr.minutes(2)
    cadence = mr.seconds(2)
    date0 = mr.utc(2024, 10, 10, 8, 0, 0)
    s0 = np.array([-0.33333333, -0.33333333, -0.33333333])
    w0 = 0.1 * np.array([1.0, 1.0, 1.0])
    itensor = np.diag([1.0, 1.0, 0.5])

    station = mr.Station()
    mr.set_model_directory(
        '/Users/liamrobinson/Documents/maintained-research/mirage-models/accurate_sats/'
    )
    obj_true = mr.SpaceObject('matlib_saturn_v_sii.obj', identifier=18387)
    obj_simple = mr.SpaceObject('saturn_v_sii_simple_tri.obj', identifier=18387)
    invert_obj_path = '/Users/liamrobinson/Documents/maintained-research/mirage-models/accurate_sats/saturn_v_sii_simple.obj'

    obj_true_mat = mr.load_obj(obj_true.file_name)
    obj_true_mat.material_names[obj_true_mat.material_names == 'none'] = 'aluminum'

    obj_simple_mat = mr.load_obj(obj_simple.file_name)

    # Plotting the objects themselves
    p = pv.Plotter(window_size=(1200, 800))
    mrv.render_spaceobject(
        p,
        obj_simple_mat,
        origin=(16, 0, 0),
        scalars=obj_simple_mat.material_names,
        cmap='cool',
    )
    mrv.render_spaceobject(
        p,
        obj_true_mat,
        origin=(-16, 0, 0),
        scalars=obj_true_mat.material_names,
        cmap='cool_r',
    )
    p.add_axes(shaft_length=1.0, label_size=(0.5, 0.2), line_width=3)
    p.camera.position = (0, 70, -20)
    p.camera.up = (0, 0, 1)
    p.show()
    p.screenshot(
        '/Users/liamrobinson/Documents/maintained-research/mirage/examples/13-attitude-2024/figs/rocket_bodies.png'
    )

    mrp.generate_light_curves(
        station,
        obj_true,
        obj_simple,
        invert_obj_path,
        lc_length=length,
        lc_cadence=cadence,
        integration_time_s=0.01,  # to account for the scale factor of 1/2 between this model and the Centaur
        date0=date0,
        s0=s0,
        w0=w0,
        itensor=itensor,
        self_shadowing=False,
        trial_name='rocket_body',
        model_name='Rocket Body',
        make_plots=True,
    )


if __name__ == '__main__':
    main()
