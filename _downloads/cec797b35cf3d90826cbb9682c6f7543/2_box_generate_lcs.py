"""
Generate TDRS Light Curves
==========================

Creating and comparing light curves for two TDRS models
"""

# isort: off

import numpy as np

# noqa
import mirage as mr
import mirage.papers as mrp
import pyvista as pv
import mirage.vis as mrv


def main():
    date0 = mr.utc(2024, 10, 10, 10, 0, 0)
    length = mr.minutes(1)
    cadence = mr.seconds(1)
    s0 = np.array([-0.33333333, -0.33333333, -0.33333333])
    w0 = 0.1 * np.array([1.0, 1.0, 1.0])
    itensor = np.diag([1, 0.345, 1.034])
    integration_time_s = 0.1

    station = mr.Station()
    mr.set_model_directory(
        '/Users/liamrobinson/Documents/maintained-research/mirage-models/accurate_sats/'
    )
    obj_true = mr.SpaceObject('matlib_hylas4.obj', identifier='TDRS 12')
    obj_simple = mr.SpaceObject('box_wing_antenna_tri.obj', identifier='TDRS 12')

    obj_true_mat = mr.load_obj(obj_true.file_name)
    obj_true_mat.material_names[obj_true_mat.material_names == 'starlink_panel'] = (
        'solar_panel'
    )
    obj_true_mat.material_names[obj_true_mat.material_names == 'none'] = 'solar_panel'

    obj_simple_mat = mr.load_obj(obj_simple.file_name)
    obj_simple_mat.material_names[obj_simple_mat.material_names == 'starlink_panel'] = (
        'solar_panel'
    )

    # %%
    # Plotting the objects themselves
    p = pv.Plotter(window_size=(1200, 800))
    mrv.render_spaceobject(
        p, obj_true_mat, origin=(-6, 0, 0), scalars=obj_true_mat.material_names
    )
    mrv.render_spaceobject(
        p, obj_simple_mat, origin=(6, 0, 0), scalars=obj_simple_mat.material_names
    )
    p.add_axes(shaft_length=1.0, label_size=(0.5, 0.2), line_width=3)
    p.camera.position = (0, 0, 50)
    p.camera.up = (1, 0, 0)
    p.show()
    p.screenshot(
        '/Users/liamrobinson/Documents/maintained-research/mirage/examples/13-attitude-2024/figs/box_wings.png'
    )

    invert_obj_path = '/Users/liamrobinson/Documents/maintained-research/mirage-models/accurate_sats/box_wing_antenna_quad.obj'

    mrp.generate_light_curves(
        station,
        obj_true,
        obj_simple,
        invert_obj_path,
        lc_length=length,
        lc_cadence=cadence,
        date0=date0,
        s0=s0,
        w0=w0,
        itensor=itensor,
        self_shadowing=True,
        trial_name='box_wing',
        model_name='Box-Wing',
        make_plots=True,
        integration_time_s=integration_time_s,
    )


if __name__ == '__main__':
    main()
