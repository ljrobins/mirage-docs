"""
Parametric Space Object Models
==============================

"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# _, (obj, _) = mr.parametric_shape_hypercube(np.zeros(5)+0.5, 'scyl_wing')

# p = pv.Plotter()
# mrv.render_spaceobject(p, obj)
# p.show_bounds()
# p.show()
# endd
shape_class_names = dict(
    scyl_wing='Cylinder-wing',
    box_wing='Box-Wing',
    sbox_wing='Single-Wing Box',
    rocket_body='Rocket Body',
    blob='Generic Debris',
)


for _class in ['scyl_wing', 'sbox_wing', 'rocket_body', 'box_wing', 'blob']:
    d = 3
    p = pv.Plotter(shape=(d, d), window_size=(1500, 1500))

    for i in range(d):
        for j in range(d):
            x = np.random.rand(mr.parametric_shape_state_dimension(_class))
            sx, (obj, mats) = mr.parametric_shape_hypercube(x, _class)

            p.subplot(i, j)
            if i == 0 and j == 0:
                actor = p.add_text(
                    shape_class_names[_class],
                    position='upper_left',
                    color='k',
                    shadow=True,
                    font_size=26,
                )

            mrv.render_spaceobject(
                p,
                obj,
                scalars=mats,
                show_scalar_bar=False,
                cmap='rainbow',
                preference='cell',
            )
            p.enable_3_lights()

            if _class == 'rocket_body':
                p.camera.zoom(0.8)

    p.show()
