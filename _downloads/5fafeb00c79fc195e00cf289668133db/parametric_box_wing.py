"""
Parametric Box-Wing Inversion
=============================

"""

import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.io import loadmat
from scipy.optimize import minimize

import mirage as mr
import mirage.vis as mrv

vars_enum = [
    'x_scale',
    'y_scale',
    'z_scale',
    'wing_area',
    'cd_wing',
    'cd_box',
    'cs_box',
    'cs_wing',
    'n_box',
    'n_wing',
]


def construct_from_y(attitude, y: np.ndarray, knowns: dict):
    for k, v in knowns.items():
        y[vars_enum.index(k)] = v
    cd_wing = np.clip(y[4], 0, 1)
    cd_box = np.clip(y[5], 0, 1)
    cs_box = np.clip(y[6], 0, 1)
    cs_wing = np.clip(y[7], 0, 1)
    y = np.clip(y, 1e-8, np.inf)
    n_box, n_wing = y[8], y[9]
    brdf_box = mr.Brdf(
        name='cook-torrance', cd=cd_box, cs=cs_box, n=n_box, validate=False
    )
    brdf_wing = mr.Brdf(
        name='cook-torrance', cd=cd_wing, cs=cs_wing, n=n_wing, validate=False
    )
    return BoxWingParametric(
        attitude,
        brdf_box,
        brdf_wing,
        x_scale=y[0],
        y_scale=y[1],
        z_scale=y[2],
        wing_area=y[3],
    )


class BoxWingParametric:
    def __init__(
        self,
        attitude: mr.AlignedAndConstrainedAttitude,
        brdf_box: mr.Brdf,
        brdf_wing: mr.Brdf,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
        wing_area: float = 1.0,
    ):
        self.attitude = attitude
        self.brdf_box = brdf_box
        self.brdf_wing = brdf_wing
        self._cube_template = mr.SpaceObject('cube.obj')
        self._cube_template.v /= np.sqrt(self._cube_template.unique_areas[0])
        self.box = mr.SpaceObject(
            vertices_and_faces=(self._cube_template.v, self._cube_template.f)
        )
        self.scales = [x_scale, y_scale, z_scale]
        self.wing_area = wing_area

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, scales):
        self._scales = scales
        self.box.v = self._cube_template.v.copy()
        self.box.v[:, 0] = self.box.v[:, 0] * scales[0]
        self.box.v[:, 1] = self.box.v[:, 1] * scales[1]
        self.box.v[:, 2] = self.box.v[:, 2] * scales[2]
        self.box = mr.SpaceObject(vertices_and_faces=(self.box.v, self.box.f))

    def eval(
        self,
        dates: np.ndarray[datetime.datetime, Any],
        ovi: np.ndarray,
        jds: np.ndarray = None,
        eci_to_body: np.ndarray = None,
        vc_eci: np.ndarray = None,
    ) -> np.ndarray:
        if jds is None:
            jds = mr.date_to_jd(dates)
        if eci_to_body is None:
            eci_to_body = self.attitude.dcms_at_dates(dates)
        if vc_eci is None:
            vc_eci = self.attitude.const_interpolator(jds)
        ovb = mr.stack_mat_mult_vec(eci_to_body, ovi)
        vc_body = mr.stack_mat_mult_vec(eci_to_body, vc_eci)

        box_lc = self.box.convex_light_curve(self.brdf_box, svb=vc_body, ovb=ovb)
        wing_lc = (
            self.wing_area
            * self.brdf_wing.eval_normalized_brightness(
                L=vc_body, O=ovb, N=vc_body
            ).flatten()
        )

        return box_lc, wing_lc

    def optimize(
        self,
        dates: np.ndarray[datetime.datetime, Any],
        ovi: np.ndarray,
        lc_unit_observed: np.ndarray,
        knowns: dict = None,
    ):
        if knowns is None:
            knowns = dict()
        if hasattr(lc_unit_observed, 'mask'):
            valid_inds = ~lc_unit_observed.mask
        else:
            valid_inds = ~np.isnan(lc_unit_observed)

        jds = mr.date_to_jd(dates)
        eci_to_body = self.attitude.dcms_at_dates(dates)
        vc_eci = self.attitude.const_interpolator(jds)

        def objective(y: np.ndarray) -> float:
            bwpi = construct_from_y(self.attitude, y, knowns)
            box_lc, wing_lc = bwpi.eval(dates, ovi, jds, eci_to_body, vc_eci)
            err = np.linalg.norm(
                lc_unit_observed[valid_inds] - (box_lc + wing_lc)[valid_inds]
            )
            return err

        print(y_from_bwp(self))

        opt_sol = minimize(objective, y_from_bwp(self), options={'maxiter': 1})
        return construct_from_y(self.attitude, opt_sol.x, knowns)

    def __repr__(self):
        return f'BoxWingParametric(attitude={self.attitude}, brdf_box={self.brdf_box}, brdf_wing={self.brdf_wing}, x_scale={self.scales[0]}, y_scale={self.scales[1]}, z_scale={self.scales[2]}, wing_area={self.wing_area})'


def y_from_bwp(bwp: BoxWingParametric):
    return np.array(
        [
            bwp.scales[0],
            bwp.scales[1],
            bwp.scales[2],
            bwp.wing_area,
            bwp.brdf_wing.cd,
            bwp.brdf_box.cd,
            bwp.brdf_box.cs,
            bwp.brdf_wing.cs,
            bwp.brdf_box.n,
            bwp.brdf_wing.n,
        ]
    )


orbit_sol_path = '/Volumes/Data 1/imgs/pogs/2022/2022-09-18_GPS_PRN14/ProcessedData/Fitted_Orbits/OrbitSolutions.mat'
orbit_mat = loadmat(orbit_sol_path)
sol_mat = {
    k: orbit_mat['CuratedObjects'][0][k][0].squeeze()
    for k in orbit_mat['CuratedObjects'][0].dtype.names
}
sol_mat['JD'] = np.sum(sol_mat['JD'], axis=0)
dates = mr.jd_to_date(sol_mat['JD'])
rmag = mr.vecnorm(sol_mat['r'].T).flatten()
isun = mr.total_solar_irradiance_at_dates(dates)
lc_norm_true = sol_mat['flux'] / isun * rmag**2 * 1e6

epsecs = mr.date_to_epsec(dates)
ephrs = (epsecs - epsecs[0]) / 3600

station = mr.Station()

station.constraints = [
    mr.SnrConstraint(3),
    mr.ElevationConstraint(10),
    mr.TargetIlluminatedConstraint(),
    mr.ObserverEclipseConstraint(station),
    mr.VisualMagnitudeConstraint(20),
    mr.MoonExclusionConstraint(10),
]

obj = mr.SpaceObject('matlib_gps_iii.obj', identifier='NAVSTAR 80 (USA 309)')
r_obj_j2k = obj.propagate(dates)

sv = mr.sun(dates)
nadir = -mr.hat(r_obj_j2k)
attitude = mr.AlignedAndConstrainedAttitude(
    v_align=nadir,
    v_const=sv,
    dates=dates,
    axis_order=(2, 0, 1),
)

vars_enum = [
    'x_scale',
    'y_scale',
    'z_scale',
    'wing_area',
    'cd_wing',
    'cd_box',
    'cs_box',
    'cs_wing',
    'n_box',
    'n_wing',
]

brdf_box = mr.Brdf('phong', cd=0.32, cs=0.27, n=0.165)
brdf_wing = mr.Brdf('phong', cd=0.0, cs=1.0, n=0.273)

station_pos_eci = station.j2000_at_dates(dates)
object_pos_eci = obj.propagate(dates)
ovi = mr.hat(station_pos_eci - object_pos_eci)
ovb = mr.stack_mat_mult_vec(attitude.dcms_at_dates(dates), ovi)

wing_area = 28.5212  # m^2
x_scale = 2.4638  # m
y_scale = 3.4036  # m
z_scale = 1.778  # m
knowns = dict(
    wing_area=wing_area,
    x_scale=x_scale,
    y_scale=y_scale,
    z_scale=z_scale,
)

bwp = BoxWingParametric(attitude, brdf_box, brdf_wing)
box_lc, wing_lc = bwp.eval(dates, ovb)

fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.5)
blc = plt.plot(ephrs, box_lc + wing_lc, label='Fit')
plt.scatter(ephrs, lc_norm_true, label='Observed', s=1)
mrv.texit('Parametric Box-Wing Fit', 'Date', 'Normalized Brightness')
plt.legend()


def _update_blc():
    box_lc, wing_lc = bwp.eval(dates, ovi)
    blc[0].set_ydata(box_lc + wing_lc)
    plt.draw()


def update_box_n(val):
    bwp.brdf_box.n = val
    _update_blc()


def update_wing_n(val):
    bwp.brdf_wing.n = val
    _update_blc()


def update_box_cd(val):
    bwp.brdf_box.cd = val
    _update_blc()


def update_wing_cd(val):
    bwp.brdf_wing.cd = val
    _update_blc()


def update_box_cs(val):
    bwp.brdf_box.cs = val
    _update_blc()


def update_wing_cs(val):
    bwp.brdf_wing.cs = val
    _update_blc()


def update_wing_area(val):
    bwp.wing_area = val
    _update_blc()


def update_x_scale(val):
    bwp.scales = [val, bwp.scales[1], bwp.scales[2]]
    _update_blc()


def update_y_scale(val):
    bwp.scales = [bwp.scales[0], val, bwp.scales[2]]
    _update_blc()


def update_z_scale(val):
    bwp.scales = [bwp.scales[0], bwp.scales[1], val]
    _update_blc()


dy = 0.07

slider_box_cd = Slider(
    plt.axes([0.1, 0.01, 0.2, 0.03]), 'box_cd', 0, 1, valinit=brdf_box.cd
)
slider_box_cd.on_changed(update_box_cd)

slider_box_cs = Slider(
    plt.axes([0.4, 0.01, 0.2, 0.03]), 'box_cs', 0, 1, valinit=brdf_box.cs
)
slider_box_cs.on_changed(update_box_cs)

slider_box_n = Slider(
    plt.axes([0.7, 0.01, 0.2, 0.03]), 'box_n', 1e-8, 2, valinit=brdf_box.n
)
slider_box_n.on_changed(update_box_n)

slider_wing_cd = Slider(
    plt.axes([0.1, dy, 0.2, 0.03]), 'wing_cd', 0, 1, valinit=brdf_wing.cd
)
slider_wing_cd.on_changed(update_wing_cd)

slider_wing_cs = Slider(
    plt.axes([0.4, dy, 0.2, 0.03]), 'wing_cs', 0, 1, valinit=brdf_wing.cs
)
slider_wing_cs.on_changed(update_wing_cs)

slider_wing_n = Slider(
    plt.axes([0.7, dy, 0.2, 0.03]), 'wing_n', 1e-8, 2, valinit=brdf_wing.n
)
slider_wing_n.on_changed(update_wing_n)

slider_x_scale = Slider(
    plt.axes([0.1, 2 * dy, 0.2, 0.03]), 'x_scale', 1e-8, 5, valinit=x_scale
)
slider_x_scale.on_changed(update_x_scale)

slider_y_scale = Slider(
    plt.axes([0.4, 2 * dy, 0.2, 0.03]), 'y_scale', 1e-8, 5, valinit=y_scale
)
slider_y_scale.on_changed(update_y_scale)

slider_z_scale = Slider(
    plt.axes([0.7, 2 * dy, 0.2, 0.03]), 'z_scale', 1e-8, 5, valinit=z_scale
)
slider_z_scale.on_changed(update_z_scale)

slider_wing_area = Slider(
    plt.axes([0.1, 3 * dy, 0.2, 0.03]), 'wing_area', 1e-8, 50, valinit=wing_area
)
slider_wing_area.on_changed(update_wing_area)


def _update_sliders_from_bwp(bwpi: BoxWingParametric):
    slider_box_cd.set_val(bwpi.brdf_box.cd)
    slider_box_cs.set_val(bwpi.brdf_box.cs)
    slider_box_n.set_val(bwpi.brdf_box.n)
    slider_wing_cd.set_val(bwpi.brdf_wing.cd)
    slider_wing_cs.set_val(bwpi.brdf_wing.cs)
    slider_wing_n.set_val(bwpi.brdf_wing.n)
    slider_x_scale.set_val(bwpi.scales[0])
    slider_y_scale.set_val(bwpi.scales[1])
    slider_z_scale.set_val(bwpi.scales[2])
    slider_wing_area.set_val(bwpi.wing_area)
    plt.draw()


def _opt_button_callback(*args):
    global bwp
    bwp = bwp.optimize(dates, ovi, lc_norm_true, knowns=knowns)
    _update_sliders_from_bwp(bwp)
    _update_blc()


opt_button = Button(plt.axes([0.4, 3 * dy, 0.2, 0.03]), 'Optimize')
opt_button.on_clicked(_opt_button_callback)
_opt_button_callback()

# bwp_reconstructed = construct_from_y(attitude, y_from_bwp(bwp), knowns)
# print(repr(bwp_reconstructed))
# print(repr(bwp))

plt.show()
