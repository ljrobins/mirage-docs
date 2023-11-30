"""
Parametric Box-Wing Inversion
=============================

"""

import datetime

import numpy as np
from scipy.optimize import minimize

import mirage as mr


def construct_from_y(attitude, y: np.ndarray):
    cd_wing = np.clip(y[4], 0, 1)
    cd_box = np.clip(y[5], 0, 1)
    y = np.clip(y, 1e-8, np.inf)
    n_box, n_wing = y[6], y[7]
    brdf_box = mr.Brdf(name="phong", cd=cd_box, cs=1 - cd_box, n=n_box)
    brdf_wing = mr.Brdf(name="phong", cd=cd_wing, cs=1 - cd_wing, n=n_wing)
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
        self.scales = (x_scale, y_scale, z_scale)
        cube_template = mr.SpaceObject("cube.obj")
        cube_template.v[:, 0] *= x_scale
        cube_template.v[:, 1] *= y_scale
        cube_template.v[:, 2] *= z_scale

        self.box = mr.SpaceObject(vertices_and_faces=(cube_template.v, cube_template.f))
        self.wing_area = wing_area

    def eval(self, dates: np.ndarray[datetime.datetime], ovi: np.ndarray) -> np.ndarray:
        jd = mr.date_to_jd(dates)
        eci_to_body = self.attitude.dcms_at_dates(dates)

        vc_eci = self.attitude.const_interpolator(jd)

        ovb = mr.stack_mat_mult_vec(eci_to_body, ovi)
        vc_body = mr.stack_mat_mult_vec(eci_to_body, vc_eci)

        # pl = pv.Plotter()
        # mrv.plot3(pl, vc_body)
        # mrv.plot3(pl, ovb)
        # pl.show()
        # eneddds

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
        dates: np.ndarray[datetime.datetime],
        ovi: np.ndarray,
        lc_unit_observed: np.ndarray,
    ):
        valid_inds = ~lc_unit_observed.mask

        def objective(y: np.ndarray) -> float:
            bwpi = construct_from_y(self.attitude, y)
            box_lc, wing_lc = bwpi.eval(dates, ovi)
            err = np.linalg.norm(
                lc_unit_observed[valid_inds] - (box_lc + wing_lc)[valid_inds]
            )
            print(err)
            return err

        opt_sol = minimize(objective, 0.5 * np.ones(8), options={"maxiter": 20})
        return construct_from_y(self.attitude, opt_sol.x)


date_start = mr.utc(2023, 5, 20, 20, 45, 0)
(dates, epsecs) = mr.date_linspace(
    date_start - mr.days(3), date_start, 1e3, return_epsecs=True
)

station = mr.Station()

station.constraints = [
    mr.SnrConstraint(3),
    mr.ElevationConstraint(10),
    mr.TargetIlluminatedConstraint(),
    mr.ObserverEclipseConstraint(station),
    mr.VisualMagnitudeConstraint(20),
    mr.MoonExclusionConstraint(10),
    mr.HorizonMaskConstraint(station),
]

obj = mr.SpaceObject("matlib_hylas4.obj", identifier="superbird 6")
# obj = mr.SpaceObject("matlib_tess.obj", identifier="NAVSTAR 62 (USA 201)")
r_obj_j2k = obj.propagate(dates)

sv = mr.sun(dates)
nadir = -mr.hat(r_obj_j2k)
attitude = mr.AlignedAndConstrainedAttitude(
    v_align=nadir,
    v_const=sv,
    dates=dates,
    axis_order=(1, 2, 0),
)
brdf = mr.Brdf("phong")

(lc_ccd_signal, aux_data) = station.observe_light_curve(
    obj,
    attitude,
    brdf,
    dates,
    model_scale_factor=1,
    use_engine=True,
    show_window=True,
    frame_rate=1000,
    instances=1,
    rotate_panels=True,
)

brdf_bwp = mr.Brdf("phong", cd=0.5, cs=0.5, n=10)
bwp = BoxWingParametric(attitude, brdf_bwp, brdf_bwp)

lc_noisy_adu = lc_ccd_signal()

lc_noisy_irrad = lc_noisy_adu / aux_data["sint"]
lc_noisy_unit_irrad = lc_noisy_irrad * (aux_data["rmag_station_to_sat"] * 1e3) ** 2
invalid_inds = np.isnan(lc_noisy_unit_irrad)

ovi = mr.hat(aux_data["station_pos_eci"] - aux_data["object_pos_eci"])
ovb = mr.stack_mat_mult_vec(attitude.dcms_at_dates(dates), ovi)

bwp_opt = bwp.optimize(dates, ovi, lc_noisy_unit_irrad)
box_lc_opt, wing_lc_opt = bwp_opt.eval(dates, ovi)

print(
    f"{bwp_opt.wing_area=}",
    f"{bwp_opt.scales=}",
    f"{bwp_opt.brdf_box.cd=}",
    f"{bwp_opt.brdf_wing.cd=}",
    f"{bwp_opt.brdf_box.n=}",
    f"{bwp_opt.brdf_wing.n=}",
    f"{bwp_opt.brdf_box.cs=}",
    f"{bwp_opt.brdf_wing.cs=}",
)

import matplotlib.pyplot as plt

plt.plot(dates, lc_noisy_unit_irrad)
plt.plot(dates, aux_data["lc_clean_norm"])
plt.plot(dates, wing_lc_opt + box_lc_opt)
# plt.plot(dates, box_lc_opt)
# plt.plot(dates, wing_lc_opt)
plt.legend(["noisy_true", "clean_true", "recon"])
plt.show()
