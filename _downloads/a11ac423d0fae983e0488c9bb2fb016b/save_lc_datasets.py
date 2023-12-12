"""
Saving Light Curve Datasets
===========================
"""

import datetime
import json
import os
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv


def save_lc_info(
    dates: np.ndarray[datetime.datetime],
    obj: mr.SpaceObject,
    attitude,
    aux_data: dict,
    rotate_panels: bool,
):
    if not os.path.exists("lc_database"):
        os.mkdir("lc_database")
    save_dict = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in aux_data.items()
    }
    save_dict["obj_file_name"] = obj.file_name
    save_dict["rotate_panels"] = rotate_panels
    save_dict["jds"] = mr.date_to_jd(dates).tolist()
    save_dict["attitude_type"] = repr(attitude)
    with open(os.path.join("lc_database", f"{abs(hash(dates[0]))}.lc_info"), "w") as f:
        json.dump(save_dict, f, indent=2)


def aligned_nadir_constrained_sun_attitude(
    obj: mr.SpaceObject, dates: np.ndarray[datetime.datetime]
) -> mr.AlignedAndConstrainedAttitude:
    r_obj_j2k = obj.propagate(dates)
    sv = mr.sun(dates)
    nadir = -mr.hat(r_obj_j2k)
    return mr.AlignedAndConstrainedAttitude(
        v_align=nadir, v_const=sv, dates=dates, axis_order=(1, 2, 0)
    )


def random_brdf() -> mr.Brdf:
    return mr.Brdf("phong")


def random_station() -> mr.Station:
    station = mr.Station()
    station.constraints = [
        mr.SnrConstraint(3),
        mr.ElevationConstraint(10),
        mr.TargetIlluminatedConstraint(),
        mr.ObserverEclipseConstraint(station),
        mr.VisualMagnitudeConstraint(20),
    ]
    return station


mr.set_model_directory("/Users/liamrobinson/Documents/Light-Curve-Models/accurate_sats")
space_objects = [
    mr.SpaceObject("matlib_tdrs.obj", identifier=19548),
    mr.SpaceObject("matlib_astra.obj", identifier=26853),  # ASTRA 2C
    mr.SpaceObject("matlib_hylas4.obj", identifier=44333),  # AT&T T-16
    mr.SpaceObject("matlib_hispasat_30w-6.obj", identifier=44333),  # AT&T T-16
    mr.SpaceObject("matlib_telstar19v.obj", identifier=44333),  # AT&T T-16
    # mr.SpaceObject('matlib_tess.obj', identifier=43435),
    # mr.SpaceObject('matlib_landsat8.obj', identifier=39084),
    mr.SpaceObject("matlib_saturn_v_sii.obj", identifier=43652),  # ATLAS 5 CENTAUR DEB
    # mr.SpaceObject('matlib_starlink_v1.obj', identifier=44743), # STARLINK 1038
]


def random_spaceobject() -> mr.SpaceObject:
    return random.choice(space_objects)


def random_dates() -> np.ndarray:
    idate = mr.now() - mr.days(np.random.rand() * 365.25)
    fdate = idate + mr.days(1)
    return mr.date_arange(idate, fdate, mr.seconds(10))


def random_attitude(
    dates: np.ndarray[datetime.datetime], obj: mr.SpaceObject
) -> Union[
    mr.RbtfAttitude, mr.SpinStabilizedAttitude, mr.AlignedAndConstrainedAttitude
]:
    type_choice = np.random.rand()
    if type_choice < 1 / 3:
        q0 = mr.rand_quaternions(1)
        w0 = mr.rand_points_in_ball(1e-3, 1)
        itensor = np.diag(np.random.rand(3))
        attitude = mr.RbtfAttitude(w0, q0, itensor)
    elif type_choice < 2 / 3:
        rot_rate = np.random.rand() * 1e-3
        intertial_rot_axis = mr.rand_unit_vectors(1)
        jd0 = mr.date_to_jd(dates[0])
        rot_angle0 = 2 * np.pi * np.random.rand()
        attitude = mr.SpinStabilizedAttitude(
            rot_rate, intertial_rot_axis, jd0, rot_angle0
        )
    else:
        attitude = aligned_nadir_constrained_sun_attitude(obj, dates)

    return attitude


def generate_light_curve():
    dates = random_dates()
    obj = random_spaceobject()
    station = random_station()
    brdf = random_brdf()
    attitude = random_attitude(dates, obj)

    rotate_panels = not any([x in obj.file_name for x in ["orion", "saturn"]])
    lc_sampler, aux_data = station.observe_light_curve(
        obj,
        attitude,
        brdf,
        dates,
        use_engine=True,
        rotate_panels=rotate_panels,
        show_window=False,
        silent=True,
        instances=4,
    )

    print(obj.file_name, attitude.__class__)
    plt.scatter(
        dates, mr.irradiance_to_apparent_magnitude(lc_sampler() / aux_data["sint"]), s=2
    )
    mrv.texit(f"{obj.file_name[:-4]}", "Date", "Apparent Magnitude")
    # plt.show()

    if np.any(lc_sampler()):
        save_lc_info(dates, obj, attitude, aux_data, rotate_panels)


for i in range(10):
    generate_light_curve()
