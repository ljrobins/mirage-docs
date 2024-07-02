"""
Light Time Delay
================
Calculating the delay in observations due to light travel time
"""

import mirage as mr

station = mr.Station(preset='pogs')
dates = mr.date_linspace(mr.now(), mr.now() + mr.minutes(1), 1e3)
obj = mr.SpaceObject('cube.obj', identifier=44333)

station_pos_j2000 = station.j2000_at_dates(dates)
obj_pos_j2000 = obj.propagate(dates)

tau = mr.light_time_delay(station_pos_j2000, obj_pos_j2000)

print(tau)
