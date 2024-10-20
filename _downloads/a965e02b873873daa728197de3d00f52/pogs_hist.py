"""
POGS Status
===========

The weather status of the Purdue Optical Ground Station
"""

import datetime
import itertools

import matplotlib.pyplot as plt

csv = """2024-07-03 22:15:04.958184,clouds
2024-07-04 07:59:04.628433,daylight
2024-07-04 22:15:05.176451,clouds
2024-07-05 00:53:04.358703,open
2024-07-05 08:06:04.931694,daylight
2024-07-05 22:15:04.495289,clouds
2024-07-05 23:46:04.000372,fog
2024-07-06 08:00:05.210266,daylight
2024-07-06 22:14:04.767616,clouds
2024-07-06 22:22:04.199623,open
2024-07-07 08:12:04.496914,daylight
2024-07-07 22:14:04.568198,clouds
2024-07-07 22:22:04.940400,open
2024-07-07 22:57:04.047537,clouds
2024-07-08 08:01:04.548877,daylight
2024-07-08 22:14:04.105869,clouds
2024-07-08 22:26:04.537847,fog
2024-07-09 08:01:05.409548,daylight
2024-07-09 22:14:04.518691,clouds
2024-07-10 05:26:05.673129,fog
2024-07-10 08:02:04.294682,daylight
2024-07-10 22:13:04.740676,clouds
2024-07-10 22:32:04.169360,rain
2024-07-10 23:42:05.737735,clouds
2024-07-11 00:16:04.573813,open
2024-07-11 01:16:05.626543,fog
2024-07-11 02:47:04.845346,open"""

vals = [x.split(',') for x in csv.split('\n')]
vals = [
    {'date': datetime.datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S.%f'), 'status': x[1]}
    for x in vals
]

cols = {
    'fog': 'lightgray',
    'clouds': 'gray',
    'rain': 'b',
    'open': 'g',
    'daylight': 'y',
}

seen = {}
plt.figure(figsize=(10, 3))
for pair in itertools.pairwise(vals):
    start_date = pair[0]['date']
    end_date = pair[1]['date']
    status = pair[1]['status']

    label = status if status not in seen else '_'
    plt.fill_between([start_date, end_date], 0, 1, color=cols[status], label=label)
    seen[status] = True

plt.yticks([])
plt.autoscale(enable=True, tight=True)
plt.title('POGS Status')
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.tight_layout()
plt.show()
