
import numpy

from matplotlib import pyplot

from baselines.common import plot_util as pu

results = pu.load_results('./tmp')
r = results[0]

fig, ax = pyplot.subplots()
ax.plot(numpy.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
pyplot.show()
