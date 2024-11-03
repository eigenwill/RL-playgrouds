import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

x = numpy.linspace(0, 10, 100)
y = numpy.sin(x)

plt.plot(x, y)
plt.show()