import matplotlib.pyplot as plt
import numpy as np

x, y = np.meshgrid(np.arange(0, 100, 0.1), np.arange(0, 100, 0.1))
Z = np.ones(np.shape(x))
interior = np.sqrt((x-50)**2+(y-50)**2)<10
Z[interior] = 0
plt.contourf(x, y, Z)
plt.show()
