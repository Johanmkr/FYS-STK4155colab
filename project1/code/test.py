import numpy as np
import sys
import os



sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from designMatrix import DesignMatrix


Nx, Ny = 100, 100
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
x, y = np.meshgrid(x, y)

dM = DesignMatrix(5)

dM.create_X(x, y)

print(dM)