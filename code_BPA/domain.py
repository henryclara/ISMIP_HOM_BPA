from firedrake import *
import numpy as np
import time
from config import *

base = RectangleMesh(nx, ny, Lx, Ly)
base.coordinates.dat.data[:, 1] += Ly_full / 2.0
mesh3D = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)
x, y, sigma = SpatialCoordinate(mesh3D)

Xref = Function(mesh3D.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh3D))
xref, yref, sigmaref = split(Xref)

eps = Constant(1e-6)
