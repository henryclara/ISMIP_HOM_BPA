from firedrake import *
import numpy as np
import os
import time
from config import *

# ------------------------------------------------------------
# New simulation: load 2D Gmsh mesh
# ------------------------------------------------------------
base = Mesh("mesh_res_2000_500.msh")

# Keep this if you want the same y-coordinate shift
# as in your previous RectangleMesh setup
base.coordinates.dat.data[:, 1] += Ly_full / 2.0

    # Extrude vertically
mesh3D = ExtrudedMesh(base,layers=nz,layer_height=1.0 / nz)

# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------
x, y, z = SpatialCoordinate(mesh3D)


Xref = Function(
    mesh3D.coordinates.function_space(),
    name="Xref"
)

Xref.interpolate(SpatialCoordinate(mesh3D))

xref, yref, zref = split(Xref)

eps = Constant(1e-6)
