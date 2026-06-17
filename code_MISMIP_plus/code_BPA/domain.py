from firedrake import *
import numpy as np
import os
import time
from config import *

restart_mesh_file = os.environ.get("RESTART_MESH_FILE")
if restart_mesh_file:
    from firedrake import CheckpointFile
    with CheckpointFile(restart_mesh_file, "r") as afile:
        mesh3D = afile.load_mesh(name="firedrake_default_extruded")
else:
    base = RectangleMesh(nx, ny, Lx, Ly)
    base.coordinates.dat.data[:, 1] += Ly_full / 2.0
    mesh3D = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)

x, y, sigma = SpatialCoordinate(mesh3D)

Xref = Function(mesh3D.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh3D))
xref, yref, sigmaref = split(Xref)

eps = Constant(1e-6)
