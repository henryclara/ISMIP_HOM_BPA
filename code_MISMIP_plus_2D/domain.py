from firedrake import *
import numpy as np
import os
import time
from config import *

slice_y = 60000.0
y = Constant(slice_y)

restart_mesh_file = os.environ.get("RESTART_MESH_FILE")
if restart_mesh_file:
    from firedrake import CheckpointFile
    with CheckpointFile(restart_mesh_file, "r") as afile:
        mesh3D = afile.load_mesh(name="firedrake_default_extruded")
else:
    base = IntervalMesh(nx, Lx)
    mesh3D = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)

x, sigma = SpatialCoordinate(mesh3D)

Xref = Function(mesh3D.coordinates.function_space(), name="Xref")
Xref.interpolate(SpatialCoordinate(mesh3D))
xref, sigmaref = split(Xref)

eps = Constant(1e-6)
