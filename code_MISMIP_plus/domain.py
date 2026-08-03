from firedrake import *
import os
from config import *

restart_mesh_file = os.environ.get("RESTART_MESH_FILE")
if restart_mesh_file:
    from firedrake import CheckpointFile
    with CheckpointFile(restart_mesh_file, "r") as afile:
        mesh3D = afile.load_mesh(name="firedrake_default")
else:
    mesh3D = RectangleMesh(nx, nz, Lx, Lz)

x, z_ref = SpatialCoordinate(mesh3D)

Xref = Function(mesh3D.coordinates.function_space(), name="Xref")
Xref.interpolate(as_vector([x, z_ref]))
xref, z_ref_split = split(Xref)

eps = Constant(1e-6)
