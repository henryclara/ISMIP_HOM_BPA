from firedrake import *
import numpy as np
import os
from config import *

# ------------------------------------------------------------
# Load mesh: either from a checkpoint (restart) or from Gmsh
# ------------------------------------------------------------
restart_mesh_file = os.environ.get("RESTART_MESH_FILE")
if restart_mesh_file:
    from firedrake import CheckpointFile
    import numpy as _np

    with CheckpointFile(restart_mesh_file, "r") as afile:
        try:
            mesh3D = afile.load_mesh(name="firedrake_default_extruded", reorder=False)
        except Exception:
            mesh3D = afile.load_mesh(reorder=False)

    base = mesh3D._base_mesh
    coords = mesh3D.coordinates.dat.data_ro.copy()
    ymin = float(coords[:, 1].min())
    if ymin < 1.0e-6:
        shift = float(Ly_full / 2.0)
        coords[:, 1] += shift
        mesh3D.coordinates.dat.data[:] = coords

        base_coords = base.coordinates.dat.data_ro.copy()
        base_coords[:, 1] += shift
        base.coordinates.dat.data[:] = base_coords

else:
    # New simulation: load 2D Gmsh mesh and extrude
    base = Mesh(f"Meshes/mesh_res_{str(int(coarse_res))}_{str(int(refined_res))}_unstructured.msh")

    # Keep this if you want the same y-coordinate shift
    # as in your previous RectangleMesh setup
    base.coordinates.dat.data[:, 1] += Ly_full / 2.0

    # Extrude vertically
    mesh3D = ExtrudedMesh(base, layers=nz, layer_height=1.0 / nz)

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
