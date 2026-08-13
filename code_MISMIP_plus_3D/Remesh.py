from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from firedrake import *
from config import *

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

INPUT_RESTART = Path(
    "SimulationsOld/MISMIP_output_theta1_dt5_GL_predFalse_nx_80_nz_5/"
    "restart_t8000.h5"
)

OUTPUT_RESTART = Path(
    f"Simulations/remesh_mesh_res_{str(int(coarse_res))}_{str(int(refined_res))}/restart_t8000.h5"
)

# Existing *horizontal* unstructured Gmsh mesh.
# This mesh is loaded directly; this script does not reconstruct or modify the
# horizontal triangulation.
TARGET_BASE_MESH = Path("Meshes/mesh_res_2000_1000.msh")

# Number of vertical layers used when extruding the supplied horizontal mesh.
NZ = 10

MINIMUM_THICKNESS = 10.0


# -----------------------------------------------------------------------------
# TARGET MESH
# -----------------------------------------------------------------------------

def load_target_base_mesh():
    """Load the existing 2-D horizontal target mesh exactly as supplied."""
    if not TARGET_BASE_MESH.is_file():
        raise FileNotFoundError(
            f"Target base mesh does not exist:\n{TARGET_BASE_MESH}"
        )

    print(f"Loading existing unstructured target mesh: {TARGET_BASE_MESH}")
    return Mesh(str(TARGET_BASE_MESH), reorder=False)


# -----------------------------------------------------------------------------
# INTERPOLATION HELPERS
# -----------------------------------------------------------------------------

def _scaled_points(source_points, target_points):
    """
    Scale coordinates to O(1) before the SciPy Delaunay triangulation.

    This is particularly important for the 3-D interpolation because x and y
    are O(10^5 m), while sigma lies between 0 and 1.
    """
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    pmin = np.min(source_points, axis=0)
    pmax = np.max(source_points, axis=0)
    scale = pmax - pmin
    scale[scale == 0.0] = 1.0

    source_scaled = (source_points - pmin) / scale
    target_scaled = (target_points - pmin) / scale
    return source_scaled, target_scaled


def interpolate_values(source_points, source_values, target_points, label="field"):
    """
    Interpolate scalar or vector nodal data from source_points to target_points.

    Linear interpolation is used inside the convex hull. Any target points that
    fall just outside the hull because of roundoff/mesh-boundary differences are
    filled using nearest-neighbour interpolation.
    """
    source_scaled, target_scaled = _scaled_points(source_points, target_points)

    linear = LinearNDInterpolator(
        source_scaled,
        source_values,
        fill_value=np.nan,
    )

    result = np.asarray(linear(target_scaled), dtype=float)

    if result.ndim == 1:
        missing = ~np.isfinite(result)
    else:
        missing = ~np.all(np.isfinite(result), axis=1)

    nmissing = int(np.count_nonzero(missing))
    if nmissing:
        print(
            f"  {label}: {nmissing}/{len(target_points)} target points "
            "outside linear interpolation hull; using nearest neighbour"
        )
        nearest = NearestNDInterpolator(source_scaled, source_values)
        result[missing] = nearest(target_scaled[missing])

    return result


def coordinates_2d(function_space):
    """Return x-y coordinates at the DOFs of a depth-independent space."""
    mesh = function_space.mesh()
    xyz = SpatialCoordinate(mesh)
    x, y = xyz[0], xyz[1]

    x_field = Function(function_space).interpolate(x)
    y_field = Function(function_space).interpolate(y)

    return np.column_stack((x_field.dat.data_ro, y_field.dat.data_ro))


def source_coordinates_3d(function_space, zb, thick):
    """
    Return source coordinates in (x, y, sigma), with

        sigma = (z - zb) / H.

    Interpolating velocity in sigma coordinates transfers the vertical profile
    between meshes without confusing a change in geometry with a change in
    vertical position within the ice column.
    """
    mesh = function_space.mesh()
    x, y, z = SpatialCoordinate(mesh)

    x_field = Function(function_space).interpolate(x)
    y_field = Function(function_space).interpolate(y)
    sigma_field = Function(function_space).interpolate((z - zb) / thick)

    return np.column_stack(
        (
            x_field.dat.data_ro,
            y_field.dat.data_ro,
            sigma_field.dat.data_ro,
        )
    )


def target_coordinates_3d(function_space):
    """Return target (x, y, sigma) coordinates before ALE deformation."""
    mesh = function_space.mesh()
    x, y, sigma = SpatialCoordinate(mesh)

    x_field = Function(function_space).interpolate(x)
    y_field = Function(function_space).interpolate(y)
    sigma_field = Function(function_space).interpolate(sigma)

    return np.column_stack(
        (
            x_field.dat.data_ro,
            y_field.dat.data_ro,
            sigma_field.dat.data_ro,
        )
    )


def report_range(name, old_values, new_values):
    old_values = np.asarray(old_values)
    new_values = np.asarray(new_values)
    print(
        f"  {name}: old [{np.nanmin(old_values):.6g}, {np.nanmax(old_values):.6g}] "
        f"-> new [{np.nanmin(new_values):.6g}, {np.nanmax(new_values):.6g}]"
    )


# -----------------------------------------------------------------------------
# MAIN REMESHING
# -----------------------------------------------------------------------------

def main():
    if COMM_WORLD.size != 1:
        raise RuntimeError(
            "Run the remeshing script with one MPI process, e.g.\n"
            "python Remesh_existing_mesh.py"
        )

    input_meta = INPUT_RESTART.with_name(INPUT_RESTART.stem + "_meta.npz")
    output_meta = OUTPUT_RESTART.with_name(OUTPUT_RESTART.stem + "_meta.npz")

    if not INPUT_RESTART.is_file():
        raise FileNotFoundError(f"Input restart does not exist:\n{INPUT_RESTART}")

    if not input_meta.is_file():
        raise FileNotFoundError(f"Input metadata does not exist:\n{input_meta}")

    OUTPUT_RESTART.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read old structured/extruded checkpoint.
    # ------------------------------------------------------------------
    print(f"Reading {INPUT_RESTART}")

    with CheckpointFile(str(INPUT_RESTART), "r") as checkpoint:
        try:
            old_mesh = checkpoint.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )
        except Exception:
            old_mesh = checkpoint.load_mesh(reorder=False)

        old_thick = checkpoint.load_function(old_mesh, "thick")
        old_zb = checkpoint.load_function(old_mesh, "zb")
        old_zs = checkpoint.load_function(old_mesh, "zs")
        old_w = checkpoint.load_function(old_mesh, "w")
        old_uvec_out = checkpoint.load_function(old_mesh, "uvec_out")

    # ------------------------------------------------------------------
    # Load the supplied fine unstructured horizontal mesh, then extrude it in
    # reference sigma coordinates 0 <= sigma <= 1.
    # ------------------------------------------------------------------
    base_mesh = load_target_base_mesh()

    new_mesh = ExtrudedMesh(
        base_mesh,
        layers=NZ,
        layer_height=1.0 / NZ,
    )

    # Same element choices as the current MISMIP+ scripts.
    horizontal_element = FiniteElement("CG", "triangle", 1)
    vertical_element = FiniteElement("CG", "interval", 1)
    scalar_element = TensorProductElement(horizontal_element, vertical_element)
    vector_element = VectorElement(scalar_element, dim=2)

    old_V = FunctionSpace(old_mesh, scalar_element)
    old_Vbar = FunctionSpace(
        old_mesh,
        "CG",
        1,
        vfamily="R",
        vdegree=0,
    )

    new_V = FunctionSpace(new_mesh, scalar_element)
    new_VV = FunctionSpace(new_mesh, vector_element)
    new_Vbar = FunctionSpace(
        new_mesh,
        "CG",
        1,
        vfamily="R",
        vdegree=0,
    )
    new_VVbar = VectorFunctionSpace(
        new_mesh,
        "CG",
        1,
        vfamily="R",
        vdegree=0,
        dim=2,
    )
    new_W = new_VV * new_VVbar

    new_thick = Function(new_Vbar, name="thick")
    new_zb = Function(new_Vbar, name="zb")
    new_zs = Function(new_Vbar, name="zs")
    new_w = Function(new_W, name="w")
    new_uvec_out = Function(new_VV, name="uvec_out")

    # ------------------------------------------------------------------
    # 1. Horizontal geometry transfer in (x, y).
    # ------------------------------------------------------------------
    print("Interpolating horizontal geometry fields")
    old_xy = coordinates_2d(old_Vbar)
    new_xy = coordinates_2d(new_Vbar)

    new_thick.dat.data[:] = interpolate_values(
        old_xy,
        old_thick.dat.data_ro,
        new_xy,
        label="thick",
    )

    new_zb.dat.data[:] = interpolate_values(
        old_xy,
        old_zb.dat.data_ro,
        new_xy,
        label="zb",
    )

    # Avoid invalid columns if linear interpolation introduces a tiny local
    # undershoot near the margin.
    new_thick.dat.data[:] = np.maximum(new_thick.dat.data, MINIMUM_THICKNESS)

    # Reconstruct zs exactly so zb + H = zs remains internally consistent.
    new_zs.interpolate(new_zb + new_thick)

    # ------------------------------------------------------------------
    # 2. Velocity transfer in (x, y, sigma).
    # ------------------------------------------------------------------
    print("Interpolating 3-D velocity fields in (x, y, sigma)")
    old_xyz = source_coordinates_3d(old_V, old_zb, old_thick)
    new_xyz = target_coordinates_3d(new_V)

    old_velocity, old_flux = old_w.subfunctions
    new_velocity, new_flux = new_w.subfunctions

    new_uvec_out.dat.data[:, :] = interpolate_values(
        old_xyz,
        old_uvec_out.dat.data_ro,
        new_xyz,
        label="uvec_out",
    )

    new_velocity.dat.data[:, :] = interpolate_values(
        old_xyz,
        old_velocity.dat.data_ro,
        new_xyz,
        label="velocity",
    )

    # q/flux is depth-independent, so transfer it horizontally only.
    new_flux.dat.data[:, :] = interpolate_values(
        old_xy,
        old_flux.dat.data_ro,
        new_xy,
        label="flux",
    )

    # ------------------------------------------------------------------
    # 3. Deform the reference extrusion to the physical ice geometry.
    # ------------------------------------------------------------------
    print("Deforming target mesh to physical ice geometry")
    x, y, sigma = SpatialCoordinate(new_mesh)
    new_mesh.coordinates.interpolate(
        as_vector([x, y, new_zb + sigma * new_thick])
    )

    # Basic transfer diagnostics before writing.
    print("Interpolation ranges")
    report_range("thick", old_thick.dat.data_ro, new_thick.dat.data_ro)
    report_range("zb", old_zb.dat.data_ro, new_zb.dat.data_ro)
    report_range("zs", old_zs.dat.data_ro, new_zs.dat.data_ro)
    report_range("uvec_out", old_uvec_out.dat.data_ro, new_uvec_out.dat.data_ro)

    # ------------------------------------------------------------------
    # 4. Write a restart checkpoint on the new unstructured/extruded mesh.
    # ------------------------------------------------------------------
    print(f"Writing {OUTPUT_RESTART}")

    with CheckpointFile(str(OUTPUT_RESTART), "w") as checkpoint:
        checkpoint.save_mesh(new_mesh)
        checkpoint.save_function(new_thick, name="thick")
        checkpoint.save_function(new_zb, name="zb")
        checkpoint.save_function(new_zs, name="zs")
        checkpoint.save_function(new_w, name="w")
        checkpoint.save_function(new_uvec_out, name="uvec_out")

    # Preserve the restart metadata.
    with np.load(input_meta) as metadata:
        np.savez(
            output_meta,
            t=float(metadata["t"]),
            step=int(metadata["step"]),
            dt=float(metadata["dt"]),
            theta_out=float(metadata["theta_out"]),
        )

    print("Finished")
    print(f"Target horizontal mesh (loaded directly): {TARGET_BASE_MESH}")
    print(f"Vertical layers: {NZ}")
    print(f"Output restart: {OUTPUT_RESTART}")


if __name__ == "__main__":
    main()
