import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from firedrake import *


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

INPUT_RESTART = (
    "Simulations/MISMIP_refined_output_theta1_dt0.5_GL_predFalse_nx_320_nz_10/"
    "restart_t10000.h5"
)

OUTPUT_RESTART = (
    "Simulations/remesh_refined_MISMIP_output_theta1_dt0.5_GL_predFalse_nx_640_nz_10/"
    "restart_t10000.h5"
)

# Number of cells in each direction on the new mesh.
NX = int(640)
NY = int(40)
NZ = int(10)

# MISMIP+ domain dimensions.
LX = 640_000.0
LY_FULL = 80_000.0

MINIMUM_THICKNESS = 10.0


# -----------------------------------------------------------------------------
# INTERPOLATION HELPERS
# -----------------------------------------------------------------------------

def interpolate_values(source_points, source_values, target_points):
    """Linear interpolation with nearest-neighbour filling."""

    linear = LinearNDInterpolator(
        source_points,
        source_values,
        fill_value=np.nan,
    )

    result = np.asarray(
        linear(target_points),
        dtype=float,
    )

    if result.ndim == 1:
        missing = ~np.isfinite(result)
    else:
        missing = ~np.all(np.isfinite(result), axis=1)

    if np.any(missing):
        nearest = NearestNDInterpolator(
            source_points,
            source_values,
        )

        result[missing] = nearest(
            target_points[missing]
        )

    return result


def coordinates_2d(function_space):
    """Return x-y coordinates at the degrees of freedom."""

    mesh = function_space.mesh()
    x, y, _ = SpatialCoordinate(mesh)

    x_field = Function(function_space).interpolate(x)
    y_field = Function(function_space).interpolate(y)

    return np.column_stack(
        (
            x_field.dat.data_ro,
            y_field.dat.data_ro,
        )
    )


def source_coordinates_3d(function_space, zb, thick):
    """Return source x-y-sigma coordinates."""

    mesh = function_space.mesh()
    x, y, z = SpatialCoordinate(mesh)

    x_field = Function(function_space).interpolate(x)
    y_field = Function(function_space).interpolate(y)
    sigma_field = Function(function_space).interpolate(
        (z - zb) / thick
    )

    return np.column_stack(
        (
            x_field.dat.data_ro,
            y_field.dat.data_ro,
            sigma_field.dat.data_ro,
        )
    )


def target_coordinates_3d(function_space):
    """Return target x-y-sigma coordinates before mesh deformation."""

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


# -----------------------------------------------------------------------------
# MAIN REMESHING
# -----------------------------------------------------------------------------

def main():

    if COMM_WORLD.size != 1:
        raise RuntimeError(
            "Run this script with one process: "
            "python remesh_mismip_restart_simple.py"
        )

    input_meta = INPUT_RESTART.replace(
        ".h5",
        "_meta.npz",
    )

    output_meta = OUTPUT_RESTART.replace(
        ".h5",
        "_meta.npz",
    )

    print(f"Reading {INPUT_RESTART}")

    with CheckpointFile(INPUT_RESTART, "r") as checkpoint:
        try:
            old_mesh = checkpoint.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )
        except Exception:
            old_mesh = checkpoint.load_mesh(
                reorder=False,
            )

        old_thick = checkpoint.load_function(
            old_mesh,
            "thick",
        )
        old_zb = checkpoint.load_function(
            old_mesh,
            "zb",
        )
        old_zs = checkpoint.load_function(
            old_mesh,
            "zs",
        )
        old_w = checkpoint.load_function(
            old_mesh,
            "w",
        )
        old_uvec_out = checkpoint.load_function(
            old_mesh,
            "uvec_out",
        )

    # Create the new reference mesh.
    ly = LY_FULL / 2.0

    base_mesh = RectangleMesh(
        NX,
        NY,
        LX,
        ly,
    )

    base_mesh.coordinates.dat.data[:, 1] += ly

    new_mesh = ExtrudedMesh(
        base_mesh,
        layers=NZ,
        layer_height=1.0 / NZ,
    )

    # Spaces used by the MISMIP+ scripts.
    horizontal_element = FiniteElement(
        "CG",
        "triangle",
        1,
    )

    vertical_element = FiniteElement(
        "CG",
        "interval",
        1,
    )

    scalar_element = TensorProductElement(
        horizontal_element,
        vertical_element,
    )

    vector_element = VectorElement(
        scalar_element,
        dim=2,
    )

    old_V = FunctionSpace(
        old_mesh,
        scalar_element,
    )

    old_Vbar = FunctionSpace(
        old_mesh,
        "CG",
        1,
        vfamily="R",
        vdegree=0,
    )

    new_V = FunctionSpace(
        new_mesh,
        scalar_element,
    )

    new_VV = FunctionSpace(
        new_mesh,
        vector_element,
    )

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

    new_thick = Function(
        new_Vbar,
        name="thick",
    )

    new_zb = Function(
        new_Vbar,
        name="zb",
    )

    new_zs = Function(
        new_Vbar,
        name="zs",
    )

    new_w = Function(
        new_W,
        name="w",
    )

    new_uvec_out = Function(
        new_VV,
        name="uvec_out",
    )

    # Interpolate horizontal geometry fields.
    old_xy = coordinates_2d(old_Vbar)
    new_xy = coordinates_2d(new_Vbar)

    new_thick.dat.data[:] = interpolate_values(
        old_xy,
        old_thick.dat.data_ro,
        new_xy,
    )

    new_zb.dat.data[:] = interpolate_values(
        old_xy,
        old_zb.dat.data_ro,
        new_xy,
    )

    new_thick.dat.data[:] = np.maximum(
        new_thick.dat.data,
        MINIMUM_THICKNESS,
    )

    new_zs.interpolate(
        new_zb + new_thick
    )

    # Interpolate velocity and mixed solution fields in x-y-sigma space.
    old_xyz = source_coordinates_3d(
        old_V,
        old_zb,
        old_thick,
    )

    new_xyz = target_coordinates_3d(
        new_V,
    )

    old_velocity, old_flux = old_w.subfunctions
    new_velocity, new_flux = new_w.subfunctions

    new_uvec_out.dat.data[:, :] = interpolate_values(
        old_xyz,
        old_uvec_out.dat.data_ro,
        new_xyz,
    )

    new_velocity.dat.data[:, :] = interpolate_values(
        old_xyz,
        old_velocity.dat.data_ro,
        new_xyz,
    )

    new_flux.dat.data[:, :] = interpolate_values(
        old_xy,
        old_flux.dat.data_ro,
        new_xy,
    )

    # Convert the new reference extrusion into the physical ice geometry.
    x, y, sigma = SpatialCoordinate(new_mesh)

    new_mesh.coordinates.interpolate(
        as_vector(
            [
                x,
                y,
                new_zb + sigma * new_thick,
            ]
        )
    )

    os.makedirs(
        os.path.dirname(OUTPUT_RESTART),
        exist_ok=True,
    )

    print(f"Writing {OUTPUT_RESTART}")

    with CheckpointFile(OUTPUT_RESTART, "w") as checkpoint:
        checkpoint.save_mesh(new_mesh)
        checkpoint.save_function(new_thick, name="thick")
        checkpoint.save_function(new_zb, name="zb")
        checkpoint.save_function(new_zs, name="zs")
        checkpoint.save_function(new_w, name="w")
        checkpoint.save_function(new_uvec_out, name="uvec_out")

    # Copy restart metadata unchanged.
    metadata = np.load(input_meta)

    np.savez(
        output_meta,
        t=float(metadata["t"]),
        step=int(metadata["step"]),
        dt=float(metadata["dt"]),
        theta_out=float(metadata["theta_out"]),
    )

    print("Finished")
    print(f"New resolution: NX={NX}, NY={NY}, NZ={NZ}")

if __name__ == "__main__":
    main()