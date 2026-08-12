from pathlib import Path
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
    "Simulations/remesh_refined_MISMIP_output_theta1_dt0.5_GL_predFalse_res_100_nz_10/"
    "restart_t10000.h5"
)

# MISMIP+ domain dimensions.
LX = 640_000.0
LY_FULL = 80_000.0

# Region to refine in x.
REFINED_X_MIN = 400_000.0
REFINED_X_MAX = 500_000.0

# Requested horizontal cell sizes in metres.
#
# Example:
#     8_000.0 = 8 km
#     4_000.0 = 4 km
#     2_000.0 = 2 km
#
# The script calculates the number of cells in each region automatically.
DX_LEFT = 2000.0
DX_REFINED = 100.0
DX_RIGHT = 2000.0

# Requested y-direction cell size in metres.
DY = 2000.0

# Number of vertical layers.
NZ = 10

MINIMUM_THICKNESS = 10.0


# -----------------------------------------------------------------------------
# MESH HELPERS
# -----------------------------------------------------------------------------

def cell_count(length, requested_resolution, region_name):
    """
    Return an integer number of cells for a requested resolution.

    If the region length is not exactly divisible by the requested resolution,
    the nearest integer number of cells is used. The resulting actual
    resolution is printed later.
    """

    if requested_resolution <= 0.0:
        raise ValueError(
            f"{region_name} resolution must be positive, "
            f"but received {requested_resolution}."
        )

    cells = int(round(length / requested_resolution))

    if cells < 1:
        raise ValueError(
            f"{region_name} is too short for the requested resolution."
        )

    return cells


def create_locally_refined_base_mesh():
    """
    Create a structured triangular mesh with piecewise-uniform x resolution.

    The mesh has:
      - DX_LEFT before REFINED_X_MIN,
      - DX_REFINED between REFINED_X_MIN and REFINED_X_MAX,
      - DX_RIGHT after REFINED_X_MAX,
      - DY in the y direction.
    """

    if not (
        0.0 < REFINED_X_MIN < REFINED_X_MAX < LX
    ):
        raise ValueError(
            "Require 0 < REFINED_X_MIN < REFINED_X_MAX < LX."
        )

    ly = LY_FULL / 2.0

    left_length = REFINED_X_MIN
    refined_length = REFINED_X_MAX - REFINED_X_MIN
    right_length = LX - REFINED_X_MAX

    nx_left = cell_count(
        left_length,
        DX_LEFT,
        "Left x-region",
    )

    nx_refined = cell_count(
        refined_length,
        DX_REFINED,
        "Refined x-region",
    )

    nx_right = cell_count(
        right_length,
        DX_RIGHT,
        "Right x-region",
    )

    ny = cell_count(
        ly,
        DY,
        "Y-region",
    )

    nx_total = nx_left + nx_refined + nx_right

    # Exact physical x-node coordinates for the three regions.
    x_left = np.linspace(
        0.0,
        REFINED_X_MIN,
        nx_left + 1,
    )

    x_refined = np.linspace(
        REFINED_X_MIN,
        REFINED_X_MAX,
        nx_refined + 1,
    )[1:]

    x_right = np.linspace(
        REFINED_X_MAX,
        LX,
        nx_right + 1,
    )[1:]

    x_nodes = np.concatenate(
        (
            x_left,
            x_refined,
            x_right,
        )
    )

    # Start from a logically uniform structured mesh.
    base_mesh = RectangleMesh(
        nx_total,
        ny,
        LX,
        ly,
    )

    coordinates = base_mesh.coordinates.dat.data
    logical_x = coordinates[:, 0].copy()

    # Map each logical x-node to the desired physical x-node.
    logical_nodes = np.linspace(
        0.0,
        LX,
        nx_total + 1,
    )

    coordinates[:, 0] = np.interp(
        logical_x,
        logical_nodes,
        x_nodes,
    )

    # Shift the half-domain to y = 40–80 km, as in the original script.
    coordinates[:, 1] += ly

    mesh_information = {
        "nx_left": nx_left,
        "nx_refined": nx_refined,
        "nx_right": nx_right,
        "nx_total": nx_total,
        "ny": ny,
        "dx_left_actual": left_length / nx_left,
        "dx_refined_actual": refined_length / nx_refined,
        "dx_right_actual": right_length / nx_right,
        "dy_actual": ly / ny,
    }

    return base_mesh, mesh_information


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
        missing = ~np.all(
            np.isfinite(result),
            axis=1,
        )

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

    x_field = Function(
        function_space
    ).interpolate(x)

    y_field = Function(
        function_space
    ).interpolate(y)

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

    x_field = Function(
        function_space
    ).interpolate(x)

    y_field = Function(
        function_space
    ).interpolate(y)

    sigma_field = Function(
        function_space
    ).interpolate(
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

    x_field = Function(
        function_space
    ).interpolate(x)

    y_field = Function(
        function_space
    ).interpolate(y)

    sigma_field = Function(
        function_space
    ).interpolate(sigma)

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
            "Run this script with one process:\n"
            "python Remesh.py"
        )

    input_meta = INPUT_RESTART.replace(
        ".h5",
        "_meta.npz",
    )

    output_meta = OUTPUT_RESTART.replace(
        ".h5",
        "_meta.npz",
    )

    if not os.path.isfile(INPUT_RESTART):
        raise FileNotFoundError(
            f"Input restart does not exist:\n{INPUT_RESTART}"
        )

    if not os.path.isfile(input_meta):
        raise FileNotFoundError(
            f"Input metadata does not exist:\n{input_meta}"
        )

    print(f"Reading {INPUT_RESTART}")

    with CheckpointFile(
        INPUT_RESTART,
        "r",
    ) as checkpoint:

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

    # Create the new, locally refined reference mesh.
    base_mesh, mesh_information = create_locally_refined_base_mesh()

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

    output_directory = os.path.dirname(OUTPUT_RESTART)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    print(f"Writing {OUTPUT_RESTART}")

    with CheckpointFile(
        OUTPUT_RESTART,
        "w",
    ) as checkpoint:

        checkpoint.save_mesh(new_mesh)
        checkpoint.save_function(
            new_thick,
            name="thick",
        )
        checkpoint.save_function(
            new_zb,
            name="zb",
        )
        checkpoint.save_function(
            new_zs,
            name="zs",
        )
        checkpoint.save_function(
            new_w,
            name="w",
        )
        checkpoint.save_function(
            new_uvec_out,
            name="uvec_out",
        )

    # Copy restart metadata unchanged.
    with np.load(input_meta) as metadata:
        np.savez(
            output_meta,
            t=float(metadata["t"]),
            step=int(metadata["step"]),
            dt=float(metadata["dt"]),
            theta_out=float(metadata["theta_out"]),
        )

    print("Finished")
    print(
        "Refined x-region: "
        f"{REFINED_X_MIN / 1000.0:g}–"
        f"{REFINED_X_MAX / 1000.0:g} km"
    )
    print(
        "Cell counts: "
        f"NX_LEFT={mesh_information['nx_left']}, "
        f"NX_REFINED={mesh_information['nx_refined']}, "
        f"NX_RIGHT={mesh_information['nx_right']}, "
        f"NX_TOTAL={mesh_information['nx_total']}, "
        f"NY={mesh_information['ny']}, "
        f"NZ={NZ}"
    )
    print(
        "Actual resolutions: "
        f"left={mesh_information['dx_left_actual'] / 1000.0:.3f} km, "
        f"refined={mesh_information['dx_refined_actual'] / 1000.0:.3f} km, "
        f"right={mesh_information['dx_right_actual'] / 1000.0:.3f} km, "
        f"dy={mesh_information['dy_actual'] / 1000.0:.3f} km"
    )


if __name__ == "__main__":
    main()

