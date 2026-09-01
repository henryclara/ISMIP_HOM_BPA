from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from firedrake import *
from config import *
from geometry import mismip_bed

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

INPUT_RESTART = Path(
    "Simulations/Ice0_theta1_dt0.5_res2000_2000_nz10_GJP_2D_thickness/"
    "restart_t8000.h5"
)

OUTPUT_RESTART = Path(
    f"Simulations/remesh_mesh_res_{str(int(coarse_res))}_"
    f"{str(int(refined_res))}_unstructured/restart_t8000.h5"
)

# Existing horizontal unstructured Gmsh mesh.
TARGET_BASE_MESH = Path(
    f"Meshes/mesh_res_{str(int(coarse_res))}_"
    f"{str(int(refined_res))}_unstructured.msh"
)

# Number of vertical layers in the target extrusion.
# The source restart used here is also nz10.  The code verifies that the
# recovered source sigma coordinates are compatible with these levels.
NZ = 10

MINIMUM_THICKNESS = 10.0

# Tolerance used when deciding whether sigma is on k/NZ.
SIGMA_TOL = 1.0e-7


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


def align_base_mesh_to_source(base_mesh, old_mesh):
    """
    Translate the horizontal Gmsh mesh so that it occupies the same x-y
    coordinate range as the source restart.

    The Gmsh mesh is generated from (0, 0).  For MISMIP+ half-domain runs the
    source restart may instead occupy e.g. y=[40000, 80000].  Rather than
    hard-coding that offset, use the source restart as the authoritative
    coordinate system and verify that the horizontal domain widths agree.
    """
    base_coords = base_mesh.coordinates.dat.data
    old_coords = old_mesh.coordinates.dat.data_ro

    base_xmin = float(np.min(base_coords[:, 0]))
    base_xmax = float(np.max(base_coords[:, 0]))
    base_ymin = float(np.min(base_coords[:, 1]))
    base_ymax = float(np.max(base_coords[:, 1]))

    old_xmin = float(np.min(old_coords[:, 0]))
    old_xmax = float(np.max(old_coords[:, 0]))
    old_ymin = float(np.min(old_coords[:, 1]))
    old_ymax = float(np.max(old_coords[:, 1]))

    print(
        "Target base coordinate range before alignment:",
        f"x=[{base_xmin}, {base_xmax}]",
        f"y=[{base_ymin}, {base_ymax}]",
    )
    print(
        "Source restart horizontal coordinate range:",
        f"x=[{old_xmin}, {old_xmax}]",
        f"y=[{old_ymin}, {old_ymax}]",
    )

    base_dx = base_xmax - base_xmin
    base_dy = base_ymax - base_ymin
    old_dx = old_xmax - old_xmin
    old_dy = old_ymax - old_ymin

    # Allow small floating-point differences, but reject genuinely different
    # domain sizes.  A translation can fix an origin mismatch; it cannot fix a
    # different-sized domain.
    scale = max(abs(old_dx), abs(old_dy), abs(base_dx), abs(base_dy), 1.0)
    tol = 1.0e-9 * scale

    if abs(base_dx - old_dx) > tol or abs(base_dy - old_dy) > tol:
        raise RuntimeError(
            "The source restart and target Gmsh mesh have different horizontal "
            "domain sizes.\n"
            f"  source span: dx={old_dx}, dy={old_dy}\n"
            f"  target span: dx={base_dx}, dy={base_dy}\n"
            "A coordinate translation cannot correct this. Check Lx/Ly and "
            "the mesh-generation configuration."
        )

    x_shift = old_xmin - base_xmin
    y_shift = old_ymin - base_ymin

    print(f"Applying target-mesh translation: dx={x_shift}, dy={y_shift}")

    base_coords[:, 0] += x_shift
    base_coords[:, 1] += y_shift

    new_xmin = float(np.min(base_coords[:, 0]))
    new_xmax = float(np.max(base_coords[:, 0]))
    new_ymin = float(np.min(base_coords[:, 1]))
    new_ymax = float(np.max(base_coords[:, 1]))

    print(
        "Target base coordinate range after alignment:",
        f"x=[{new_xmin}, {new_xmax}]",
        f"y=[{new_ymin}, {new_ymax}]",
    )


# -----------------------------------------------------------------------------
# INTERPOLATION HELPERS
# -----------------------------------------------------------------------------

def _scaled_points(source_points, target_points):
    """
    Scale interpolation coordinates to O(1) before SciPy calls Qhull.

    For horizontal interpolation, x and y can be O(10^5 m).  Scaling improves
    numerical conditioning without changing the interpolation geometry.
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

    This helper is now used only for 2-D horizontal interpolation.  Linear
    interpolation is used inside the source convex hull.  A very small number
    of target points outside the hull may be filled with nearest-neighbour
    values; a significant fraction is treated as a coordinate-system error.
    """
    source_points = np.asarray(source_points, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError(f"{label}: interpolation points must be 2-D arrays")

    if source_points.shape[1] != target_points.shape[1]:
        raise ValueError(
            f"{label}: source/target coordinate dimensions differ: "
            f"{source_points.shape[1]} vs {target_points.shape[1]}"
        )

    if source_points.shape[1] != 2:
        raise ValueError(
            f"{label}: interpolate_values is intended for horizontal (x, y) "
            f"interpolation only; received dimension {source_points.shape[1]}"
        )

    if len(source_points) != len(source_values):
        raise ValueError(
            f"{label}: source point/value counts differ: "
            f"{len(source_points)} vs {len(source_values)}"
        )

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
        fraction = nmissing / len(target_points)

        print(
            f"  {label}: {nmissing}/{len(target_points)} "
            f"({100.0 * fraction:.3f}%) target points "
            "outside linear interpolation hull"
        )

        # A tiny number can occur from boundary roundoff.  A significant
        # fraction means the meshes do not overlap correctly.
        if fraction > 0.01:
            raise RuntimeError(
                f"{label}: too many target points are outside the source "
                "interpolation hull. Check source/target coordinate systems."
            )

        nearest = NearestNDInterpolator(
            source_scaled,
            source_values,
        )
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
    Return source coordinates in (x, y, sigma), where

        sigma = (z - zb) / H.

    These coordinates are used only to identify horizontal sigma layers.  We
    deliberately do NOT pass the full (x, y, sigma) cloud to Qhull.
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


def _sigma_layer_numbers(sigma, nz, label):
    """
    Map sigma values to integer extrusion levels k=0,...,nz and verify that
    they really lie on k/nz to within SIGMA_TOL.
    """
    sigma = np.asarray(sigma, dtype=float)
    layer = np.rint(sigma * nz).astype(int)

    if np.any(layer < 0) or np.any(layer > nz):
        raise RuntimeError(
            f"{label}: sigma values fall outside the expected [0, 1] range: "
            f"min={np.min(sigma)}, max={np.max(sigma)}"
        )

    expected_sigma = layer / float(nz)
    error = np.abs(sigma - expected_sigma)
    max_error = float(np.max(error)) if len(error) else 0.0

    print(f"  {label}: max sigma-layer error = {max_error:.3e}")

    if max_error > SIGMA_TOL:
        bad = int(np.argmax(error))
        raise RuntimeError(
            f"{label}: source/target DOFs are not on the expected sigma levels "
            f"k/{nz}. Largest mismatch is {max_error:.3e} at sigma={sigma[bad]}. "
            "If the source restart uses a different number of vertical layers, "
            "set NZ accordingly or use vertical interpolation between layers."
        )

    return layer


def interpolate_sigma_layers(
    source_xyz,
    source_values,
    target_xyz,
    nz,
    label="field",
):
    """
    Transfer an extruded scalar/vector field layer-by-layer.

    Each sigma level is interpolated independently in (x, y).  This avoids the
    global 3-D Delaunay triangulation of a tensor-product point cloud that was
    causing the Qhull wide-merge/degeneracy failure.
    """
    source_xyz = np.asarray(source_xyz, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float)

    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(f"{label}: source_xyz must have shape (N, 3)")
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError(f"{label}: target_xyz must have shape (N, 3)")
    if len(source_xyz) != len(source_values):
        raise ValueError(
            f"{label}: source point/value counts differ: "
            f"{len(source_xyz)} vs {len(source_values)}"
        )

    source_layer = _sigma_layer_numbers(
        source_xyz[:, 2], nz, f"{label} source"
    )
    target_layer = _sigma_layer_numbers(
        target_xyz[:, 2], nz, f"{label} target"
    )

    if source_values.ndim == 1:
        result = np.empty(len(target_xyz), dtype=float)
    else:
        result = np.empty(
            (len(target_xyz), source_values.shape[1]),
            dtype=float,
        )

    for k in range(nz + 1):
        source_mask = source_layer == k
        target_mask = target_layer == k

        ns = int(np.count_nonzero(source_mask))
        nt = int(np.count_nonzero(target_mask))

        print(
            f"  {label}: sigma={k / float(nz):.3f}: "
            f"{ns} source DOFs -> {nt} target DOFs"
        )

        if nt == 0:
            continue

        if ns == 0:
            raise RuntimeError(
                f"{label}: no source DOFs found on sigma layer {k}/{nz}"
            )

        source_xy = source_xyz[source_mask, :2]
        target_xy = target_xyz[target_mask, :2]

        result[target_mask] = interpolate_values(
            source_xy,
            source_values[source_mask],
            target_xy,
            label=f"{label}, sigma={k / float(nz):.3f}",
        )

    return result


def report_range(name, old_values, new_values):
    old_values = np.asarray(old_values)
    new_values = np.asarray(new_values)
    print(
        f"  {name}: old [{np.nanmin(old_values):.6g}, "
        f"{np.nanmax(old_values):.6g}] "
        f"-> new [{np.nanmin(new_values):.6g}, "
        f"{np.nanmax(new_values):.6g}]"
    )


def report_xyz_ranges(name, xyz):
    xyz = np.asarray(xyz)
    print(
        f"{name}: "
        f"x=[{xyz[:, 0].min()}, {xyz[:, 0].max()}], "
        f"y=[{xyz[:, 1].min()}, {xyz[:, 1].max()}], "
        f"sigma=[{xyz[:, 2].min()}, {xyz[:, 2].max()}], "
        f"shape={xyz.shape}"
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
    # Load the supplied horizontal mesh and align its coordinate origin with
    # the source restart.  This handles e.g. Gmsh y=[0,40000] versus source
    # MISMIP+ y=[40000,80000] without hard-coding the offset.
    # ------------------------------------------------------------------
    base_mesh = load_target_base_mesh()
    align_base_mesh_to_source(base_mesh, old_mesh)

    print(
        "BASE MESH BOUNDARY MARKERS:",
        base_mesh.exterior_facets.unique_markers,
    )

    new_mesh = ExtrudedMesh(
        base_mesh,
        layers=NZ,
        layer_height=1.0 / NZ,
    )

    print(
        "EXTRUDED SIDE BOUNDARY MARKERS:",
        new_mesh.exterior_facets.unique_markers,
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
    new_bed = Function(new_Vbar, name="bed")
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

    print(
        "OLD coordinate ranges:",
        f"x=[{old_xy[:, 0].min()}, {old_xy[:, 0].max()}]",
        f"y=[{old_xy[:, 1].min()}, {old_xy[:, 1].max()}]",
    )

    print(
        "NEW coordinate ranges:",
        f"x=[{new_xy[:, 0].min()}, {new_xy[:, 0].max()}]",
        f"y=[{new_xy[:, 1].min()}, {new_xy[:, 1].max()}]",
    )

    # Interpolate thickness horizontally.
    new_thick.dat.data[:] = interpolate_values(
        old_xy,
        old_thick.dat.data_ro,
        new_xy,
        label="thick",
    )

    new_thick.dat.data[:] = np.maximum(
        new_thick.dat.data,
        MINIMUM_THICKNESS,
    )

    # ------------------------------------------------------------------
    # Evaluate the analytical bedrock on the NEW horizontal mesh.
    # ------------------------------------------------------------------
    x_new, y_new, sigma = SpatialCoordinate(new_mesh)

    new_bed.interpolate(
        mismip_bed(x_new, y_new)
    )

    print(
        "New bed range:",
        new_bed.dat.data_ro.min(),
        new_bed.dat.data_ro.max(),
    )

    # ------------------------------------------------------------------
    # Reconstruct lower and upper ice surfaces.
    # ------------------------------------------------------------------
    zb_float = -(rhoi / rhow) * new_thick

    # Grounded ice: zb = bed
    # Floating ice: zb = hydrostatic flotation depth
    new_zb.interpolate(
        max_value(new_bed, zb_float)
    )

    new_zs.interpolate(
        new_zb + new_thick
    )

    print(
        "New zb range:",
        new_zb.dat.data_ro.min(),
        new_zb.dat.data_ro.max(),
    )

    print(
        "New zs range:",
        new_zs.dat.data_ro.min(),
        new_zs.dat.data_ro.max(),
    )

    # ------------------------------------------------------------------
    # 2. Velocity transfer layer-by-layer in sigma.
    #
    # IMPORTANT:
    # Do not use a global LinearNDInterpolator in (x, y, sigma).  The old
    # extruded tensor-product mesh contains huge numbers of coplanar/cospherical
    # point configurations, which caused the Qhull "wide merge" failure.
    # Instead, interpolate each sigma layer independently in 2-D (x, y).
    # ------------------------------------------------------------------
    print("Interpolating velocity fields layer-by-layer in sigma")

    old_xyz = source_coordinates_3d(old_V, old_zb, old_thick)
    new_xyz = target_coordinates_3d(new_V)

    report_xyz_ranges("OLD (x,y,sigma)", old_xyz)
    report_xyz_ranges("NEW (x,y,sigma)", new_xyz)

    print(
        "OLD sigma levels:",
        np.unique(np.round(old_xyz[:, 2], 10)),
    )
    print(
        "NEW sigma levels:",
        np.unique(np.round(new_xyz[:, 2], 10)),
    )

    old_velocity, old_flux = old_w.subfunctions
    new_velocity, new_flux = new_w.subfunctions

    print("old_uvec_out shape:", old_uvec_out.dat.data_ro.shape)
    print("old_velocity shape:", old_velocity.dat.data_ro.shape)
    print("old_flux shape:", old_flux.dat.data_ro.shape)

    new_uvec_out.dat.data[:, :] = interpolate_sigma_layers(
        old_xyz,
        old_uvec_out.dat.data_ro,
        new_xyz,
        NZ,
        label="uvec_out",
    )

    new_velocity.dat.data[:, :] = interpolate_sigma_layers(
        old_xyz,
        old_velocity.dat.data_ro,
        new_xyz,
        NZ,
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
    report_range(
        "uvec_out",
        old_uvec_out.dat.data_ro,
        new_uvec_out.dat.data_ro,
    )

    # ------------------------------------------------------------------
    # 4. Write a restart checkpoint on the new unstructured/extruded mesh.
    # ------------------------------------------------------------------
    print(f"Writing {OUTPUT_RESTART}")

    with CheckpointFile(str(OUTPUT_RESTART), "w") as checkpoint:
        checkpoint.save_mesh(new_mesh)
        checkpoint.save_function(new_thick, name="thick")
        checkpoint.save_function(new_bed, name="bed")
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
    print(f"Target horizontal mesh: {TARGET_BASE_MESH}")
    print(f"Vertical layers: {NZ}")
    print(f"Output restart: {OUTPUT_RESTART}")


if __name__ == "__main__":
    main()
