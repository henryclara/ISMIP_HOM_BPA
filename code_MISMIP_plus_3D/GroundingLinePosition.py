import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 10100

theta = 1
resolutions = [250, 500]
zeta_pred = False

# SEP3 simulations to compare
dts = [1, 2, 5, 10, 20]

# Reference timestep, if you want it visually distinguished
ref_dt = 1

output_directory = "Simulations"

figure_dpi = 700

# Hydrostatic flotation criterion
rhoi = 917.0
rhow = 1028.0

# Same tolerance as your existing GL plotting script
grounding_line_tolerance = 0.1

output_filename = (
    f"{output_directory}/"
    f"MISMIP_Ice1r_SEP3_grounding_line_vs_dt.png"
)


# ---------------------------------------------------------------------
# Load restart state
# ---------------------------------------------------------------------

def load_state(dt, resolution):

    filename = (
        f"{output_directory}/"
        f"MISMIP_Ice1r_theta{theta:g}_dt{dt:g}_"
        f"GL_pred{zeta_pred}_res_{resolution}_GL_SEP3/"
        f"restart_t{T:g}.h5"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Restart file not found:\n{filename}"
        )

    print(
        f"Loading dt={dt:g}:\n"
        f"  {filename}"
    )

    with CheckpointFile(filename, "r") as afile:

        try:

            mesh = afile.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )

        except Exception:

            mesh_names = list(
                afile._get_mesh_name_topology_name_map().keys()
            )

            if not mesh_names:
                raise RuntimeError(
                    f"No mesh found in {filename}"
                )

            extruded = [
                name
                for name in mesh_names
                if "extruded" in name.lower()
            ]

            if extruded:

                mesh_name = extruded[0]

            elif len(mesh_names) == 1:

                mesh_name = mesh_names[0]

            else:

                raise RuntimeError(
                    f"Several meshes found in {filename}: "
                    f"{mesh_names}"
                )

            mesh = afile.load_mesh(
                name=mesh_name,
                reorder=False,
            )

        H = afile.load_function(
            mesh,
            "thick",
        )

        zb = afile.load_function(
            mesh,
            "zb",
        )

    return mesh, H, zb


# ---------------------------------------------------------------------
# Combine repeated horizontal coordinates
# ---------------------------------------------------------------------

def combine_duplicate_xy(
    x_values,
    y_values,
    field_values,
    decimals=12,
):

    x_values = np.asarray(
        x_values,
        dtype=float,
    ).reshape(-1)

    y_values = np.asarray(
        y_values,
        dtype=float,
    ).reshape(-1)

    field_values = np.asarray(
        field_values,
        dtype=float,
    ).reshape(-1)

    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.isfinite(field_values)
    )

    x_values = x_values[finite]
    y_values = y_values[finite]
    field_values = field_values[finite]

    xy_values = np.column_stack(
        (
            x_values,
            y_values,
        )
    )

    xy_rounded = np.round(
        xy_values,
        decimals=decimals,
    )

    xy_unique, inverse = np.unique(
        xy_rounded,
        axis=0,
        return_inverse=True,
    )

    field_sum = np.zeros(
        xy_unique.shape[0],
        dtype=float,
    )

    counts = np.zeros(
        xy_unique.shape[0],
        dtype=int,
    )

    np.add.at(
        field_sum,
        inverse,
        field_values,
    )

    np.add.at(
        counts,
        inverse,
        1,
    )

    field_unique = (
        field_sum / counts
    )

    order = np.lexsort(
        (
            xy_unique[:, 1],
            xy_unique[:, 0],
        )
    )

    return (
        xy_unique[order, 0],
        xy_unique[order, 1],
        field_unique[order],
    )


# ---------------------------------------------------------------------
# Extract a horizontal scalar field
# ---------------------------------------------------------------------

def horizontal_field(field):

    V = field.function_space()
    mesh = field.ufl_domain()

    coordinates = SpatialCoordinate(
        mesh
    )

    x_field = Function(
        V,
        name="x_coordinate",
    )

    y_field = Function(
        V,
        name="y_coordinate",
    )

    x_field.interpolate(
        coordinates[0]
    )

    y_field.interpolate(
        coordinates[1]
    )

    x_values = np.asarray(
        x_field.dat.data_ro,
        dtype=float,
    )

    y_values = np.asarray(
        y_field.dat.data_ro,
        dtype=float,
    )

    values = np.asarray(
        field.dat.data_ro,
        dtype=float,
    )

    x_unique, y_unique, values_unique = (
        combine_duplicate_xy(
            x_values,
            y_values,
            values,
        )
    )

    # Return horizontal coordinates in km
    return (
        x_unique / 1000.0,
        y_unique / 1000.0,
        values_unique,
    )


# ---------------------------------------------------------------------
# Extract grounding-line curve
# ---------------------------------------------------------------------

def grounding_line_curve(H, zb):

    """
    Extract x_GL(y) from

        D = zb + (rho_i / rho_w) H

    using the grounded-to-floating transition on each y-row.
    """

    V = H.function_space()

    flotation_difference = Function(
        V,
        name="hydrostatic_flotation_difference",
    )

    flotation_difference.interpolate(
        zb
        + Constant(rhoi / rhow) * H
    )

    x, y, difference_values = horizontal_field(
        flotation_difference
    )

    # Group nodes by y coordinate
    y_rounded = np.round(
        y,
        decimals=10,
    )

    unique_y = np.unique(
        y_rounded
    )

    x_gl = []
    y_gl = []

    for y_value in unique_y:

        row = np.where(
            y_rounded == y_value
        )[0]

        if row.size < 2:
            continue

        # Sort along flow direction
        row_order = np.argsort(
            x[row]
        )

        row = row[row_order]

        x_row = x[row]

        difference_row = (
            difference_values[row]
        )

        grounded_row = (
            difference_row
            > grounding_line_tolerance
        )

        # Find changes between grounded/floating states
        state_changes = np.where(
            grounded_row[:-1]
            != grounded_row[1:]
        )[0]

        if state_changes.size == 0:
            continue

        # Prefer grounded -> floating transition
        grounded_to_floating = state_changes[
            grounded_row[state_changes]
            & ~grounded_row[state_changes + 1]
        ]

        if grounded_to_floating.size > 0:

            # Use downstream-most transition
            transition = (
                grounded_to_floating[-1]
            )

        else:

            transition = (
                state_changes[-1]
            )

        x0 = x_row[transition]
        x1 = x_row[transition + 1]

        d0 = difference_row[transition]
        d1 = difference_row[transition + 1]

        # Interpolate the GL location
        if np.isclose(
            d0,
            d1,
        ):

            x_crossing = (
                0.5 * (x0 + x1)
            )

        else:

            fraction = (
                grounding_line_tolerance - d0
            ) / (
                d1 - d0
            )

            fraction = np.clip(
                fraction,
                0.0,
                1.0,
            )

            x_crossing = (
                x0
                + fraction
                * (x1 - x0)
            )

        x_gl.append(
            x_crossing
        )

        y_gl.append(
            float(y_value)
        )

    x_gl = np.asarray(
        x_gl,
        dtype=float,
    )

    y_gl = np.asarray(
        y_gl,
        dtype=float,
    )

    if x_gl.size == 0:

        print(
            "Warning: no grounding line "
            "was found."
        )

        return (
            x_gl,
            y_gl,
        )

    order = np.argsort(
        y_gl
    )

    x_gl = x_gl[order]
    y_gl = y_gl[order]

    print(
        f"  Grounding line: "
        f"{x_gl.size} rows, "
        f"x = {np.min(x_gl):.3f}"
        f"–{np.max(x_gl):.3f} km"
    )

    return (
        x_gl,
        y_gl,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # One dictionary per horizontal resolution:
    #
    # grounding_lines[resolution][dt] = (x_gl, y_gl)
    grounding_lines = {
        resolution: {}
        for resolution in resolutions
    }

    # -------------------------------------------------------------
    # Extract GL for every resolution and timestep simulation
    # -------------------------------------------------------------

    for resolution in resolutions:

        print("\n" + "=" * 72)
        print(f"Processing dx = {resolution:g} m")
        print("=" * 72)

        for dt in dts:

            try:

                mesh, H, zb = load_state(
                    dt,
                    resolution,
                )

            except (
                FileNotFoundError,
                RuntimeError,
            ) as exc:

                print(
                    f"Warning: {exc}"
                )

                continue

            x_gl, y_gl = grounding_line_curve(
                H,
                zb,
            )

            if x_gl.size == 0:
                continue

            grounding_lines[resolution][dt] = (
                x_gl,
                y_gl,
            )

    if not any(
        grounding_lines[resolution]
        for resolution in resolutions
    ):

        raise RuntimeError(
            "No grounding lines were extracted."
        )

    # -------------------------------------------------------------
    # Plot: panel (a) dx = 250 m, panel (b) dx = 500 m
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    cmap = plt.get_cmap(
        "viridis"
    )

    # Use the same colours for each dt in both panels.
    colors = cmap(
        np.linspace(
            0.15,
            0.85,
            len(dts),
        )
    )

    dt_colors = {
        dt: color
        for dt, color in zip(
            dts,
            colors,
        )
    }

    panel_labels = ["(a)", "(b)"]

    for panel_index, (ax, resolution) in enumerate(
        zip(axes, resolutions)
    ):

        panel_data = grounding_lines[
            resolution
        ]

        for dt in dts:

            if dt not in panel_data:
                continue

            x_gl, y_gl = panel_data[
                dt
            ]

            if np.isclose(
                dt,
                ref_dt,
            ):

                linewidth = 3.0
                linestyle = "-"
                zorder = 10

            else:

                linewidth = 2.0
                linestyle = "--"
                zorder = 5

            ax.plot(
                x_gl,
                y_gl,
                color=dt_colors[dt],
                linewidth=linewidth,
                linestyle=linestyle,
                label=fr"$\Delta t={dt:g}$ a",
                zorder=zorder,
            )

        ax.set_xlabel(
            r"Grounding-line position $x_{\mathrm{GL}}$ [km]",
            fontsize=18,
        )

        ax.tick_params(
            axis="both",
            labelsize=15,
        )

        ax.set_yticks(
            [40, 50, 60, 70, 80]
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_title(
            rf"$\Delta x={resolution:g}$ m",
            fontsize=17,
        )

        ax.text(
            0.03,
            0.96,
            panel_labels[panel_index],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.legend(
            fontsize=11,
            frameon=True,
        )

    axes[0].set_ylabel(
        r"$y$ [km]",
        fontsize=18,
    )

    output_filename = (
        f"{output_directory}/"
        f"MISMIP_Ice1r_SEP3_grounding_line_"
        f"dx250_dx500_vs_dt.png"
    )

    fig.savefig(
        output_filename,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    print(
        f"\nSaved grounding-line comparison to:\n"
        f"  {output_filename}"
    )

    plt.show()


if __name__ == "__main__":
    main()
