import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 2000

dt = 2
theta = 1
zeta_pred = False

output_directory = "Simulations"

number_of_filled_levels = 40
number_of_contour_levels = 10

plot_contour_lines = True
figure_dpi = 700


def truncate_colormap(cmap_name, minimum, maximum, name):
    """Return a colormap containing only part of a Matplotlib colormap."""

    base_cmap = plt.get_cmap(cmap_name)
    colors = base_cmap(
        np.linspace(
            minimum,
            maximum,
            256,
        )
    )

    return LinearSegmentedColormap.from_list(
        name,
        colors,
    )


# Split Spectral into two complementary halves.
# Lower half: red/orange through pale yellow.
# Upper half: pale yellow through green/blue.
spectral_lower = truncate_colormap(
    "Spectral_r",
    0.6,
    1.0,
    "Spectral_lower_half",
)

spectral_upper = truncate_colormap(
    "Spectral",
    0.6,
    1.0,
    "Spectral_upper_half",
)

# Nodes for which the hydrostatic flotation difference is larger than
# this tolerance are treated as grounded. The difference is
#
#     zb + (rho_i / rho_w) * H.
#
# It is zero for freely floating ice and positive for grounded ice.
grounding_line_tolerance = 0.1

# Only the density ratio is needed for the flotation criterion.
rhoi = 917.0
rhow = 1028.0
grounding_line_color = "black"
grounding_line_width = 2.2


# ---------------------------------------------------------------------
# Load restart data
# ---------------------------------------------------------------------

def load_state(dt, theta):
    """
    Load the mesh, saved geometry fields and velocity.
    """

    filename = (
        f"{output_directory}/"
        f"MISMIP_output_theta{theta:g}_dt{dt:g}_GL_pred{zeta_pred}/"
        f"restart_t{T:g}.h5"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Restart file not found:\n{filename}\n"
            "Check T, dt, theta and zeta_pred."
        )

    print(f"Loading restart file:\n{filename}")

    with CheckpointFile(filename, "r") as afile:

        try:
            mesh = afile.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )

        except Exception:
            mesh = afile.load_mesh(
                reorder=False,
            )

        H = afile.load_function(
            mesh,
            "thick",
        )

        zs = afile.load_function(
            mesh,
            "zs",
        )

        zb = afile.load_function(
            mesh,
            "zb",
        )

        # Velocity used by the restart state.
        velocity = afile.load_function(
            mesh,
            "uvec_out",
        )

    return mesh, H, zs, zb, velocity


# ---------------------------------------------------------------------
# Combine repeated horizontal coordinates
# ---------------------------------------------------------------------

def combine_duplicate_xy(
    x_values,
    y_values,
    field_values,
    decimals=12,
):
    """
    Average values that have identical horizontal coordinates.
    """

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

    if not (
        x_values.size
        == y_values.size
        == field_values.size
    ):
        raise ValueError(
            "Coordinate and field arrays have different sizes:\n"
            f"x: {x_values.size}\n"
            f"y: {y_values.size}\n"
            f"field: {field_values.size}"
        )

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

    field_unique = field_sum / counts

    # Sort first by x and then by y.
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
# Extract horizontal thickness field
# ---------------------------------------------------------------------

def thickness_surface(H):
    """
    Return the horizontal thickness field H(x, y).

    Repeated values through the vertical column are averaged.
    """

    V = H.function_space()
    mesh = H.ufl_domain()

    coordinates = SpatialCoordinate(mesh)

    x_on_H = Function(
        V,
        name="x_on_H",
    )

    y_on_H = Function(
        V,
        name="y_on_H",
    )

    x_on_H.interpolate(
        coordinates[0]
    )

    y_on_H.interpolate(
        coordinates[1]
    )

    x_values = np.asarray(
        x_on_H.dat.data_ro,
        dtype=float,
    )

    y_values = np.asarray(
        y_on_H.dat.data_ro,
        dtype=float,
    )

    H_values = np.asarray(
        H.dat.data_ro,
        dtype=float,
    )

    x_unique, y_unique, H_unique = combine_duplicate_xy(
        x_values,
        y_values,
        H_values,
    )

    return (
        x_unique / 1000.0,
        y_unique / 1000.0,
        H_unique,
    )


# ---------------------------------------------------------------------
# Extract the grounding-line curve from hydrostatic equilibrium
# ---------------------------------------------------------------------

def grounding_line_curve(H, zb):
    """
    Extract an explicit grounding-line curve x_GL(y).

    The hydrostatic flotation difference is

        D = zb + (rho_i / rho_w) * H.

    Floating ice has D approximately zero, while grounded ice has
    D greater than zero. For each horizontal y-row, this function finds
    the transition from grounded to floating ice as x increases and
    linearly interpolates its x-coordinate.

    Returns
    -------
    x_gl : ndarray
        Grounding-line x coordinates in km.
    y_gl : ndarray
        Grounding-line y coordinates in km.
    difference_values : ndarray
        Hydrostatic flotation difference at all horizontal nodes in m.
    """

    V = H.function_space()

    flotation_difference = Function(
        V,
        name="hydrostatic_flotation_difference",
    )

    flotation_difference.interpolate(
        zb + Constant(rhoi / rhow) * H
    )

    x, y, difference_values = thickness_surface(
        flotation_difference
    )

    # Group nodes into horizontal y-rows. Coordinates are already in km.
    y_rounded = np.round(y, decimals=10)
    unique_y = np.unique(y_rounded)

    x_gl = []
    y_gl = []

    total_grounded = int(
        np.count_nonzero(
            difference_values > grounding_line_tolerance
        )
    )
    total_floating = int(
        difference_values.size - total_grounded
    )

    print(
        "Grounding-line classification: "
        f"{total_grounded} grounded nodes, "
        f"{total_floating} floating nodes "
        f"using tolerance {grounding_line_tolerance:g} m"
    )

    for y_value in unique_y:
        row = np.where(y_rounded == y_value)[0]

        if row.size < 2:
            continue

        row_order = np.argsort(x[row])
        row = row[row_order]

        x_row = x[row]
        difference_row = difference_values[row]
        grounded_row = (
            difference_row > grounding_line_tolerance
        )

        # Find every change between grounded and floating. In MISMIP+
        # the relevant transition is normally grounded -> floating as x
        # increases. If several transitions occur, keep the downstream-most.
        state_changes = np.where(
            grounded_row[:-1] != grounded_row[1:]
        )[0]

        if state_changes.size == 0:
            continue

        grounded_to_floating = state_changes[
            grounded_row[state_changes]
            & ~grounded_row[state_changes + 1]
        ]

        if grounded_to_floating.size > 0:
            transition = grounded_to_floating[-1]
        else:
            transition = state_changes[-1]

        x0 = x_row[transition]
        x1 = x_row[transition + 1]
        d0 = difference_row[transition]
        d1 = difference_row[transition + 1]

        # Interpolate to D = grounding_line_tolerance. If the two values
        # are effectively identical, use the midpoint.
        if np.isclose(d0, d1):
            x_crossing = 0.5 * (x0 + x1)
        else:
            fraction = (
                (grounding_line_tolerance - d0)
                / (d1 - d0)
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            x_crossing = x0 + fraction * (x1 - x0)

        x_gl.append(x_crossing)
        y_gl.append(float(y_value))

    x_gl = np.asarray(x_gl, dtype=float)
    y_gl = np.asarray(y_gl, dtype=float)

    if x_gl.size == 0:
        print(
            "Warning: no grounded-to-floating transition was found on "
            "any y-row. Check the printed flotation-difference range and "
            "try changing grounding_line_tolerance."
        )
        return x_gl, y_gl, difference_values

    order = np.argsort(y_gl)
    x_gl = x_gl[order]
    y_gl = y_gl[order]

    print(
        f"Grounding line extracted on {x_gl.size} y-rows; "
        f"x range = {np.min(x_gl):.3f} to {np.max(x_gl):.3f} km"
    )

    return x_gl, y_gl, difference_values


def add_grounding_line(ax, x_gl, y_gl):
    """Draw an explicit grounding-line curve on an existing axis."""

    if x_gl.size == 0:
        return None

    line, = ax.plot(
        x_gl,
        y_gl,
        color=grounding_line_color,
        linewidth=grounding_line_width,
        zorder=40,
        label="Grounding line",
    )

    return line


# ---------------------------------------------------------------------
# Extract upper-surface velocity magnitude
# ---------------------------------------------------------------------

def velocity_surface(velocity):
    """
    Extract the upper-surface horizontal speed.

    The plotted speed is

        sqrt(u_x**2 + u_y**2)

    and therefore excludes any vertical velocity component.
    """

    mesh = velocity.ufl_domain()

    if len(velocity.ufl_shape) != 1:
        raise ValueError(
            "uvec_out is not vector-valued."
        )

    if velocity.ufl_shape[0] < 2:
        raise ValueError(
            "uvec_out has fewer than two components."
        )

    # Scalar space with explicit vertical degrees of freedom.
    Q1 = FunctionSpace(
        mesh,
        "CG",
        1,
    )

    coordinates = SpatialCoordinate(mesh)

    x_field = Function(
        Q1,
        name="velocity_x_coordinate",
    )

    y_field = Function(
        Q1,
        name="velocity_y_coordinate",
    )

    z_field = Function(
        Q1,
        name="velocity_z_coordinate",
    )

    speed_field = Function(
        Q1,
        name="surface_horizontal_speed",
    )

    x_field.interpolate(
        coordinates[0]
    )

    y_field.interpolate(
        coordinates[1]
    )

    z_field.interpolate(
        coordinates[2]
    )

    speed_field.interpolate(
        sqrt(
            velocity[0] ** 2
            + velocity[1] ** 2
        )
    )

    # Select degrees of freedom on the logical top boundary.
    top_nodes = np.asarray(
        DirichletBC(
            Q1,
            0.0,
            "top",
        ).nodes,
        dtype=np.int64,
    )

    if top_nodes.size > 0:

        x_values = np.asarray(
            x_field.dat.data_ro_with_halos[top_nodes],
            dtype=float,
        )

        y_values = np.asarray(
            y_field.dat.data_ro_with_halos[top_nodes],
            dtype=float,
        )

        speed_values = np.asarray(
            speed_field.dat.data_ro_with_halos[top_nodes],
            dtype=float,
        )

    else:
        # Fallback: select the node with maximum physical z in
        # each vertical column.
        print(
            "Warning: no logical top nodes were found. "
            "Selecting maximum-z nodes instead."
        )

        x_all = np.asarray(
            x_field.dat.data_ro,
            dtype=float,
        ).reshape(-1)

        y_all = np.asarray(
            y_field.dat.data_ro,
            dtype=float,
        ).reshape(-1)

        z_all = np.asarray(
            z_field.dat.data_ro,
            dtype=float,
        ).reshape(-1)

        speed_all = np.asarray(
            speed_field.dat.data_ro,
            dtype=float,
        ).reshape(-1)

        xy_rounded = np.round(
            np.column_stack(
                (
                    x_all,
                    y_all,
                )
            ),
            decimals=12,
        )

        xy_unique, inverse = np.unique(
            xy_rounded,
            axis=0,
            return_inverse=True,
        )

        selected_nodes = np.empty(
            xy_unique.shape[0],
            dtype=np.int64,
        )

        for column in range(
            xy_unique.shape[0]
        ):
            column_nodes = np.where(
                inverse == column
            )[0]

            selected_nodes[column] = column_nodes[
                np.argmax(
                    z_all[column_nodes]
                )
            ]

        x_values = x_all[selected_nodes]
        y_values = y_all[selected_nodes]
        speed_values = speed_all[selected_nodes]

    x_unique, y_unique, speed_unique = combine_duplicate_xy(
        x_values,
        y_values,
        speed_values,
    )

    return (
        x_unique / 1000.0,
        y_unique / 1000.0,
        speed_unique,
    )


# ---------------------------------------------------------------------
# Plot one horizontal scalar field
# ---------------------------------------------------------------------

def plot_horizontal_field(
    ax,
    x,
    y,
    values,
    title,
    colorbar_label,
    cmap,
    contour_format,
):
    """
    Plot a scalar field using filled triangular contours.
    """

    triangulation = mtri.Triangulation(
        x,
        y,
    )

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if np.isclose(
        minimum,
        maximum,
    ):
        minimum -= 1.0
        maximum += 1.0

    filled_levels = np.linspace(
        minimum,
        maximum,
        number_of_filled_levels,
    )

    contour_levels = np.linspace(
        minimum,
        maximum,
        number_of_contour_levels,
    )

    surface = ax.tricontourf(
        triangulation,
        values,
        levels=filled_levels,
        cmap=cmap,
        extend="both",
    )

    if plot_contour_lines:
        contours = ax.tricontour(
            triangulation,
            values,
            levels=contour_levels,
            colors="black",
            linewidths=0.4,
            alpha=0.5,
        )

        ax.clabel(
            contours,
            fontsize=7,
            fmt=contour_format,
        )

    colorbar = plt.colorbar(
        surface,
        ax=ax,
        pad=0.02,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=13,
    )

    colorbar.ax.tick_params(
        labelsize=10,
    )

    ax.set_xlabel(
        r"$x$ [km]",
        fontsize=13,
    )

    ax.set_ylabel(
        r"$y$ [km]",
        fontsize=13,
    )

    ax.set_title(
        title,
        fontsize=14,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    # Use "equal" for the true physical aspect ratio.
    # "auto" makes the long domain easier to inspect.
    ax.set_aspect("auto")

    return surface


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    mesh, H, zs, zb, velocity = load_state(
        dt,
        theta,
    )

    x_H, y_H, H_values = thickness_surface(
        H
    )

    x_u, y_u, speed_values = velocity_surface(
        velocity
    )

    (
        x_gl,
        y_gl,
        flotation_difference,
    ) = grounding_line_curve(
        H,
        zb,
    )

    print(
        "Thickness range: "
        f"{np.min(H_values):.3f} to "
        f"{np.max(H_values):.3f} m"
    )

    print(
        "Upper-surface horizontal speed range: "
        f"{np.min(speed_values):.3f} to "
        f"{np.max(speed_values):.3f} m/yr"
    )

    print(
        "Hydrostatic flotation-difference range: "
        f"{np.min(flotation_difference):.6e} to "
        f"{np.max(flotation_difference):.6e} m"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        constrained_layout=True,
    )

    plot_horizontal_field(
        ax=axes[0],
        x=x_H,
        y=y_H,
        values=H_values,
        title=(
            fr"Ice thickness at $T={T:g}$ years, "
            fr"$\Delta t={dt:g}$, $\theta={theta:g}$"
        ),
        colorbar_label=(
            r"Ice thickness, $H$ [m]"
        ),
        cmap=spectral_lower,
        contour_format="%.0f",
    )

    plot_horizontal_field(
        ax=axes[1],
        x=x_u,
        y=y_u,
        values=speed_values,
        title=(
            fr"Upper-surface horizontal speed at "
            fr"$T={T:g}$ years"
        ),
        colorbar_label=(
            r"Horizontal speed, "
            r"$\sqrt{u_x^2+u_y^2}$ "
            r"[m yr$^{-1}$]"
        ),
        cmap=spectral_upper,
        contour_format="%.0f",
    )

    grounding_line_drawn = False

    for ax in axes:
        line = add_grounding_line(
            ax,
            x_gl,
            y_gl,
        )

        grounding_line_drawn = (
            grounding_line_drawn
            or line is not None
        )

    if grounding_line_drawn:
        grounding_line_handle = Line2D(
            [0],
            [0],
            color=grounding_line_color,
            linewidth=grounding_line_width,
            label="Grounding line",
        )

        for ax in axes:
            ax.legend(
                handles=[grounding_line_handle],
                loc="best",
                fontsize=10,
                frameon=True,
            )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    output_filename = (
        f"{output_directory}/"
        f"thickness_velocity_and_grounding_line_"
        f"theta{theta:g}_dt{dt:g}_T{T:g}.png"
    )

    plt.savefig(
        output_filename,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    print(
        f"Saved figure to:\n{output_filename}"
    )

    plt.show()


if __name__ == "__main__":
    main()