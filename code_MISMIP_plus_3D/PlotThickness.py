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

T = 10000

dt = 2
theta = 1
zeta_pred = False

output_directory = "Simulations"

number_of_filled_levels = 40
number_of_contour_levels = 10

plot_contour_lines = False
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
        f"Ice0_theta1_dt1_res2000_1000_nz10/"
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
# Extract grounding line using the TRUE Firedrake base-mesh connectivity
# ---------------------------------------------------------------------

def get_base_mesh(mesh):

    base_mesh = getattr(
        mesh,
        "_base_mesh",
        None,
    )

    if base_mesh is None:

        topology = getattr(
            mesh,
            "topology",
            None,
        )

        if topology is not None:

            base_mesh = getattr(
                topology,
                "_base_mesh",
                None,
            )

    if base_mesh is None:

        raise RuntimeError(
            "Could not obtain the horizontal base mesh from the "
            "extruded checkpoint mesh."
        )

    return base_mesh


def make_base_grounding_fields(mesh, H, zb):
    """
    Build grounded/floating information on the actual horizontal base mesh.

    This avoids creating a new Delaunay triangulation from an unordered
    x-y point cloud.
    """

    base_mesh = get_base_mesh(mesh)

    # Horizontally CG1, vertically constant.
    U = FunctionSpace(
        mesh,
        "CG",
        1,
        vfamily="R",
        vdegree=0,
    )

    D_column = Function(
        U,
        name="D_column",
    )

    zeta_column = Function(
        U,
        name="zeta_column",
    )

    D_column.interpolate(
        zb
        + Constant(rhoi / rhow) * H
    )

    zeta_column.interpolate(
        conditional(
            D_column > grounding_line_tolerance,
            0.0,
            1.0,
        )
    )

    Vbase = FunctionSpace(
        base_mesh,
        "CG",
        1,
    )

    zeta_base = Function(
        Vbase,
        name="zeta_base",
    )

    if (
        zeta_column.dat.data_ro.size
        != zeta_base.dat.data_ro.size
    ):

        raise RuntimeError(
            "Cannot safely transfer the vertically constant grounding "
            "mask to the base mesh: the DoF counts differ."
        )

    zeta_base.dat.data[:] = (
        zeta_column.dat.data_ro[:]
    )

    return (
        base_mesh,
        zeta_base,
    )


def edge_crossing(
    p0,
    p1,
    v0,
    v1,
    level=0.5,
):

    a0 = v0 - level
    a1 = v1 - level

    if a0 * a1 >= 0.0:
        return None

    fraction = (
        level - v0
    ) / (
        v1 - v0
    )

    return (
        p0
        + fraction
        * (
            p1 - p0
        )
    )


def grounding_line_segments_on_base_mesh(
    base_mesh,
    zeta_base,
):
    """
    Find zeta=0.5 on each REAL triangular base-mesh cell.
    """

    V = zeta_base.function_space()

    xcoord, ycoord = SpatialCoordinate(
        base_mesh
    )

    x_field = Function(
        V,
        name="x_base",
    )

    y_field = Function(
        V,
        name="y_base",
    )

    x_field.interpolate(
        xcoord
    )

    y_field.interpolate(
        ycoord
    )

    x = np.asarray(
        x_field.dat.data_ro_with_halos,
        dtype=float,
    )

    y = np.asarray(
        y_field.dat.data_ro_with_halos,
        dtype=float,
    )

    zeta = np.asarray(
        zeta_base.dat.data_ro_with_halos,
        dtype=float,
    )

    cell_nodes = np.asarray(
        V.cell_node_map().values,
        dtype=np.int64,
    )

    if cell_nodes.shape[1] != 3:

        raise RuntimeError(
            "Expected triangular CG1 base cells."
        )

    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    segments = []

    for nodes in cell_nodes:

        points = np.column_stack(
            (
                x[nodes],
                y[nodes],
            )
        )

        values = zeta[
            nodes
        ]

        crossings = []

        for i, j in edges:

            point = edge_crossing(
                points[i],
                points[j],
                values[i],
                values[j],
                level=0.5,
            )

            if point is not None:

                crossings.append(
                    point
                )

        if len(crossings) != 2:
            continue

        segment = np.vstack(
            (
                crossings[0],
                crossings[1],
            )
        ) / 1000.0

        segments.append(
            segment
        )

    if not segments:
        return []

    # Remove duplicate copies from shared cell boundaries.
    unique_segments = {}

    for segment in segments:

        a = tuple(
            np.round(
                segment[0],
                8,
            )
        )

        b = tuple(
            np.round(
                segment[1],
                8,
            )
        )

        key = tuple(
            sorted(
                (
                    a,
                    b,
                )
            )
        )

        unique_segments[key] = segment

    return list(
        unique_segments.values()
    )


def point_key(point):

    return tuple(
        np.round(
            point,
            7,
        )
    )


def connected_polylines(segments):
    """
    Join contour segments only when they actually share the same endpoint.
    """

    if not segments:
        return []

    adjacency = {}
    coordinates = {}
    edges = set()

    for segment in segments:

        a = point_key(
            segment[0]
        )

        b = point_key(
            segment[1]
        )

        if a == b:
            continue

        coordinates[a] = np.asarray(
            segment[0],
            dtype=float,
        )

        coordinates[b] = np.asarray(
            segment[1],
            dtype=float,
        )

        adjacency.setdefault(
            a,
            set(),
        ).add(
            b
        )

        adjacency.setdefault(
            b,
            set(),
        ).add(
            a
        )

        edges.add(
            tuple(
                sorted(
                    (
                        a,
                        b,
                    )
                )
            )
        )

    unvisited = set(
        adjacency.keys()
    )

    components = []

    while unvisited:

        seed = next(
            iter(
                unvisited
            )
        )

        stack = [seed]
        component = set()

        while stack:

            node = stack.pop()

            if node in component:
                continue

            component.add(
                node
            )

            unvisited.discard(
                node
            )

            for neighbour in adjacency.get(
                node,
                (),
            ):

                if neighbour not in component:

                    stack.append(
                        neighbour
                    )

        components.append(
            component
        )

    polylines = []

    for component in components:

        degrees = {
            node: len(
                [
                    neighbour
                    for neighbour in adjacency[node]
                    if neighbour in component
                ]
            )
            for node in component
        }

        endpoints = [
            node
            for node, degree in degrees.items()
            if degree == 1
        ]

        if endpoints:

            start = min(
                endpoints,
                key=lambda node: coordinates[node][1],
            )

        else:

            start = min(
                component,
                key=lambda node: coordinates[node][1],
            )

        ordered_nodes = [start]
        used_edges = set()

        previous = None
        current = start

        while True:

            candidates = []

            for neighbour in adjacency[
                current
            ]:

                if neighbour not in component:
                    continue

                edge = tuple(
                    sorted(
                        (
                            current,
                            neighbour,
                        )
                    )
                )

                if edge in used_edges:
                    continue

                candidates.append(
                    neighbour
                )

            if not candidates:
                break

            if (
                previous is None
                or len(candidates) == 1
            ):

                next_node = candidates[0]

            else:

                incoming = (
                    coordinates[current]
                    - coordinates[previous]
                )

                incoming_norm = np.linalg.norm(
                    incoming
                )

                best_score = -np.inf
                next_node = candidates[0]

                for candidate in candidates:

                    outgoing = (
                        coordinates[candidate]
                        - coordinates[current]
                    )

                    outgoing_norm = np.linalg.norm(
                        outgoing
                    )

                    if (
                        incoming_norm == 0.0
                        or outgoing_norm == 0.0
                    ):

                        score = -np.inf

                    else:

                        score = float(
                            np.dot(
                                incoming,
                                outgoing,
                            )
                            / (
                                incoming_norm
                                * outgoing_norm
                            )
                        )

                    if score > best_score:

                        best_score = score
                        next_node = candidate

            edge = tuple(
                sorted(
                    (
                        current,
                        next_node,
                    )
                )
            )

            used_edges.add(
                edge
            )

            previous = current
            current = next_node

            ordered_nodes.append(
                current
            )

            if current == start:
                break

        polyline = np.vstack(
            [
                coordinates[node]
                for node in ordered_nodes
            ]
        )

        polylines.append(
            polyline
        )

    return polylines


def grounding_line_curve(mesh, H, zb):
    """
    Return the connected zeta=0.5 contour component with the largest
    transverse y-span.
    """

    (
        base_mesh,
        zeta_base,
    ) = make_base_grounding_fields(
        mesh,
        H,
        zb,
    )

    segments = grounding_line_segments_on_base_mesh(
        base_mesh,
        zeta_base,
    )

    polylines = connected_polylines(
        segments
    )

    if not polylines:

        print(
            "Warning: no grounding-line contour was found."
        )

        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    def score(polyline):

        y_span = (
            np.max(
                polyline[:, 1]
            )
            - np.min(
                polyline[:, 1]
            )
        )

        return (
            y_span,
            polyline.shape[0],
        )

    grounding_line = max(
        polylines,
        key=score,
    )

    # Only reverse the complete polyline if needed.
    # Never sort x and y independently.
    if (
        grounding_line.shape[0] > 1
        and grounding_line[0, 1]
        > grounding_line[-1, 1]
    ):

        grounding_line = grounding_line[
            ::-1
        ]

    x_gl = grounding_line[
        :,
        0,
    ]

    y_gl = grounding_line[
        :,
        1,
    ]

    print(
        f"Grounding line: "
        f"{x_gl.size} ordered points, "
        f"x={np.min(x_gl):.3f} to "
        f"{np.max(x_gl):.3f} km, "
        f"y={np.min(y_gl):.3f} to "
        f"{np.max(y_gl):.3f} km"
    )

    return (
        x_gl,
        y_gl,
    )


def add_grounding_line(
    ax,
    x_gl,
    y_gl,
):
    """Draw the grounding line on an existing axis."""

    if x_gl.size == 0:
        return None

    line, = ax.plot(
        x_gl,
        y_gl,
        color=grounding_line_color,
        linewidth=grounding_line_width,
        linestyle="-",
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
    vmin=None,
    vmax=None,
    colorbar_ticks=None,
    yticks=None,
):
    """
    Plot a scalar field using filled triangular contours.
    """

    triangulation = mtri.Triangulation(
        x,
        y,
    )

    minimum = np.nanmin(values) if vmin is None else vmin
    maximum = np.nanmax(values) if vmax is None else vmax

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
        extend="max",
    )

    if yticks is not None:
        ax.set_yticks(yticks)

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
            fontsize=20,
            fmt=contour_format,
        )

    colorbar = plt.colorbar(
        surface,
        ax=ax,
        pad=0.02,
    )

    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)

    colorbar.set_label(
        colorbar_label,
        fontsize=25,
    )

    colorbar.ax.tick_params(
        labelsize=20,
    )

    ax.set_xlabel(
        r"$x$ [km]",
        fontsize=25,
    )

    ax.set_ylabel(
        r"$y$ [km]",
        fontsize=25,
    )

    ax.tick_params(
        axis="both",
        labelsize=20,
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
    ) = grounding_line_curve(
        mesh,
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
            r"$H$ [m]"
        ),
        cmap=spectral_lower,
        contour_format="%.0f",
        vmin=0,
        vmax=2000,
        yticks=[40, 50, 60, 70, 80],
        colorbar_ticks=[0, 500, 1000, 1500, 2000],
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
            r"$|\mathbf{u}_\mathrm{2D}|_{z_s}$"
            r" [m a$^{-1}$]"
        ),
        cmap=spectral_upper,
        contour_format="%.0f",
        vmin=0,
        vmax=1000,
        yticks=[40, 50, 60, 70, 80],
        colorbar_ticks=[0, 250, 500, 750, 1000],
    )

    axes[0].text(
        0.02,
        0.95,
        "(a)",
        transform=axes[0].transAxes,
        fontsize=25,
        va="top",
        ha="left",
    )

    axes[1].text(
        0.02,
        0.95,
        "(b)",
        transform=axes[1].transAxes,
        fontsize=25,
        va="top",
        ha="left",
    )

    # Limit the horizontal plotting range.
    for ax in axes:
        ax.set_xlim(300, 550)

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

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    output_filename = (
        f"{output_directory}/"
        f"thickness_velocity_and_grounding_line_t10000.png"
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
