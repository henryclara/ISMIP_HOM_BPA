import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 10100
theta = 1

dts = [0.1, 1, 2, 5, 10, 20]
ref_dt = 0.1

simulation_directory = "Simulations"
figure_dpi = 700

rhoi = 917.0
rhow = 1028.0

# zeta_out = 0 grounded, 1 floating.
zeta_level = 0.5

# Floating restart nodes theoretically satisfy
#     zb + (rho_i/rho_w) H = 0.
# This suppresses tiny positive roundoff in that clipped quantity.
grounding_tolerance = 1.0e-6

run_families = [
    {
        "key": "res2000_1000_im_mi",
        "folder_suffix": "res2000_1000_nz10_time_stepping_im_mi",
        "title": r"$\Delta x_{\mathrm{f}}=1000$ m, $\theta=0.5$",
    },
    {
        "key": "res2000_1000",
        "folder_suffix": "res2000_1000_nz10",
        "title": r"$\Delta x_{\mathrm{f}}=1000$ m",
    },
    {
        "key": "res2000_500",
        "folder_suffix": "res2000_500_nz10",
        "title": r"$\Delta x_{\mathrm{f}}=500$ m",
    },
]

output_filename = os.path.join(
    simulation_directory,
    f"Ice1r_grounding_lines_base_mesh_T{T:g}.png",
)

csv_directory = os.path.join(
    simulation_directory,
    f"grounding_line_base_mesh_T{T:g}",
)


# ---------------------------------------------------------------------
# Restart-file discovery
# ---------------------------------------------------------------------

def run_directory_candidates(dt, family):

    suffix = family["folder_suffix"]

    return [
        f"Ice1rr_theta{theta:g}_dt{dt:g}_{suffix}",
        f"Ice1r_theta{theta:g}_dt{dt:g}_{suffix}",
    ]


def find_restart_file(dt, family):

    tried = []

    for directory_name in run_directory_candidates(dt, family):

        filename = os.path.join(
            simulation_directory,
            directory_name,
            f"restart_t{T:g}.h5",
        )

        tried.append(filename)

        if os.path.exists(filename):
            return filename

    raise FileNotFoundError(
        "Restart file not found. Tried:\n  "
        + "\n  ".join(tried)
    )


# ---------------------------------------------------------------------
# Load restart state
# ---------------------------------------------------------------------

def load_mesh_from_checkpoint(afile, filename):

    try:

        return afile.load_mesh(
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

        return afile.load_mesh(
            name=mesh_name,
            reorder=False,
        )


def load_state(dt, family):

    filename = find_restart_file(
        dt,
        family,
    )

    print(
        f"\nLoading {family['key']}, dt={dt:g}:\n"
        f"  {filename}"
    )

    with CheckpointFile(filename, "r") as afile:

        mesh = load_mesh_from_checkpoint(
            afile,
            filename,
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
# Get the genuine horizontal base mesh
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
            "Could not obtain the base mesh from the extruded checkpoint "
            "mesh. This script needs the real horizontal mesh connectivity."
        )

    return base_mesh


# ---------------------------------------------------------------------
# Transfer a vertically constant field to the actual 2-D base mesh
# ---------------------------------------------------------------------

def make_base_fields(mesh, H, zb):
    """
    Build zeta, zb and zs on the ACTUAL base mesh.

    The important point is that we do not collect unordered (x, y) points
    and create a new Delaunay triangulation.  We use the original base-mesh
    cells and their node connectivity.
    """

    base_mesh = get_base_mesh(
        mesh
    )

    # CG1 horizontally, constant in the extrusion direction.
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

    zb_column = Function(
        U,
        name="zb_column",
    )

    zs_column = Function(
        U,
        name="zs_column",
    )

    D_column.interpolate(
        zb
        + Constant(rhoi / rhow) * H
    )

    zeta_column.interpolate(
        conditional(
            D_column > grounding_tolerance,
            0.0,
            1.0,
        )
    )

    zb_column.interpolate(
        zb
    )

    zs_column.interpolate(
        zb + H
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

    zb_base = Function(
        Vbase,
        name="zb_base",
    )

    zs_base = Function(
        Vbase,
        name="zs_base",
    )

    source_size = zeta_column.dat.data_ro.size
    target_size = zeta_base.dat.data_ro.size

    if source_size != target_size:

        raise RuntimeError(
            "The vertically constant extruded space and base CG1 space "
            "do not have the same number of owned DoFs:\n"
            f"  extruded CG1 x R: {source_size}\n"
            f"  base CG1:        {target_size}\n"
            "So a direct base-mesh transfer is not safe for this mesh."
        )

    zeta_base.dat.data[:] = zeta_column.dat.data_ro[:]
    zb_base.dat.data[:] = zb_column.dat.data_ro[:]
    zs_base.dat.data[:] = zs_column.dat.data_ro[:]

    print(
        f"  Base-mesh zeta range: "
        f"{np.min(zeta_base.dat.data_ro):.3g} to "
        f"{np.max(zeta_base.dat.data_ro):.3g}"
    )

    return (
        base_mesh,
        zeta_base,
        zb_base,
        zs_base,
    )


# ---------------------------------------------------------------------
# Marching triangles on the TRUE base-mesh cells
# ---------------------------------------------------------------------

def edge_crossing(p0, p1, v0, v1, level):

    a0 = v0 - level
    a1 = v1 - level

    if a0 * a1 >= 0.0:
        return None

    fraction = (
        level - v0
    ) / (
        v1 - v0
    )

    point = (
        p0
        + fraction
        * (p1 - p0)
    )

    return (
        point,
        fraction,
    )


def contour_segments_on_base_mesh(
    base_mesh,
    zeta_base,
    zb_base,
    zs_base,
):
    """
    Find zeta=0.5 using each REAL triangular base-mesh cell.

    This is the key difference from the earlier scripts:
    no np.unique(x, y) -> mtri.Triangulation(x, y) step is used.
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

    zb_values = np.asarray(
        zb_base.dat.data_ro_with_halos,
        dtype=float,
    )

    zs_values = np.asarray(
        zs_base.dat.data_ro_with_halos,
        dtype=float,
    )

    cell_nodes = np.asarray(
        V.cell_node_map().values,
        dtype=np.int64,
    )

    if cell_nodes.ndim != 2:

        raise RuntimeError(
            f"Unexpected base cell-node map shape: {cell_nodes.shape}"
        )

    if cell_nodes.shape[1] != 3:

        raise RuntimeError(
            "Expected triangular CG1 base cells with 3 nodes, but the "
            f"cell-node map has {cell_nodes.shape[1]} nodes per cell."
        )

    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    segments = []

    rejected_by_z0 = 0

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

            result = edge_crossing(
                points[i],
                points[j],
                values[i],
                values[j],
                zeta_level,
            )

            if result is None:
                continue

            point, fraction = result

            # Interpolate bottom/top elevation to this edge-crossing point.
            zb_cross = (
                zb_values[nodes[i]]
                + fraction
                * (
                    zb_values[nodes[j]]
                    - zb_values[nodes[i]]
                )
            )

            zs_cross = (
                zs_values[nodes[i]]
                + fraction
                * (
                    zs_values[nodes[j]]
                    - zs_values[nodes[i]]
                )
            )

            crossings.append(
                (
                    point,
                    float(zb_cross),
                    float(zs_cross),
                )
            )

        if len(crossings) != 2:
            continue

        p0, zb0, zs0 = crossings[0]
        p1, zb1, zs1 = crossings[1]

        # Mimic the z=0 slice: the contour curtain must span z=0.
        # In the grounding-line region this should normally be true.
        intersects0 = (
            (zb0 <= 0.0 <= zs0)
            or (zb1 <= 0.0 <= zs1)
            or (
                min(zb0, zb1) <= 0.0
                and max(zs0, zs1) >= 0.0
            )
        )

        if not intersects0:

            rejected_by_z0 += 1
            continue

        segment = np.vstack(
            (
                p0,
                p1,
            )
        ) / 1000.0

        segments.append(
            segment
        )

    if rejected_by_z0 > 0:

        print(
            f"  z=0 slice rejected {rejected_by_z0} contour segments."
        )

    if not segments:

        return []

    # Deduplicate cell-edge copies.
    unique = {}

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

        unique[key] = segment

    return list(
        unique.values()
    )


# ---------------------------------------------------------------------
# Stitch cell segments into connected ordered polylines
# ---------------------------------------------------------------------

def point_key(point):

    return tuple(
        np.round(
            point,
            7,
        )
    )


def connected_polylines(segments):

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

        stack = [
            seed
        ]

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

        component_edges = {
            edge
            for edge in edges
            if (
                edge[0] in component
                and edge[1] in component
            )
        }

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

            # Begin from the lower-y end.
            start = min(
                endpoints,
                key=lambda node: coordinates[node][1],
            )

        else:

            # Closed component: start from its lowest-y point.
            start = min(
                component,
                key=lambda node: coordinates[node][1],
            )

        ordered_nodes = [
            start
        ]

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

            if previous is None or len(candidates) == 1:

                next_node = candidates[0]

            else:

                # Preserve local direction if a rare branch/ambiguity occurs.
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

            if (
                current == start
                and len(used_edges) == len(component_edges)
            ):

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


def select_grounding_line(polylines):
    """
    Choose the connected contour branch with the largest transverse y-span.

    This should select the grounding line crossing the whole MISMIP+
    channel rather than small closed islands.
    """

    if not polylines:

        return np.empty(
            (
                0,
                2,
            ),
            dtype=float,
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

    ordered = sorted(
        polylines,
        key=score,
        reverse=True,
    )

    print(
        f"  Found {len(ordered)} connected zeta=0.5 contour components."
    )

    for index, polyline in enumerate(
        ordered[:5],
        start=1,
    ):

        print(
            f"    component {index}: "
            f"{polyline.shape[0]} points, "
            f"x={np.min(polyline[:,0]):.3f}--"
            f"{np.max(polyline[:,0]):.3f} km, "
            f"y={np.min(polyline[:,1]):.3f}--"
            f"{np.max(polyline[:,1]):.3f} km"
        )

    return ordered[0]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    os.makedirs(
        simulation_directory,
        exist_ok=True,
    )

    os.makedirs(
        csv_directory,
        exist_ok=True,
    )

    grounding_lines = {
        family["key"]: {}
        for family in run_families
    }

    for family in run_families:

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"Processing {family['key']}"
        )

        print(
            "=" * 78
        )

        for dt in dts:

            try:

                mesh, H, zb = load_state(
                    dt,
                    family,
                )

                (
                    base_mesh,
                    zeta_base,
                    zb_base,
                    zs_base,
                ) = make_base_fields(
                    mesh,
                    H,
                    zb,
                )

                segments = contour_segments_on_base_mesh(
                    base_mesh,
                    zeta_base,
                    zb_base,
                    zs_base,
                )

                polylines = connected_polylines(
                    segments
                )

                grounding_line = select_grounding_line(
                    polylines
                )

            except (
                FileNotFoundError,
                RuntimeError,
                ValueError,
            ) as exc:

                print(
                    f"Warning: {exc}"
                )

                continue

            if grounding_line.size == 0:

                print(
                    f"Warning: no grounding line found for "
                    f"{family['key']}, dt={dt:g}"
                )

                continue

            # If needed, reverse the whole ordered curve.  Never sort the
            # individual x/y points independently or by y.
            if (
                grounding_line.shape[0] > 1
                and grounding_line[0, 1]
                > grounding_line[-1, 1]
            ):

                grounding_line = grounding_line[
                    ::-1
                ]

            grounding_lines[
                family["key"]
            ][dt] = grounding_line

            csv_filename = os.path.join(
                csv_directory,
                (
                    f"{family['key']}"
                    f"_dt{dt:g}"
                    f"_T{T:g}.csv"
                ),
            )

            np.savetxt(
                csv_filename,
                grounding_line,
                delimiter=",",
                header="x_GL_km,y_km",
                comments="",
            )

            print(
                f"  SELECTED grounding line: "
                f"{grounding_line.shape[0]} ordered points, "
                f"x={np.min(grounding_line[:,0]):.3f}--"
                f"{np.max(grounding_line[:,0]):.3f} km, "
                f"y={np.min(grounding_line[:,1]):.3f}--"
                f"{np.max(grounding_line[:,1]):.3f} km"
            )

    if not any(
        grounding_lines[
            family["key"]
        ]
        for family in run_families
    ):

        raise RuntimeError(
            "No grounding lines were extracted."
        )

    # -------------------------------------------------------------
    # Plot limits
    # -------------------------------------------------------------

    all_lines = [
        line
        for family in run_families
        for line in grounding_lines[
            family["key"]
        ].values()
    ]

    all_points = np.vstack(
        all_lines
    )

    x_min = float(
        np.min(
            all_points[:, 0]
        )
    )

    x_max = float(
        np.max(
            all_points[:, 0]
        )
    )

    y_min = float(
        np.min(
            all_points[:, 1]
        )
    )

    y_max = float(
        np.max(
            all_points[:, 1]
        )
    )

    x_span = max(
        x_max - x_min,
        1.0,
    )

    y_span = max(
        y_max - y_min,
        1.0,
    )

    plot_xlim = (
        x_min - 0.06 * x_span,
        x_max + 0.06 * x_span,
    )

    plot_ylim = (
        y_min - 0.03 * y_span,
        y_max + 0.03 * y_span,
    )

    # -------------------------------------------------------------
    # Styling
    # -------------------------------------------------------------

    cmap = plt.get_cmap(
        "viridis"
    )

    colors = cmap(
        np.linspace(
            0.08,
            0.92,
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

    def curve_style(dt):

        if np.isclose(
            dt,
            ref_dt,
        ):

            return {
                "linewidth": 3.0,
                "linestyle": "-",
                "zorder": 20,
            }

        return {
            "linewidth": 2.0,
            "linestyle": "-",
            "zorder": 8,
        }

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    data_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
    ]

    legend_ax = axes[1, 1]

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
    ]

    for panel_index, (
        ax,
        family,
    ) in enumerate(
        zip(
            data_axes,
            run_families,
        )
    ):

        for dt in dts:

            if dt not in grounding_lines[
                family["key"]
            ]:

                continue

            line = grounding_lines[
                family["key"]
            ][dt]

            style = curve_style(
                dt
            )

            ax.plot(
                line[:, 0],
                line[:, 1],
                color=dt_colors[dt],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                zorder=style["zorder"],
            )

        ax.set_xlim(
            plot_xlim
        )

        ax.set_ylim(
            plot_ylim
        )

        ax.set_title(
            family["title"],
            fontsize=17,
            pad=10,
        )

        ax.tick_params(
            axis="both",
            labelsize=14,
        )

        ax.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )

        ax.text(
            0.025,
            0.965,
            panel_labels[
                panel_index
            ],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
        )

    axes[0, 0].set_ylabel(
        r"$y$ [km]",
        fontsize=17,
    )

    axes[1, 0].set_ylabel(
        r"$y$ [km]",
        fontsize=17,
    )

    axes[1, 0].set_xlabel(
        r"Grounding-line position $x_{\mathrm{GL}}$ [km]",
        fontsize=17,
    )

    axes[0, 1].set_xlabel(
        r"Grounding-line position $x_{\mathrm{GL}}$ [km]",
        fontsize=17,
    )

    # -------------------------------------------------------------
    # Shared legend
    # -------------------------------------------------------------

    legend_ax.axis(
        "off"
    )

    handles = []

    for dt in dts:

        style = curve_style(
            dt
        )

        label = fr"$\Delta t={dt:g}$ a"

        if np.isclose(
            dt,
            ref_dt,
        ):

            label += "  (reference)"

        handles.append(
            Line2D(
                [0],
                [0],
                color=dt_colors[dt],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                label=label,
            )
        )

    legend_ax.legend(
        handles=handles,
        loc="center",
        fontsize=14,
        frameon=False,
        handlelength=3.0,
        labelspacing=1.0,
    )

    legend_ax.text(
        0.5,
        0.88,
        rf"$T={T:g}$ a",
        transform=legend_ax.transAxes,
        fontsize=18,
        ha="center",
        va="center",
    )

    fig.savefig(
        output_filename,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    print(
        f"\nSaved figure to:\n"
        f"  {output_filename}"
    )

    print(
        f"\nSaved ordered grounding-line coordinates to:\n"
        f"  {csv_directory}"
    )

    plt.show()


if __name__ == "__main__":
    main()
