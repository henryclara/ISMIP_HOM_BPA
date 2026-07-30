import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 100

dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_values = [0, 1]

ref_dt = 0.01
ref_theta = 0

resolution_pairs = [
    (100, 10),
    (200, 20),
    (400, 40),
    (800, 80),
    (1600, 160),
]

theta_colors = {
    0: plt.cm.viridis(0.25),
    1: plt.cm.viridis(0.75),
}

dt_styles = {
    0.1: "-",
    1: "--",
    2: ":",
    5: "-.",
    10: (0, (5, 2)),
    20: (0, (3, 1, 1, 1)),
    50: (0, (1, 1)),
}


# ---------------------------------------------------------------------
# Load restart data
# ---------------------------------------------------------------------

def load_state(dt, theta, nx, nz):
    filename = (
        f"Simulations/BPA_output_dt{dt:g}_theta{theta:g}_nx{nx}_nz{nz}"
        f"/restart_t{T:g}.h5"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Restart file not found: {filename}.\n"
            "Check that the simulation produced this restart."
        )

    with CheckpointFile(filename, "r") as afile:
        # Try the named extruded mesh first
        try:
            mesh = afile.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")

        # Fall back to the default mesh
        except Exception:
            mesh = afile.load_mesh()
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")

    return mesh, H, zs


# ---------------------------------------------------------------------
# Extract the horizontal thickness profile
# ---------------------------------------------------------------------

def thickness_profile(H):
    """
    Return sorted horizontal coordinates and thickness values.

    The horizontal coordinate is interpolated into the same function
    space as H, ensuring that x and H have the same degree-of-freedom
    ordering.

    Repeated horizontal coordinates are combined by averaging. This can
    be necessary when the field is represented on an extruded mesh.
    """

    V = H.function_space()
    mesh = H.ufl_domain()

    # Put the horizontal coordinate into the same function space as H
    x_on_H = Function(V, name="x_on_H")
    x_on_H.interpolate(SpatialCoordinate(mesh)[0])

    x_values = np.asarray(
        x_on_H.dat.data_ro,
        dtype=float,
    ).reshape(-1)

    H_values = np.asarray(
        H.dat.data_ro,
        dtype=float,
    ).reshape(-1)

    if x_values.size != H_values.size:
        raise ValueError(
            "Coordinate and thickness arrays have different sizes:\n"
            f"x values: {x_values.size}\n"
            f"H values: {H_values.size}"
        )

    # Sort by horizontal coordinate
    order = np.argsort(x_values)

    x_values = x_values[order]
    H_values = H_values[order]

    # Round to prevent tiny floating-point differences from splitting
    # nominally identical horizontal coordinates
    x_rounded = np.round(x_values, decimals=12)

    x_unique, inverse = np.unique(
        x_rounded,
        return_inverse=True,
    )

    H_sum = np.zeros(
        x_unique.size,
        dtype=float,
    )

    counts = np.zeros(
        x_unique.size,
        dtype=int,
    )

    np.add.at(H_sum, inverse, H_values)
    np.add.at(counts, inverse, 1)

    H_unique = H_sum / counts

    return x_unique / 1000.0, H_unique


# ---------------------------------------------------------------------
# Set up the figure
# ---------------------------------------------------------------------

ncols = 3
nrows = int(
    np.ceil(len(resolution_pairs) / ncols)
)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.8 * ncols, 3.9 * nrows),
    sharex=False,
    sharey=True,
    squeeze=False,
)

axes_flat = axes.ravel()

# ---------------------------------------------------------------------
# Plot each mesh resolution
# ---------------------------------------------------------------------

for panel_index, (nx, nz) in enumerate(resolution_pairs):
    ax = axes_flat[panel_index]

    # Create a zoomed inset for this panel
    ax_zoom = inset_axes(
        ax,
        width="20%",
        height="30%",
        loc="upper left",
        bbox_to_anchor=(0.10, -0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    # -------------------------------------------------------------
    # Reference simulation
    # -------------------------------------------------------------

    try:
        _, Href, _ = load_state(
            ref_dt,
            ref_theta,
            nx,
            nz,
        )

    except FileNotFoundError as error:
        print(f"Warning: {error}")
        ax.set_visible(False)
        continue

    x_ref, H_ref = thickness_profile(Href)

    # Main panel
    ax.plot(
        x_ref,
        H_ref,
        color="black",
        linestyle="-",
        linewidth=1,
        zorder=20,
    )

    # Zoomed inset
    ax_zoom.plot(
        x_ref,
        H_ref,
        color="black",
        linestyle="-",
        linewidth=2.2,
        zorder=20,
    )

    # -------------------------------------------------------------
    # Finite-timestep simulations
    # -------------------------------------------------------------

    for theta in theta_values:
        for dt in dts:

            try:
                _, H, _ = load_state(
                    dt,
                    theta,
                    nx,
                    nz,
                )

            except FileNotFoundError as error:
                print(f"Warning: {error}")
                continue

            x, H_values = thickness_profile(H)

            plot_options = {
                "color": theta_colors[theta],
                "linestyle": dt_styles[dt],
                "linewidth": 1.8,
                "alpha": 0.9,
                "zorder": 5,
            }

            # Main profile
            ax.plot(
                x,
                H_values,
                **plot_options,
            )

            # Same profile in the inset
            ax_zoom.plot(
                x,
                H_values,
                **plot_options,
            )

    # Main-axis limits
    ax.set_ylim(0, 2500)
    ax.set_xticks(np.arange(0, 81, 20))
    ax.tick_params(
        axis="both",
        labelsize=15,
    )
    ax_zoom.set_xlim(64.500, 65.150)
    ax_zoom.set_ylim(1230, 1350)

    ax_zoom.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax_zoom.tick_params(
        axis="both",
        labelsize=7,
    )

    # Show the inset region on the main panel
    mark_inset(
        ax,
        ax_zoom,
        loc1=2,
        loc2=4,
        fc="none",
        ec="0.4",
        linewidth=0.8,
    )

    ax.set_title(
        fr"$n_x={nx},\ n_z={nz}$", fontsize=20
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )


# Hide the unused sixth panel
for ax in axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)


# ---------------------------------------------------------------------
# Shared labels and title
# ---------------------------------------------------------------------

fig.supxlabel(r"$x$ [m]", y=0.06, fontsize=20)
fig.supylabel(r"$H(x) [km]$", fontsize=20)

# ---------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------

reference_handle = Line2D(
    [0],
    [0],
    color="black",
    linestyle="-",
    linewidth=1,
    label=fr"Reference: $\Delta t={ref_dt:g}$, $\theta={ref_theta}$",
)

theta_handles = [
    Line2D(
        [0],
        [0],
        color=theta_colors[theta],
        linestyle="-",
        linewidth=2.5,
        label=fr"$\theta={theta}$",
    )
    for theta in theta_values
]

dt_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle=dt_styles[dt],
        linewidth=1,
        label=fr"$\Delta t={dt:g}$",
    )
    for dt in dts
]

legend_theta = fig.legend(
    handles=[reference_handle] + theta_handles,
    loc="lower left",
    bbox_to_anchor=(0.08, 0.01),
    ncols=3,
    fontsize=12,
    frameon=True,
)

legend_dt = fig.legend(
    handles=dt_handles,
    loc="lower right",
    bbox_to_anchor=(0.92, 0.01),
    ncols=4,
    fontsize=12,
    frameon=True,
)

fig.add_artist(legend_theta)

# ---------------------------------------------------------------------
# Save and display
# ---------------------------------------------------------------------

os.makedirs(
    "Simulations",
    exist_ok=True,
)

plt.tight_layout(
    rect=(0, 0.03, 1, 0.95)
)

output_filename = (
    f"Simulations/thickness_profiles_all_theta_T{T:g}.png"
)

plt.savefig(
    output_filename,
    dpi=700,
    bbox_inches="tight",
)

print(f"Saved figure to: {output_filename}")

plt.show()