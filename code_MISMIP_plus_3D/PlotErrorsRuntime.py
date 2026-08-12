from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import re


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 10100

theta = 1
resolutions = [250, 500]
zeta_pred = False

# SEP3 runs only.
dts = [1, 2, 5, 10, 20]

# Reference solution.
ref_dt = 1

# Ignore locations where the reference thickness is <= this value.
H_min = 10.0

simulation_directory = "Simulations"

figure_dpi = 700

dt_figure_filename = (
    f"{simulation_directory}/"
    f"MISMIP_Ice1r_SEP3_L2_error_vs_dt_dx250_dx500.png"
)

runtime_figure_filename = (
    f"{simulation_directory}/"
    f"MISMIP_Ice1r_SEP3_L2_error_vs_runtime_dx250_dx500.png"
)

csv_filename = (
    f"{simulation_directory}/"
    f"MISMIP_Ice1r_SEP3_L2_error_dx250_dx500.csv"
)

runtime_log_filename = (
    f"{simulation_directory}/"
    f"MISMIP_Ice1r_simulation_times.txt"
)


# ---------------------------------------------------------------------
# Restart-file discovery
# ---------------------------------------------------------------------

def find_restart_file(dt, resolution):

    filename = os.path.join(
        simulation_directory,
        (
            f"MISMIP_Ice1r_theta{theta:g}_dt{dt:g}_"
            f"GL_pred{zeta_pred}_res_{resolution}_GL_SEP3"
        ),
        f"restart_t{T:g}.h5",
    )

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"Restart file not found:{filename}"
        )

    print(
        f"dx={resolution:g} m, dt={dt:g}: "
        f"using restart file"
        f"  {filename}"
    )

    return filename


# ---------------------------------------------------------------------
# Load one simulation state
# ---------------------------------------------------------------------

def load_state(dt, resolution):

    filename = find_restart_file(dt, resolution)

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
                    f"No mesh found in checkpoint {filename}"
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

        zs = afile.load_function(
            mesh,
            "zs",
        )

    return mesh, H, zs


# ---------------------------------------------------------------------
# Absolute L2 thickness error
# ---------------------------------------------------------------------

def L2_error(
    H,
    Href,
    zs,
    h_min=10.0,
):
    """
    Absolute L2 error in ice thickness:

        ||H - Href||_L2
        =
        sqrt(
            integral (H - Href)^2 dA
        )

    where dA is horizontal plan-view area.

    Locations where Href <= h_min are excluded.
    """

    if H.dat.data_ro.size != Href.dat.data_ro.size:

        raise ValueError(
            "Reference and test thickness fields have "
            "different numbers of degrees of freedom."
        )

    # Put reference data into the current function space.
    Href_on_H = Function(
        H.function_space(),
        name="Href_on_H",
    )

    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    # Vertical component of the upper-surface unit normal.
    #
    # n_z * ds_t = horizontal plan-view area.
    n_z = 1.0 / sqrt(
        1.0
        + zs.dx(0)**2
        + zs.dx(1)**2
    )

    error_squared = conditional(
        Href_on_H > h_min,
        (H - Href_on_H)**2,
        0.0,
    )

    integral_error_squared = assemble(
        error_squared
        * n_z
        * ds_t
    )

    return float(
        np.sqrt(
            integral_error_squared
        )
    )


# ---------------------------------------------------------------------
# Runtime parsing
# ---------------------------------------------------------------------

def runtime_to_seconds(runtime_text):

    parts = runtime_text.strip().split(":")

    if len(parts) != 3:

        raise ValueError(
            f"Unexpected runtime format: "
            f"{runtime_text}"
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return (
        3600.0 * hours
        + 60.0 * minutes
        + seconds
    )


def load_runtime_records(filename):

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"Runtime log not found: "
            f"{filename}"
        )

    # Supports both:
    #
    # dt=..., theta_out=..., resolution=250, T_start=..., T_end=..., ...
    #
    # and older lines without a resolution field. Older lines cannot be
    # assigned safely once more than one resolution is present, so they are
    # ignored for the runtime panels.
    pattern = re.compile(
        r"dt=(?P<dt>[-+0-9.eE]+),\s*"
        r"theta_out=(?P<theta>[-+0-9.eE]+),\s*"
        r"(?:resolution=(?P<resolution>\d+),\s*)?"
        r"T_start=(?P<T_start>[-+0-9.eE]+),\s*"
        r"T_end=(?P<T_end>[-+0-9.eE]+),.*?"
        r"runtime=(?P<runtime>\d+:\d+:\d+(?:\.\d+)?)"
    )

    records = {}

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as runtime_file:

        for line_number, line in enumerate(
            runtime_file,
            start=1,
        ):

            match = pattern.search(
                line.strip()
            )

            if match is None:
                continue

            dt_value = float(
                match.group("dt")
            )

            theta_value = float(
                match.group("theta")
            )

            T_end = float(
                match.group("T_end")
            )

            runtime_text = match.group(
                "runtime"
            )

            resolution_text = match.group(
                "resolution"
            )

            if not np.isclose(
                theta_value,
                theta,
            ):
                continue

            if not np.isclose(
                T_end,
                T,
            ):
                continue

            # Once we compare multiple resolutions, runtime lines must
            # include the resolution explicitly.
            if resolution_text is None:
                print(
                    f"Warning: runtime-log line {line_number} has no "
                    "resolution field; skipping it."
                )
                continue

            resolution_value = int(
                resolution_text
            )

            if resolution_value not in resolutions:
                continue

            key = (
                resolution_value,
                dt_value,
            )

            records[key] = {
                "runtime_text": runtime_text,
                "runtime_seconds": runtime_to_seconds(
                    runtime_text
                ),
                "line_number": line_number,
            }

    print(
        f"Read {len(records)} matching runtime records "
        f"from {filename}"
    )

    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    os.makedirs(
        simulation_directory,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Read runtimes
    # -------------------------------------------------------------

    runtime_records = load_runtime_records(
        runtime_log_filename
    )

    all_rows = []

    # -------------------------------------------------------------
    # Compute L2 error separately for each resolution
    # -------------------------------------------------------------

    for resolution in resolutions:

        print("\n" + "=" * 72)
        print(
            f"Processing resolution = {resolution:g} m"
        )
        print("=" * 72)

        # Each resolution gets its own dt = ref_dt reference.
        _, Href, _ = load_state(
            ref_dt,
            resolution,
        )

        print(
            f"\nReference solution: "
            f"SEP3, dx={resolution:g} m, "
            f"dt={ref_dt:g} a, T={T:g} a"
        )

        print(
            f"Reference thickness range: "
            f"{np.min(Href.dat.data_ro):.3f} to "
            f"{np.max(Href.dat.data_ro):.3f} m\n"
        )

        for dt in dts:

            if np.isclose(
                dt,
                ref_dt,
            ):
                continue

            try:

                _, H, zs = load_state(
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

            error = L2_error(
                H,
                Href,
                zs,
                h_min=H_min,
            )

            runtime_record = runtime_records.get(
                (
                    resolution,
                    float(dt),
                )
            )

            if runtime_record is None:

                print(
                    f"Warning: no runtime found for "
                    f"dx={resolution:g}, dt={dt:g}"
                )

                runtime_seconds = None
                runtime_text = ""

            else:

                runtime_seconds = (
                    runtime_record[
                        "runtime_seconds"
                    ]
                )

                runtime_text = (
                    runtime_record[
                        "runtime_text"
                    ]
                )

            all_rows.append({
                "resolution": int(resolution),
                "dt": float(dt),
                "L2_error": error,
                "runtime_seconds": runtime_seconds,
                "runtime_text": runtime_text,
            })

            print(
                f"dx={resolution:g} m, "
                f"dt={dt:g} a: "
                f"L2 error={error:.8e}, "
                f"runtime="
                f"{runtime_text if runtime_text else 'not found'}"
            )

    if not all_rows:

        raise RuntimeError(
            "No non-reference SEP3 experiments "
            "were successfully loaded."
        )

    all_rows.sort(
        key=lambda row: (
            row["resolution"],
            row["dt"],
        )
    )

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------

    with open(
        csv_filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        fieldnames = [
            "T",
            "theta",
            "resolution_m",
            "reference_dt",
            "dt",
            "H_min_m",
            "L2_error",
            "runtime_seconds",
            "runtime_text",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in all_rows:

            writer.writerow({
                "T": f"{T:g}",
                "theta": f"{theta:g}",
                "resolution_m":
                    row["resolution"],
                "reference_dt":
                    f"{ref_dt:g}",
                "dt":
                    f"{row['dt']:g}",
                "H_min_m":
                    f"{H_min:g}",
                "L2_error":
                    f"{row['L2_error']:.10e}",
                "runtime_seconds":
                    ""
                    if row["runtime_seconds"] is None
                    else f"{row['runtime_seconds']:.9f}",
                "runtime_text":
                    row["runtime_text"],
            })

    print(
        f"\nSaved table to:\n"
        f"  {csv_filename}"
    )

    # -------------------------------------------------------------
    # Figure 1:
    # L2 error vs timestep, one panel per resolution
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        sharey=True,
        constrained_layout=True,
    )

    panel_labels = ["(a)", "(b)"]

    for panel, resolution in enumerate(
        resolutions
    ):

        ax = axes[panel]

        resolution_rows = [
            row
            for row in all_rows
            if row["resolution"] == resolution
        ]

        if not resolution_rows:

            ax.set_visible(False)
            continue

        plot_dts = np.asarray(
            [
                row["dt"]
                for row in resolution_rows
            ],
            dtype=float,
        )

        plot_errors = np.asarray(
            [
                row["L2_error"]
                for row in resolution_rows
            ],
            dtype=float,
        )

        order = np.argsort(
            plot_dts
        )

        ax.loglog(
            plot_dts[order],
            plot_errors[order],
            marker="o",
            linewidth=2.0,
            markersize=7,
        )

        ax.set_xlabel(
            r"$\Delta t$ [a]",
            fontsize=16,
        )

        ax.set_title(
            rf"$\Delta x={resolution:g}$ m",
            fontsize=16,
        )

        ax.tick_params(
            axis="both",
            labelsize=13,
        )

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        ax.text(
            0.04,
            0.95,
            panel_labels[panel],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )

    axes[0].set_ylabel(
        r"$L_2$ error in $H$ [m$^2$]",
        fontsize=16,
    )

    fig.savefig(
        dt_figure_filename,
        dpi=figure_dpi,
        bbox_inches="tight",
    )

    print(
        f"Saved timestep-error figure to:\n"
        f"  {dt_figure_filename}"
    )

    # -------------------------------------------------------------
    # Figure 2:
    # L2 error vs runtime, one panel per resolution
    # -------------------------------------------------------------

    fig_runtime, runtime_axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        sharey=True,
        constrained_layout=True,
    )

    any_runtime_data = False

    for panel, resolution in enumerate(
        resolutions
    ):

        ax = runtime_axes[panel]

        resolution_rows = [
            row
            for row in all_rows
            if (
                row["resolution"] == resolution
                and row["runtime_seconds"] is not None
                and row["runtime_seconds"] > 0.0
                and row["L2_error"] > 0.0
            )
        ]

        if not resolution_rows:

            ax.set_visible(False)
            continue

        any_runtime_data = True

        runtimes = np.asarray(
            [
                row["runtime_seconds"]
                for row in resolution_rows
            ],
            dtype=float,
        )

        runtime_errors = np.asarray(
            [
                row["L2_error"]
                for row in resolution_rows
            ],
            dtype=float,
        )

        runtime_dts = np.asarray(
            [
                row["dt"]
                for row in resolution_rows
            ],
            dtype=float,
        )

        order = np.argsort(
            runtimes
        )

        ax.loglog(
            runtimes[order],
            runtime_errors[order],
            marker="o",
            linewidth=2.0,
            markersize=7,
        )

        for runtime_value, error_value, dt_value in zip(
            runtimes,
            runtime_errors,
            runtime_dts,
        ):

            ax.annotate(
                rf"$\Delta t={dt_value:g}$",
                (
                    runtime_value,
                    error_value,
                ),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=11,
            )

        ax.set_xlabel(
            "Wall-clock runtime [s]",
            fontsize=16,
        )

        ax.set_title(
            rf"$\Delta x={resolution:g}$ m",
            fontsize=16,
        )

        ax.tick_params(
            axis="both",
            labelsize=13,
        )

        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )

        ax.text(
            0.04,
            0.95,
            panel_labels[panel],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if any_runtime_data:

        runtime_axes[0].set_ylabel(
            r"$L_2$ error in $H$ [m$^2$]",
            fontsize=16,
        )

        fig_runtime.savefig(
            runtime_figure_filename,
            dpi=figure_dpi,
            bbox_inches="tight",
        )

        print(
            f"Saved L2-error/runtime figure to:\n"
            f"  {runtime_figure_filename}"
        )

    else:

        print(
            "Warning: no matching runtime records with "
            "resolution information were found, so the "
            "runtime figure was not made."
        )

    plt.show()


if __name__ == "__main__":
    main()
