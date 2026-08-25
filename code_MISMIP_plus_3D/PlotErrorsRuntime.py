from firedrake import *
from firedrake import CheckpointFile

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

times = [10100, 10200]
theta_values = [0, 1]

# These are the six dt values visible in the screenshot.
dts = [0.1, 1, 2, 5, 10, 20]

# Use the smallest timestep in each run family as that family's reference.
# Its error is exactly zero, so it is used as input but is not shown on a
# logarithmic error plot.
ref_dt = 0.1

H_min = 10.0
simulation_directory = "Simulations"
figure_dpi = 700

# Plot styling: match the Viridis theta colours used in the comparison script.
theta_colors = {
    0: plt.cm.viridis(0.1),
    1: plt.cm.viridis(0.8),
}

theta_styles = {
    0: "-",
    1: "--",
}

# Both run families appear on the same axes, so use marker shape to
# distinguish spatial resolution while colour/linestyle distinguish theta.
family_markers = {
    "res2000_1000": "o",
    "res2000_500": "s",
}

# The three folder families visible in the screenshot.  For each family the
# complete directory name is
#
#   theta=1: Ice1rr_theta1_dt<dt>_<folder_suffix>
#   theta=0: Ice1rr_theta0_dt<dt>_<folder_suffix>_time
#
run_families = [
    #{
    #   "key": "res2000_1000_im_mi",
    #    "folder_suffix": "res2000_1000_nz10_time_stepping_im_mi",
    #    "label": r"res2000\_1000, IM/MI",
    #},
    {
        "key": "res2000_1000",
        "folder_suffix": "res2000_1000_nz10",
        "label": r"$\Delta x_\mathrm{f} = 1000$",
    },
    {
        "key": "res2000_500",
        "folder_suffix": "res2000_500_nz10",
        "label": r"$\Delta x_\mathrm{f} = 500$",
    },
]

dt_figure_filename = os.path.join(
    simulation_directory,
    "Ice1r_theta0_theta1_T10100_T10200_L2_error_vs_dt.png",
)

runtime_figure_filename = os.path.join(
    simulation_directory,
    "Ice1r_theta0_theta1_T10100_T10200_L2_error_vs_runtime.png",
)

csv_filename = os.path.join(
    simulation_directory,
    "Ice1r_theta0_theta1_T10100_T10200_L2_error.csv",
)

runtime_log_filename = os.path.join(
    simulation_directory,
    "MISMIP_Ice1r_simulation_times.txt",
)


# ---------------------------------------------------------------------
# Restart-file discovery
# ---------------------------------------------------------------------

def run_directory_name(theta, dt, family):
    """Return the exact folder name for one run.

    theta=0 simulations have an extra ``_time`` suffix on the directory
    name, whereas theta=1 simulations do not.
    """
    theta_suffix = "_time" if np.isclose(theta, 0.0) else ""

    return (
        f"Ice1rr_theta{theta:g}_dt{dt:g}_"
        f"{family['folder_suffix']}{theta_suffix}"
    )


def find_restart_file(T, theta, dt, family):
    folder_name = run_directory_name(theta, dt, family)
    filename = os.path.join(
        simulation_directory,
        folder_name,
        f"restart_t{T:g}.h5",
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            "Restart file not found:\n"
            f"  {filename}\n"
            "Check simulation_directory, T, and the folder names."
        )

    print(
        f"{family['key']}, dt={dt:g}: using restart file\n"
        f"  {filename}"
    )
    return filename


# ---------------------------------------------------------------------
# Load one simulation state
# ---------------------------------------------------------------------

def load_state(T, theta, dt, family):
    filename = find_restart_file(T, theta, dt, family)

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
                name for name in mesh_names
                if "extruded" in name.lower()
            ]

            if extruded:
                mesh_name = extruded[0]
            elif len(mesh_names) == 1:
                mesh_name = mesh_names[0]
            else:
                raise RuntimeError(
                    f"Several meshes found in {filename}: {mesh_names}"
                )

            mesh = afile.load_mesh(
                name=mesh_name,
                reorder=False,
            )

        H = afile.load_function(mesh, "thick")
        zs = afile.load_function(mesh, "zs")

    return mesh, H, zs


# ---------------------------------------------------------------------
# Absolute L2 thickness error
# ---------------------------------------------------------------------

def L2_error(H, Href, zs, h_min=10.0):
    """
    Absolute L2 error in ice thickness:

        ||H - Href||_L2
        = sqrt(integral (H - Href)^2 dA)

    where dA is horizontal plan-view area.

    Locations where Href <= h_min are excluded.
    """

    if H.dat.data_ro.size != Href.dat.data_ro.size:
        raise ValueError(
            "Reference and test thickness fields have different numbers "
            "of degrees of freedom.  Each run family must use a reference "
            "on the same mesh/discretisation."
        )

    Href_on_H = Function(
        H.function_space(),
        name="Href_on_H",
    )
    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    # n_z * ds_t is horizontal plan-view area.
    n_z = 1.0 / sqrt(
        1.0 + zs.dx(0)**2 + zs.dx(1)**2
    )

    error_squared = conditional(
        Href_on_H > h_min,
        (H - Href_on_H)**2,
        0.0,
    )

    integral_error_squared = assemble(
        error_squared * n_z * ds_t
    )

    return float(np.sqrt(integral_error_squared))


# ---------------------------------------------------------------------
# Runtime parsing
# ---------------------------------------------------------------------

def runtime_to_seconds(runtime_text):
    """Convert timedelta-like runtime text to seconds.

    Accepts both forms used in the runtime log, e.g.
    ``7:18:40.037871`` and ``1 day, 4:19:41.400880``.
    """
    text = runtime_text.strip()
    days = 0

    day_match = re.match(
        r"(?P<days>\d+)\s+day(?:s)?,\s*(?P<clock>.*)$",
        text,
    )
    if day_match is not None:
        days = int(day_match.group("days"))
        text = day_match.group("clock")

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Unexpected runtime format: {runtime_text}"
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return (
        86400.0 * days
        + 3600.0 * hours
        + 60.0 * minutes
        + seconds
    )


def seconds_to_runtime_text(total_seconds):
    """Format seconds as H:MM:SS.ssssss for CSV/debug output."""
    hours = int(total_seconds // 3600.0)
    remainder = total_seconds - 3600.0 * hours
    minutes = int(remainder // 60.0)
    seconds = remainder - 60.0 * minutes
    return f"{hours:d}:{minutes:02d}:{seconds:09.6f}"


def identify_family_from_runtime_line(line, theta_value, dt_value):
    """Associate one runtime-log line with a plotted run family.

    The runtime log does not contain the full simulation directory name.
    In the supplied data, theta=0 runs live in ``res2000_1000..._time``
    directories even though the runtime log records ``resolution=500``.
    Therefore theta=0 needs an explicit override.
    """

    # If a future log contains the actual folder text, prefer that because it
    # is unambiguous.
    for family in run_families:
        full_folder = run_directory_name(theta_value, dt_value, family)
        if (
            family["folder_suffix"] in line
            or full_folder in line
        ):
            return family["key"]

    resolution_match = re.search(
        r"resolution=(?P<resolution>[^,\s]+)",
        line,
    )
    if resolution_match is None:
        return None

    token = resolution_match.group("resolution")

    # IMPORTANT: in the supplied runtime file, theta=0 is logged as
    # resolution=500, but those simulations are stored in the
    # res2000_1000_nz10_time directories.
    if np.isclose(theta_value, 0.0) and token in {"500", "2000_500"}:
        return "res2000_1000"

    # theta=1 follows the ordinary mapping used by the folder families.
    if token in {"500", "2000_500"}:
        return "res2000_500"

    if token in {"1000", "2000_1000"}:
        if "time_stepping_im_mi" in line:
            return "res2000_1000_im_mi"
        return "res2000_1000"

    return None


def load_runtime_records(filename):
    """
    Read runtime records when they can be matched to a run family.

    If the log is absent, or if a line cannot distinguish the run family,
    the L2-vs-dt figure is still produced; only the runtime point is omitted.
    """

    if not os.path.exists(filename):
        print(
            "Warning: runtime log not found; the runtime figure will be "
            f"skipped unless runtime data are available:\n  {filename}"
        )
        return {}

    pattern = re.compile(
        r"dt=(?P<dt>[-+0-9.eE]+),\s*"
        r"theta_out=(?P<theta>[-+0-9.eE]+),\s*"
        r".*?"
        r"T_end=(?P<T_end>[-+0-9.eE]+),.*?"
        r"runtime=(?P<runtime>(?:\d+\s+day(?:s)?,\s*)?"
        r"\d+:\d+:\d+(?:\.\d+)?)"
    )

    records = {}

    with open(filename, "r", encoding="utf-8") as runtime_file:
        for line_number, line in enumerate(runtime_file, start=1):
            stripped = line.strip()
            match = pattern.search(stripped)
            if match is None:
                continue

            dt_value = float(match.group("dt"))
            theta_value = float(match.group("theta"))
            T_end = float(match.group("T_end"))
            runtime_text = match.group("runtime")

            if not any(np.isclose(theta_value, value) for value in theta_values):
                continue
            if not any(np.isclose(T_end, T_value) for T_value in times):
                continue

            family_key = identify_family_from_runtime_line(
                stripped,
                theta_value,
                dt_value,
            )

            if family_key is None:
                print(
                    f"Warning: runtime-log line {line_number} could not "
                    "be assigned to one of the screenshot run families; "
                    "skipping it."
                )
                continue

            records[(float(theta_value), family_key, float(T_end), float(dt_value))] = {
                "runtime_text": runtime_text,
                "runtime_seconds": runtime_to_seconds(runtime_text),
                "line_number": line_number,
            }

    print(
        f"Read {len(records)} matching runtime records from {filename}"
    )
    return records


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(simulation_directory, exist_ok=True)

    runtime_records = load_runtime_records(
        runtime_log_filename
    )

    all_rows = []

    # -------------------------------------------------------------
    # Compute L2 error separately for each theta, run family, and time.
    # Each (theta, family, T) combination gets its own dt=0.1 reference.
    # -------------------------------------------------------------

    for theta in theta_values:
        print("\n" + "#" * 78)
        print(f"Processing theta={theta:g}")
        print("#" * 78)

        for family in run_families:
            print("\n" + "=" * 78)
            print(f"Processing theta={theta:g}, {family['key']}")
            print("=" * 78)

            for T in times:
                print("\n" + "-" * 78)
                print(f"theta={theta:g}, T = {T:g} a")
                print("-" * 78)

                try:
                    _, Href, _ = load_state(T, theta, ref_dt, family)
                except (FileNotFoundError, RuntimeError) as exc:
                    print(f"Warning: {exc}")
                    print(
                        f"Skipping theta={theta:g}, {family['key']} at T={T:g} "
                        f"because its dt={ref_dt:g} reference is unavailable."
                    )
                    print(
                        "Expected reference path:\n"
                        f"  {os.path.join(simulation_directory, run_directory_name(theta, ref_dt, family), f'restart_t{T:g}.h5')}"
                    )
                    continue

                print(
                    f"\nReference solution: theta={theta:g}, {family['key']}, "
                    f"T={T:g} a, dt={ref_dt:g} a"
                )
                print(
                    "Reference thickness range: "
                    f"{np.min(Href.dat.data_ro):.3f} to "
                    f"{np.max(Href.dat.data_ro):.3f} m\n"
                )

                for dt in dts:
                    if np.isclose(dt, ref_dt):
                        # This directory is still used: it supplies Href.
                        continue

                    try:
                        _, H, zs = load_state(T, theta, dt, family)
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(f"Warning: {exc}")
                        continue

                    error = L2_error(
                        H,
                        Href,
                        zs,
                        h_min=H_min,
                    )

                    runtime_record = runtime_records.get(
                        (float(theta), family["key"], float(T), float(dt))
                    )

                    # Runtime shown in Figure 2 is the cost of the interval
                    # ending at T, not always the cumulative cost from 10000.
                    #
                    #   T=10100 -> runtime(10000 -> 10100)
                    #   T=10200 -> runtime(10000 -> 10200)
                    #              - runtime(10000 -> 10100)
                    #            = runtime(10100 -> 10200)
                    if runtime_record is None:
                        runtime_seconds = None
                        runtime_text = ""
                    elif np.isclose(T, 10200.0):
                        previous_runtime_record = runtime_records.get(
                            (
                                float(theta),
                                family["key"],
                                10100.0,
                                float(dt),
                            )
                        )

                        if previous_runtime_record is None:
                            print(
                                f"Warning: cannot form interval runtime for "
                                f"theta={theta:g}, {family['key']}, dt={dt:g}: "
                                "T=10100 runtime is missing."
                            )
                            runtime_seconds = None
                            runtime_text = ""
                        else:
                            runtime_seconds = (
                                runtime_record["runtime_seconds"]
                                - previous_runtime_record["runtime_seconds"]
                            )

                            if runtime_seconds <= 0.0:
                                print(
                                    f"Warning: non-positive interval runtime for "
                                    f"theta={theta:g}, {family['key']}, dt={dt:g}: "
                                    f"{runtime_seconds:.6f} s. Skipping runtime point."
                                )
                                runtime_seconds = None
                                runtime_text = ""
                            else:
                                runtime_text = seconds_to_runtime_text(
                                    runtime_seconds
                                )
                    else:
                        runtime_seconds = runtime_record["runtime_seconds"]
                        runtime_text = runtime_record["runtime_text"]

                    all_rows.append({
                        "theta": float(theta),
                        "T": float(T),
                        "family_key": family["key"],
                        "family_label": family["label"],
                        "folder_suffix": family["folder_suffix"],
                        "dt": float(dt),
                        "L2_error": error,
                        "runtime_seconds": runtime_seconds,
                        "runtime_text": runtime_text,
                    })

                    print(
                        f"theta={theta:g}, {family['key']}, T={T:g} a, "
                        f"dt={dt:g} a: L2 error={error:.8e}, "
                        f"runtime={runtime_text if runtime_text else 'not found'}"
                    )

    if not all_rows:
        raise RuntimeError(
            "No non-reference screenshot experiments were successfully loaded."
        )

    family_order = {
        family["key"]: i
        for i, family in enumerate(run_families)
    }

    all_rows.sort(
        key=lambda row: (
            family_order[row["family_key"]],
            row["theta"],
            row["T"],
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
            "run_family",
            "folder_suffix",
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
                "T": f"{row['T']:g}",
                "theta": f"{row['theta']:g}",
                "run_family": row["family_key"],
                "folder_suffix": row["folder_suffix"],
                "reference_dt": f"{ref_dt:g}",
                "dt": f"{row['dt']:g}",
                "H_min_m": f"{H_min:g}",
                "L2_error": f"{row['L2_error']:.10e}",
                "runtime_seconds": (
                    ""
                    if row["runtime_seconds"] is None
                    else f"{row['runtime_seconds']:.9f}"
                ),
                "runtime_text": row["runtime_text"],
            })

    print(f"\nSaved table to:\n  {csv_filename}")

    print("\nSUMMARY OF SUCCESSFULLY LOADED L2 RESULTS")
    print("-" * 72)
    for theta in theta_values:
        for T in times:
            for family in run_families:
                rows_here = [
                    row for row in all_rows
                    if (
                        np.isclose(row["theta"], theta)
                        and np.isclose(row["T"], T)
                        and row["family_key"] == family["key"]
                    )
                ]

                if not rows_here:
                    print(
                        f"theta={theta:g}, T={T:g}, {family['key']}: "
                        "NO SUCCESSFULLY LOADED RESULTS"
                    )
                    continue

                rows_here = sorted(
                    rows_here,
                    key=lambda row: row["dt"],
                )

                print(
                    f"theta={theta:g}, T={T:g}, {family['key']}: "
                    f"{len(rows_here)} points"
                )

                for row in rows_here:
                    print(
                        f"    dt={row['dt']:g}, "
                        f"L2_error={row['L2_error']:.8e}"
                    )

    # -------------------------------------------------------------
    # Figure 1: L2 error vs timestep.
    # One panel per output time T; one curve per run family and theta.
    # -------------------------------------------------------------

    n_times = len(times)
    fig, axes = plt.subplots(
        1,
        n_times,
        figsize=(6.0 * n_times, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]

    panel_labels = [
        f"({chr(ord('a') + i)})"
        for i in range(n_times)
    ]

    for panel, T in enumerate(times):
        ax = axes[panel]
        any_time_data = False

        for family_index, family in enumerate(run_families):
            for theta in theta_values:
                family_rows = [
                    row for row in all_rows
                    if (
                        row["family_key"] == family["key"]
                        and np.isclose(row["theta"], theta)
                        and np.isclose(row["T"], T)
                    )
                ]

                if not family_rows:
                    continue

                plot_dts = np.asarray(
                    [row["dt"] for row in family_rows],
                    dtype=float,
                )
                plot_errors = np.asarray(
                    [row["L2_error"] for row in family_rows],
                    dtype=float,
                )

                print(
                    f"PLOT DATA: theta={theta:g}, T={T:g}, "
                    f"family={family['key']}"
                )
                print("  dt     =", plot_dts)
                print("  errors =", plot_errors)

                valid = (
                    np.isfinite(plot_dts)
                    & np.isfinite(plot_errors)
                    & (plot_dts > 0.0)
                    & (plot_errors > 0.0)
                )

                if not np.all(valid):
                    print(
                        f"Warning: removing invalid log-plot values for "
                        f"theta={theta:g}, T={T:g}, {family['key']}:"
                    )

                    for dt_value, error_value, is_valid in zip(
                        plot_dts,
                        plot_errors,
                        valid,
                    ):
                        if not is_valid:
                            print(
                                f"    dt={dt_value:g}, "
                                f"L2_error={error_value}"
                            )

                plot_dts = plot_dts[valid]
                plot_errors = plot_errors[valid]

                if plot_dts.size == 0:
                    print(
                        f"WARNING: no plottable points for theta={theta:g}, "
                        f"T={T:g}, {family['key']}"
                    )
                    continue

                any_time_data = True
                order = np.argsort(plot_dts)

                ax.loglog(
                    plot_dts[order],
                    plot_errors[order],
                    color=theta_colors[theta],
                    linestyle=theta_styles[theta],
                    marker=family_markers.get(family["key"], "o"),
                    linewidth=1.7,
                    markersize=5,
                    label=rf"{family['label']}, $\theta={theta:g}$",
                )

        if not any_time_data:
            ax.set_visible(False)
            continue

        ax.set_xlabel(r"$\Delta t$ [a]", fontsize=16)
        ax.set_title(rf"$T={T:g}$ a", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=12)

        ax.text(
            0.04,
            0.95,
            panel_labels[panel],
            transform=ax.transAxes,
            fontsize=16,
            va="top",
            ha="left",
        )

    # Put the shared y-label on the first visible panel.
    for ax in axes:
        if ax.get_visible():
            ax.set_ylabel(
                r"$L_2$ error in ice thickness",
                fontsize=16,
            )
            break

    fig.savefig(
        dt_figure_filename,
        dpi=figure_dpi,
        bbox_inches="tight",
    )
    print(
        "Saved timestep-error figure to:\n"
        f"  {dt_figure_filename}"
    )

    # -------------------------------------------------------------
    # Figure 2: L2 error vs interval wall-clock runtime.
    # T=10100 uses 10000->10100; T=10200 uses 10100->10200.
    # One panel per output time T; one curve per run family and theta.
    # -------------------------------------------------------------

    fig_runtime, runtime_axes = plt.subplots(
        1,
        n_times,
        figsize=(6.0 * n_times, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    runtime_axes = runtime_axes[0]

    any_runtime_data = False

    for panel, T in enumerate(times):
        ax = runtime_axes[panel]
        any_time_runtime = False

        for family_index, family in enumerate(run_families):
            for theta in theta_values:
                family_rows = [
                    row for row in all_rows
                    if (
                        row["family_key"] == family["key"]
                        and np.isclose(row["theta"], theta)
                        and np.isclose(row["T"], T)
                        and row["runtime_seconds"] is not None
                        and row["runtime_seconds"] > 0.0
                        and row["L2_error"] > 0.0
                    )
                ]

                if not family_rows:
                    continue

                any_runtime_data = True
                any_time_runtime = True

                runtimes = np.asarray(
                    [row["runtime_seconds"] for row in family_rows],
                    dtype=float,
                )
                runtime_errors = np.asarray(
                    [row["L2_error"] for row in family_rows],
                    dtype=float,
                )
                runtime_dts = np.asarray(
                    [row["dt"] for row in family_rows],
                    dtype=float,
                )

                order = np.argsort(runtimes)

                ax.loglog(
                    runtimes[order],
                    runtime_errors[order],
                    color=theta_colors[theta],
                    linestyle=theta_styles[theta],
                    marker=family_markers.get(family["key"], "o"),
                    linewidth=1.7,
                    markersize=5,
                    label=rf"{family['label']}, $\theta={theta:g}$",
                )

                for runtime_value, error_value, dt_value in zip(
                    runtimes,
                    runtime_errors,
                    runtime_dts,
                ):
                    ax.annotate(
                        rf"$\Delta t={dt_value:g}$",
                        (runtime_value, error_value),
                        xytext=(6, 5),
                        textcoords="offset points",
                        fontsize=10,
                    )

        if not any_time_runtime:
            ax.set_visible(False)
            continue

        ax.set_xlabel("Wall-clock runtime for interval [s]", fontsize=16)
        ax.set_title(rf"$T={T:g}$ a", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=12)

        ax.text(
            0.04,
            0.95,
            panel_labels[panel],
            transform=ax.transAxes,
            fontsize=16,
            va="top",
            ha="left",
        )

    if any_runtime_data:
        # Put the shared y-label on the first visible runtime panel.
        for ax in runtime_axes:
            if ax.get_visible():
                ax.set_ylabel(
                    r"$L_2$ error in ice thickness",
                    fontsize=16,
                )
                break

        fig_runtime.savefig(
            runtime_figure_filename,
            dpi=figure_dpi,
            bbox_inches="tight",
        )
        print(
            "Saved L2-error/runtime figure to:\n"
            f"  {runtime_figure_filename}"
        )
    else:
        plt.close(fig_runtime)
        print(
            "Warning: no runtime records could be matched to the screenshot "
            "run families and requested output times, so the runtime figure "
            "was not made."
        )

    plt.show()


if __name__ == "__main__":
    main()
