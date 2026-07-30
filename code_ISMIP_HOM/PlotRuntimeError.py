from firedrake import *
from firedrake import CheckpointFile

import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

T = 100

dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_values = [0, 0.5, 1]

# Reference remains theta = 0, dt = 0.01
ref_dt = 0.01
ref_theta = 0

resolution_pairs = [
    (100, 10),
    (200, 20),
    (400, 40),
    (800, 80),
    (1600, 160),
]

# Change this path if the runtime log is stored elsewhere.
runtime_log_filename = "Simulations/simulation_times.txt"

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

theta_colors = {
    0: plt.cm.viridis(0.1),
    0.5: plt.cm.viridis(0.45),
    1: plt.cm.viridis(0.85),
}

theta_styles = {
    0: "-",
    0.5: ":",
    1: "--",
}

theta_markers = {
    0: "^",
    0.5: "o",
    1: "s",
}

dt_offsets = {
    0.1: (8, 6),
    1: (6, 6),
    2: (6, 7),
    5: (8, 0),
    10: (-0.2, 8),
    20: (-2, -8),
    50: (6, 0),
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
            "Check that the corresponding simulation was completed."
        )

    with CheckpointFile(filename, "r") as afile:
        try:
            mesh = afile.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")

        except Exception:
            mesh = afile.load_mesh()
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")

    return mesh, H, zs


# ---------------------------------------------------------------------
# L2 error
# ---------------------------------------------------------------------

def L2_error(H, Href, zs):
    """
    Horizontal-plane L2 error in ice thickness.

    Assumes H and Href use the same spatial mesh, function space,
    and degree-of-freedom ordering.
    """

    Href_on_H = Function(
        H.function_space(),
        name="Href_on_H",
    )

    if Href.dat.data_ro.size != H.dat.data_ro.size:
        raise ValueError(
            "H and Href have different numbers of degrees of freedom."
        )

    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    n_z = 1.0 / sqrt(
        1.0 + zs.dx(0) ** 2
    )

    error_squared = assemble(
        (H - Href_on_H) ** 2 * n_z * ds_t
    )

    return float(sqrt(error_squared))


# ---------------------------------------------------------------------
# Runtime-log parsing
# ---------------------------------------------------------------------

def runtime_to_seconds(runtime_text):
    """
    Convert a runtime such as

        0:04:25.218533
        1:42:55.578311

    into seconds.
    """

    parts = runtime_text.strip().split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected runtime format: {runtime_text!r}"
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return 3600.0 * hours + 60.0 * minutes + seconds


def load_runtime_records(filename):
    """
    Parse the runtime log.

    Records are keyed by

        (dt, theta, T, nx, nz)

    If a duplicate key occurs, assignment into the dictionary
    overwrites the earlier record. Consequently, the last occurrence
    in the text file is retained.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Runtime log not found: {filename}"
        )

    pattern = re.compile(
        r"dt=(?P<dt>[-+0-9.eE]+),\s*"
        r"theta_out=(?P<theta>[-+0-9.eE]+),\s*"
        r"nx=(?P<nx>\d+),\s*"
        r"nz=(?P<nz>\d+),\s*"
        r"T=(?P<T>[-+0-9.eE]+),.*?"
        r"runtime=(?P<runtime>\d+:\d+:\d+(?:\.\d+)?)"
    )

    runtime_records = {}
    duplicate_counts = {}
    skipped_lines = []

    with open(filename, "r", encoding="utf-8") as runtime_file:
        for line_number, line in enumerate(runtime_file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            match = pattern.search(stripped)

            if match is None:
                skipped_lines.append(line_number)
                continue

            dt = float(match.group("dt"))
            theta = float(match.group("theta"))
            nx = int(match.group("nx"))
            nz = int(match.group("nz"))
            run_T = float(match.group("T"))

            runtime_text = match.group("runtime")
            runtime_seconds = runtime_to_seconds(runtime_text)

            key = (
                dt,
                theta,
                run_T,
                nx,
                nz,
            )

            if key in runtime_records:
                duplicate_counts[key] = duplicate_counts.get(key, 1) + 1

            # This deliberately replaces any earlier duplicate.
            runtime_records[key] = {
                "dt": dt,
                "theta": theta,
                "T": run_T,
                "nx": nx,
                "nz": nz,
                "runtime_text": runtime_text,
                "runtime_seconds": runtime_seconds,
                "line_number": line_number,
            }

    print(
        f"Read {len(runtime_records)} unique runtime records "
        f"from {filename}."
    )

    if duplicate_counts:
        print(
            f"Found {len(duplicate_counts)} duplicated configurations. "
            "The last occurrence of each was retained."
        )

        for key, count in sorted(duplicate_counts.items()):
            dt, theta, run_T, nx, nz = key

            retained = runtime_records[key]

            print(
                "  Duplicate: "
                f"dt={dt:g}, theta={theta:g}, T={run_T:g}, "
                f"nx={nx}, nz={nz}; "
                f"{count} occurrences, retained line "
                f"{retained['line_number']}."
            )

    if skipped_lines:
        print(
            "Warning: could not parse runtime-log lines: "
            + ", ".join(map(str, skipped_lines))
        )

    return runtime_records


def runtime_key(dt, theta, run_T, nx, nz):
    """
    Construct keys consistently for runtime lookup.
    """

    return (
        float(dt),
        float(theta),
        float(run_T),
        int(nx),
        int(nz),
    )


# ---------------------------------------------------------------------
# Load runtime information
# ---------------------------------------------------------------------

runtime_records = load_runtime_records(
    runtime_log_filename
)


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
    figsize=(4.8 * ncols, 4.0 * nrows),
    sharex=False,
    sharey=True,
    squeeze=False,
)

axes_flat = axes.ravel()

# Store matched numerical results for CSV output.
results = []


# ---------------------------------------------------------------------
# Compute errors, match runtimes, and plot
# ---------------------------------------------------------------------

for panel_index, (nx, nz) in enumerate(resolution_pairs):
    ax = axes_flat[panel_index]

    # Reference on the same spatial mesh.
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

    for theta in theta_values:
        runtimes = []
        errors = []
        matched_dts = []

        for dt in dts:
            key = runtime_key(
                dt,
                theta,
                T,
                nx,
                nz,
            )

            runtime_record = runtime_records.get(key)

            if runtime_record is None:
                print(
                    "Warning: no matching runtime for "
                    f"dt={dt:g}, theta={theta:g}, "
                    f"T={T:g}, nx={nx}, nz={nz}."
                )
                continue

            try:
                _, H, zs = load_state(
                    dt,
                    theta,
                    nx,
                    nz,
                )

            except FileNotFoundError as error:
                print(f"Warning: {error}")
                continue

            try:
                error_value = L2_error(
                    H,
                    Href,
                    zs,
                )

            except (ValueError, RuntimeError) as error:
                print(
                    "Warning: error calculation failed for "
                    f"dt={dt:g}, theta={theta:g}, "
                    f"nx={nx}, nz={nz}: {error}"
                )
                continue

            runtime_seconds = runtime_record["runtime_seconds"]

            # Logarithmic axes require positive values.
            if runtime_seconds <= 0 or error_value <= 0:
                print(
                    "Warning: skipping non-positive runtime/error for "
                    f"dt={dt:g}, theta={theta:g}, "
                    f"nx={nx}, nz={nz}."
                )
                continue

            runtimes.append(runtime_seconds)
            errors.append(error_value)
            matched_dts.append(dt)

            results.append(
                {
                    "T": T,
                    "nx": nx,
                    "nz": nz,
                    "theta": theta,
                    "dt": dt,
                    "runtime_seconds": runtime_seconds,
                    "runtime_text": runtime_record["runtime_text"],
                    "L2_error": error_value,
                    "runtime_log_line": runtime_record["line_number"],
                }
            )

        if not runtimes:
            continue

        runtimes = np.asarray(
            runtimes,
            dtype=float,
        )

        errors = np.asarray(
            errors,
            dtype=float,
        )

        matched_dts = np.asarray(
            matched_dts,
            dtype=float,
        )

        order = np.argsort(matched_dts)

        runtimes = runtimes[order]
        errors = errors[order]
        matched_dts = matched_dts[order]

        runtimes = runtimes[order]
        errors = errors[order]
        matched_dts = matched_dts[order]

        ax.loglog(
            runtimes,
            errors,
            color=theta_colors[theta],
            linestyle=theta_styles[theta],
            marker="o",
            linewidth=1.7,
            markersize=6,
            label=fr"$\theta={theta:g}$",
        )

        # Add the timestep beside each point.
        if theta == 0:
            for runtime_seconds, error_value, dt in zip(
                runtimes,
                errors,
                matched_dts,
            ):
                dx, dy = dt_offsets.get(dt, (6, 6))

                ha = "left" if dx >= 0 else "right"
                va = "bottom" if dy >= 0 else "top"

                ax.annotate(
                    fr"$\Delta t={dt:g}$",
                    xy=(runtime_seconds, error_value),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=10,
                    ha=ha,
                    va=va,
                    annotation_clip=True,
                )

    ax.set_title(
        fr"$n_x={nx},\ n_z={nz}$",
        fontsize=14,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        fontsize=9,
    )


# Hide unused panels.
for ax in axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)


# ---------------------------------------------------------------------
# Shared labels
# ---------------------------------------------------------------------

fig.supxlabel(
    "Wall clock time [s]",
    fontsize=16,
)

fig.supylabel(
    r"$L_2$ error in ice thickness",
    fontsize=16,
)

plt.tight_layout()


# ---------------------------------------------------------------------
# Save the figure
# ---------------------------------------------------------------------

os.makedirs(
    "Simulations",
    exist_ok=True,
)

figure_filename = (
    f"Simulations/runtime_vs_thickness_error_T{T:g}.png"
)

plt.savefig(
    figure_filename,
    dpi=700,
    bbox_inches="tight",
)

print(f"Saved figure to: {figure_filename}")


# ---------------------------------------------------------------------
# Save matched runtime/error data
# ---------------------------------------------------------------------

results.sort(
    key=lambda row: (
        row["nx"],
        row["nz"],
        row["theta"],
        row["runtime_seconds"],
    )
)

csv_filename = (
    f"Simulations/runtime_vs_thickness_error_T{T:g}.csv"
)

with open(
    csv_filename,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    fieldnames = [
        "T",
        "nx",
        "nz",
        "theta",
        "dt",
        "runtime_seconds",
        "runtime_text",
        "L2_error",
        "runtime_log_line",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in results:
        writer.writerow(
            {
                "T": f'{row["T"]:g}',
                "nx": row["nx"],
                "nz": row["nz"],
                "theta": f'{row["theta"]:g}',
                "dt": f'{row["dt"]:g}',
                "runtime_seconds": (
                    f'{row["runtime_seconds"]:.9f}'
                ),
                "runtime_text": row["runtime_text"],
                "L2_error": f'{row["L2_error"]:.12e}',
                "runtime_log_line": row["runtime_log_line"],
            }
        )

print(f"Saved runtime/error table to: {csv_filename}")

plt.show()
