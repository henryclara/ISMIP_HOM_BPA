from firedrake import *
from firedrake import CheckpointFile
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import re

T = 100
dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_values = [0, 0.5, 1]

ref_dt = 0.01
ref_theta = 0
H_min = 1.0  # Ignore reference thickness below 1 m

resolution_pairs = [
    (100, 10),
    (200, 20),
    (400, 40),
    (800, 80),
    (1600, 160),
]

# Change this if the runtime log is stored elsewhere.
runtime_log_filename = "Simulations/simulation_times.txt"

error_rows = []

theta_colors = {
    0: plt.cm.viridis(0.1),
    0.5: plt.cm.viridis(0.4),
    1: plt.cm.viridis(0.8),
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


def load_state(dt, theta, nx, nz):
    filename = (
        f"Simulations/BPA_output_dt{dt:g}_theta{theta:g}_nx{nx}_nz{nz}"
        f"/restart_t{T:g}.h5"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Restart file not found: {filename}")

    with CheckpointFile(filename, "r") as afile:
        try:
            mesh = afile.load_mesh(
                name="firedrake_default_extruded", reorder=False
            )
        except KeyError:
            mesh_names = list(
                afile._get_mesh_name_topology_name_map().keys()
            )

            if not mesh_names:
                raise RuntimeError(
                    f"No mesh found in {filename}; checkpoint may be incomplete."
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

            mesh = afile.load_mesh(name=mesh_name, reorder=False)

        H = afile.load_function(mesh, "thick")
        zs = afile.load_function(mesh, "zs")

    return mesh, H, zs


def mean_absolute_percentage_error(H, Href, zs, h_min=1.0):
    """
    Mean absolute percentage error in thickness.

    Points where the reference thickness is <= h_min are excluded.
    """

    Href_on_H = Function(H.function_space(), name="Href_on_H")
    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    n_z = 1.0 / sqrt(1.0 + zs.dx(0)**2)
    valid = conditional(Href_on_H > h_min, 1.0, 0.0)

    percentage_error = conditional(
        Href_on_H > h_min,
        sqrt((H - Href_on_H)**2) / Href_on_H,
        0.0,
    )

    integrated_error = assemble(percentage_error * n_z * ds_t)
    valid_length = assemble(valid * n_z * ds_t)

    if valid_length <= 0:
        raise ValueError("No reference thickness values exceed H_min.")

    return float(100.0 * integrated_error / valid_length)


def latex_number(x):
    if x == 0:
        return "$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent
    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


# ---------------------------------------------------------------------
# Runtime-log parsing
# ---------------------------------------------------------------------

def runtime_to_seconds(runtime_text):
    """Convert an H:MM:SS[.ffffff] runtime string to seconds."""

    parts = runtime_text.strip().split(":")

    if len(parts) != 3:
        raise ValueError(
            f"Unexpected runtime format: {runtime_text!r}"
        )

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return 3600.0 * hours + 60.0 * minutes + seconds


def seconds_to_runtime_text(runtime_seconds):
    """Format seconds as HH:MM:SS.sss for the LaTeX runtime tables."""

    if runtime_seconds < 0:
        raise ValueError("Runtime cannot be negative.")

    hours = int(runtime_seconds // 3600)
    remainder = runtime_seconds - 3600 * hours
    minutes = int(remainder // 60)
    seconds = remainder - 60 * minutes

    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def runtime_key(dt, theta, run_T, nx, nz):
    """Construct runtime-record lookup keys consistently."""

    return (
        float(dt),
        float(theta),
        float(run_T),
        int(nx),
        int(nz),
    )


def load_runtime_records(filename):
    """
    Parse the simulation runtime log.

    Expected fields on each relevant line are, in this order,

        dt=..., theta_out=..., nx=..., nz=..., T=..., runtime=H:MM:SS...

    ``theta=...`` is also accepted. If a configuration occurs more than
    once, the final occurrence in the file is retained.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Runtime log not found: {filename}"
        )

    pattern = re.compile(
        r"dt=(?P<dt>[-+0-9.eE]+),\s*"
        r"theta(?:_out)?=(?P<theta>[-+0-9.eE]+),\s*"
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

            key = runtime_key(dt, theta, run_T, nx, nz)

            if key in runtime_records:
                duplicate_counts[key] = duplicate_counts.get(key, 1) + 1

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

    if skipped_lines:
        print(
            "Warning: could not parse runtime-log lines: "
            + ", ".join(map(str, skipped_lines))
        )

    return runtime_records


runtime_records = load_runtime_records(runtime_log_filename)


# ---------------------------------------------------------------------
# Compute and plot MAPE
# ---------------------------------------------------------------------

ncols = 3
nrows = int(np.ceil(len(resolution_pairs) / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    sharex=True, sharey=True, squeeze=False,
)

axes_flat = axes.ravel()

for panel_index, (nx, nz) in enumerate(resolution_pairs):
    ax = axes_flat[panel_index]

    try:
        _, Href, _ = load_state(ref_dt, ref_theta, nx, nz)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Warning: {e}")
        ax.set_visible(False)
        continue

    for theta in theta_values:
        plot_dts = []
        errors = []

        for dt in dts:
            if dt == ref_dt and theta == ref_theta:
                continue

            try:
                _, H, zs = load_state(dt, theta, nx, nz)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"Warning: {e}")
                continue

            err = mean_absolute_percentage_error(
                H, Href, zs, h_min=H_min
            )

            runtime_record = runtime_records.get(
                runtime_key(dt, theta, T, nx, nz)
            )

            if runtime_record is None:
                print(
                    "Warning: no matching runtime for "
                    f"dt={dt:g}, theta={theta:g}, "
                    f"T={T:g}, nx={nx}, nz={nz}."
                )

            error_rows.append({
                "T": T,
                "nx": nx,
                "nz": nz,
                "dt": dt,
                "theta": theta,
                "MAPE_percent": err,
                "runtime_seconds": (
                    None if runtime_record is None
                    else runtime_record["runtime_seconds"]
                ),
                "runtime_text": (
                    "" if runtime_record is None
                    else runtime_record["runtime_text"]
                ),
            })

            plot_dts.append(dt)
            errors.append(err)

        plot_dts = np.asarray(plot_dts)
        errors = np.asarray(errors)

        if len(plot_dts) == 0:
            continue

        order = np.argsort(plot_dts)

        ax.loglog(
            plot_dts[order],
            errors[order],
            color=theta_colors[theta],
            linestyle=theta_styles[theta],
            marker="o",
            linewidth=1.7,
            markersize=5,
            label=fr"$\theta={theta:g}$",
        )

    ax.set_title(fr"$n_x={nx},\ n_z={nz}$")
    ax.grid(True, which="both")
    ax.legend(fontsize=9)

for ax in axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)

fig.supxlabel(r"$\Delta t$ [a]")
fig.supylabel(r"Mean absolute percentage error in thickness [\%]")

plt.tight_layout()
os.makedirs("Simulations", exist_ok=True)

figure_filename = (
    "Simulations/thickness_MAPE_vs_dt_by_resolution.png"
)

plt.savefig(figure_filename, dpi=700, bbox_inches="tight")
print(f"Saved figure to: {figure_filename}")


# ---------------------------------------------------------------------
# Plot runtime against timestep size
# ---------------------------------------------------------------------

runtime_fig, runtime_axes = plt.subplots(
    nrows, ncols,
    figsize=(4.8 * ncols, 4.0 * nrows),
    sharex=True, sharey=False, squeeze=False,
)

runtime_axes_flat = runtime_axes.ravel()

for panel_index, (nx, nz) in enumerate(resolution_pairs):
    ax = runtime_axes_flat[panel_index]
    panel_has_data = False

    for theta in theta_values:
        plot_dts = []
        runtimes = []

        for dt in dts:
            runtime_record = runtime_records.get(
                runtime_key(dt, theta, T, nx, nz)
            )

            if runtime_record is None:
                continue

            runtime_seconds = runtime_record["runtime_seconds"]

            # Logarithmic axes require positive values.
            if dt <= 0 or runtime_seconds <= 0:
                print(
                    "Warning: skipping non-positive timestep/runtime for "
                    f"dt={dt:g}, theta={theta:g}, "
                    f"nx={nx}, nz={nz}."
                )
                continue

            plot_dts.append(dt)
            runtimes.append(runtime_seconds)

        if not plot_dts:
            continue

        plot_dts = np.asarray(plot_dts, dtype=float)
        runtimes = np.asarray(runtimes, dtype=float)
        order = np.argsort(plot_dts)

        ax.loglog(
            plot_dts[order],
            runtimes[order],
            color=theta_colors[theta],
            linestyle=theta_styles[theta],
            marker="o",
            linewidth=1.7,
            markersize=6,
            label=fr"$\theta={theta:g}$",
        )

        panel_has_data = True

    if not panel_has_data:
        ax.set_visible(False)
        continue

    ax.set_title(fr"$n_x={nx},\ n_z={nz}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

for ax in runtime_axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)

runtime_fig.supxlabel(r"$\Delta t$ [a]")
runtime_fig.supylabel("Wall-clock runtime [s]")
runtime_fig.tight_layout()

runtime_figure_filename = (
    f"Simulations/runtime_vs_dt_by_resolution_T{T:g}.png"
)

runtime_fig.savefig(
    runtime_figure_filename,
    dpi=700,
    bbox_inches="tight",
)

print(f"Saved runtime/timestep figure to: {runtime_figure_filename}")


# ---------------------------------------------------------------------
# Save the combined MAPE/runtime CSV table
# ---------------------------------------------------------------------

error_rows.sort(
    key=lambda row: (
        row["nx"], row["nz"], row["dt"], row["theta"]
    )
)

csv_filename = f"Simulations/thickness_MAPE_T{T:g}.csv"

with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
    fieldnames = [
        "T", "nx", "nz", "dt", "theta",
        "H_min_m", "MAPE_percent",
        "runtime_seconds", "runtime_text",
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in error_rows:
        runtime_seconds = row["runtime_seconds"]

        writer.writerow({
            "T": f'{row["T"]:g}',
            "nx": row["nx"],
            "nz": row["nz"],
            "dt": f'{row["dt"]:g}',
            "theta": f'{row["theta"]:g}',
            "H_min_m": f"{H_min:g}",
            "MAPE_percent": f'{row["MAPE_percent"]:.8e}',
            "runtime_seconds": (
                "" if runtime_seconds is None
                else f"{runtime_seconds:.9f}"
            ),
            "runtime_text": row["runtime_text"],
        })

print(f"Saved MAPE/runtime CSV table to: {csv_filename}")


# ---------------------------------------------------------------------
# Save MAPE LaTeX tables
# ---------------------------------------------------------------------

latex_filename = f"Simulations/thickness_MAPE_T{T:g}.tex"

with open(latex_filename, "w", encoding="utf-8") as tex:
    tex.write("% Automatically generated\n")
    tex.write("% Requires \\usepackage{booktabs}\n\n")

    for nx, nz in resolution_pairs:
        lookup = {
            (row["dt"], row["theta"]): row["MAPE_percent"]
            for row in error_rows
            if row["nx"] == nx and row["nz"] == nz
        }

        tex.write("\\begin{table}[htbp]\n")
        tex.write("\\centering\n")
        tex.write(
            f"\\caption{{Mean absolute percentage error in ice "
            f"thickness at $T={T:g}$ years for $n_x={nx}$ and "
            f"$n_z={nz}$. Locations with "
            f"$H_{{\\mathrm{{ref}}}}\\leq {H_min:g}$ m are excluded.}}\n"
        )
        tex.write(f"\\label{{tab:mape_nx{nx}_nz{nz}}}\n")
        tex.write("\\begin{tabular}{lccc}\n")
        tex.write("\\toprule\n")
        tex.write(
            "$\\Delta t$ [a] & "
            "$\\theta=0$ & "
            "$\\theta=0.5$ & "
            "$\\theta=1$ \\\\\n"
        )
        tex.write("\\midrule\n")

        for dt in dts:
            table_row = [f"{dt:g}"]

            for theta in theta_values:
                value = lookup.get((dt, theta))
                table_row.append(
                    "--" if value is None else latex_number(value)
                )

            tex.write(" & ".join(table_row) + " \\\\\n")

        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n\n")

print(f"Saved MAPE LaTeX tables to: {latex_filename}")


# ---------------------------------------------------------------------
# Save dedicated runtime tables
# ---------------------------------------------------------------------

runtime_rows = []

for nx, nz in resolution_pairs:
    for dt in dts:
        for theta in theta_values:
            record = runtime_records.get(
                runtime_key(dt, theta, T, nx, nz)
            )

            if record is None:
                continue

            runtime_rows.append({
                "T": T,
                "nx": nx,
                "nz": nz,
                "dt": dt,
                "theta": theta,
                "runtime_seconds": record["runtime_seconds"],
                "runtime_text": record["runtime_text"],
                "runtime_log_line": record["line_number"],
            })

runtime_rows.sort(
    key=lambda row: (
        row["nx"], row["nz"], row["dt"], row["theta"]
    )
)

runtime_csv_filename = f"Simulations/simulation_runtimes_T{T:g}.csv"

with open(
    runtime_csv_filename,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:
    fieldnames = [
        "T", "nx", "nz", "dt", "theta",
        "runtime_seconds", "runtime_text", "runtime_log_line",
    ]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in runtime_rows:
        writer.writerow({
            "T": f'{row["T"]:g}',
            "nx": row["nx"],
            "nz": row["nz"],
            "dt": f'{row["dt"]:g}',
            "theta": f'{row["theta"]:g}',
            "runtime_seconds": f'{row["runtime_seconds"]:.9f}',
            "runtime_text": row["runtime_text"],
            "runtime_log_line": row["runtime_log_line"],
        })

print(f"Saved runtime CSV table to: {runtime_csv_filename}")

runtime_latex_filename = f"Simulations/simulation_runtimes_T{T:g}.tex"

with open(runtime_latex_filename, "w", encoding="utf-8") as tex:
    tex.write("% Automatically generated\n")
    tex.write("% Requires \\usepackage{booktabs}\n\n")

    for nx, nz in resolution_pairs:
        lookup = {
            (row["dt"], row["theta"]): row["runtime_seconds"]
            for row in runtime_rows
            if row["nx"] == nx and row["nz"] == nz
        }

        tex.write("\\begin{table}[htbp]\n")
        tex.write("\\centering\n")
        tex.write(
            f"\\caption{{Wall-clock runtime at $T={T:g}$ years "
            f"for $n_x={nx}$ and $n_z={nz}$. Runtimes are shown "
            f"as hours:minutes:seconds.}}\n"
        )
        tex.write(f"\\label{{tab:runtime_nx{nx}_nz{nz}}}\n")
        tex.write("\\begin{tabular}{lccc}\n")
        tex.write("\\toprule\n")
        tex.write(
            "$\\Delta t$ [a] & "
            "$\\theta=0$ & "
            "$\\theta=0.5$ & "
            "$\\theta=1$ \\\\\n"
        )
        tex.write("\\midrule\n")

        for dt in dts:
            table_row = [f"{dt:g}"]

            for theta in theta_values:
                value = lookup.get((dt, theta))

                if value is None:
                    table_row.append("--")
                else:
                    formatted = seconds_to_runtime_text(value)
                    table_row.append(f"\\texttt{{{formatted}}}")

            tex.write(" & ".join(table_row) + " \\\\\n")

        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n\n")

print(f"Saved runtime LaTeX tables to: {runtime_latex_filename}")

plt.show()
