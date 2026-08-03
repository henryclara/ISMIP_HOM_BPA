from firedrake import *
from firedrake import CheckpointFile
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

T = 100
dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_values = [0, 0.5, 1]

ref_dt = 0.01
ref_theta = 0

resolution_pairs = [
    (100, 10),
    (200, 20),
    (400, 40),
    (800, 80),
    (1600, 160),
]

error_rows = []

colors = plt.cm.viridis(
    np.linspace(0.05, 0.95, len(resolution_pairs))
)

theta_colors = {
    0: plt.cm.viridis(0.1),
    0.5: plt.cm.viridis(0.4),
    1: plt.cm.viridis(0.8),
}

def load_state(dt, theta, nx, nz):
    filename = (
        f"Simulations/BPA_output_dt{dt:g}_theta{theta:g}_nx{nx}_nz{nz}"
        f"/restart_t{T:g}.h5"
    )

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Restart file not found: {filename}.\n"
            "Check that the simulation produced this restart, or set `ref_dt` to an existing run."
        )

    with CheckpointFile(filename, "r") as afile:
        # Try extruded mesh first (some restarts store `thick`/`zs` there)
        try:
            mesh = afile.load_mesh(name="firedrake_default_extruded", reorder=False)
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")
        except Exception:
            # Fallback to default mesh
            mesh = afile.load_mesh()
            H = afile.load_function(mesh, "thick")
            zs = afile.load_function(mesh, "zs")

    return mesh, H, zs


def L2_error(H, Href, zs):
    """
    Horizontal-plane L2 error in thickness.
    Uses n_z * ds_t so that n_z ds_t = dx.
    """

    Href_on_H = Function(H.function_space(), name="Href_on_H")

    # This assumes H and Href have the same nx, nz and same dof ordering.
    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    n_z = 1.0 / sqrt(1.0 + zs.dx(0)**2)

    return float(sqrt(assemble((H - Href_on_H)**2 * n_z * ds_t)))

def latex_number(x):
    if x == 0:
        return "$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent

    return (
        f"${mantissa:.2f}"
        f"\\times10^{{{exponent}}}$"
    )

# One panel per mesh resolution
ncols = 3
nrows = int(np.ceil(len(resolution_pairs) / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.5 * ncols, 3.8 * nrows),
    sharex=True,
    sharey=True,
    squeeze=False,
)

axes_flat = axes.ravel()

theta_styles = {
    0: "-",
    0.5: ":",
    1: "--",
}

theta_markers = {
    0: "s",
    0.5: "o",
    1: "s",
}

for panel_index, ((nx, nz), color) in enumerate(zip(resolution_pairs, colors)):
    ax = axes_flat[panel_index]

    # Fixed reference on the same mesh:
    # theta = 0, dt = 0.01
    _, Href, zs_ref = load_state(ref_dt, ref_theta, nx, nz)

    for theta in theta_values:
        plot_dts = []
        errors = []

        for dt in dts:
            # Skip only the actual reference simulation itself
            if dt == ref_dt and theta == ref_theta:
                continue

            try:
                _, H, zs = load_state(dt, theta, nx, nz)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

            err = L2_error(H, Href, zs)

            error_rows.append(
                {
                    "T": T,
                    "nx": nx,
                    "nz": nz,
                    "dt": dt,
                    "theta": theta,
                    "L2_error": err,
                }
)

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
            label=fr"$\theta={theta}$",
        )

    ax.set_title(fr"$n_x={nx},\ n_z={nz}$")
    ax.grid(True, which="both")
    ax.legend(fontsize=9)

# Hide unused panels
for ax in axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)

# Shared axis labels
fig.supxlabel(r"$\Delta t$ [a]")
fig.supylabel(r"$L_2$ error in ice thickness")

plt.tight_layout()

os.makedirs("Simulations", exist_ok=True)
plt.savefig(
    "Simulations/thickness_error_vs_dt_by_resolution.png",
    dpi=700,
    bbox_inches="tight",
)

# Sort by resolution, then timestep, then theta
error_rows.sort(
    key=lambda row: (
        row["nx"],
        row["nz"],
        row["dt"],
        row["theta"],
    )
)

csv_filename = f"Simulations/thickness_L2_errors_T{T:g}.csv"

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
        "dt",
        "theta",
        "L2_error",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in error_rows:
        writer.writerow(
            {
                "T": f'{row["T"]:g}',
                "nx": row["nx"],
                "nz": row["nz"],
                "dt": f'{row["dt"]:g}',
                "theta": f'{row["theta"]:g}',
                "L2_error": f'{row["L2_error"]:.8e}',
            }
        )

print(f"Saved error table to: {csv_filename}")

# ----------------------------------------------------------
# Save LaTeX tables
# ----------------------------------------------------------

latex_filename = f"Simulations/thickness_L2_errors_T{T:g}.tex"

with open(latex_filename, "w", encoding="utf-8") as tex:

    tex.write("% Automatically generated\n")
    tex.write("% L2 errors in ice thickness\n\n")

    for nx, nz in resolution_pairs:

        lookup = {
            (row["dt"], row["theta"]): row["L2_error"]
            for row in error_rows
            if row["nx"] == nx and row["nz"] == nz
        }

        tex.write("\\begin{table}[htbp]\n")
        tex.write("\\centering\n")
        tex.write(
            f"\\caption{{$L_2$ error in ice thickness at "
            f"$T={T:g}$ years for "
            f"$n_x={nx}$, $n_z={nz}$.}}\n"
        )
        tex.write(
            f"\\label{{tab:l2_nx{nx}_nz{nz}}}\n"
        )

        tex.write("\\begin{tabular}{lccc}\n")
        tex.write("\\toprule\n")
        tex.write(
            "$\\Delta t$ [a] & "
            "$\\theta=0$ & "
            "$\\theta=0.5$ & "
            "$\\theta=1$\\\\\n"
        )
        tex.write("\\midrule\n")

        for dt in dts:

            row = [f"{dt:g}"]

            for theta in theta_values:

                value = lookup.get((dt, theta))

                if value is None:
                    row.append("-")
                else:
                    row.append(latex_number(value))

            tex.write(
                " & ".join(row) + "\\\\\n"
            )

        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n\n")

print(f"Saved LaTeX tables to {latex_filename}")

plt.show()
