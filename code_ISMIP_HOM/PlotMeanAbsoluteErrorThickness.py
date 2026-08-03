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
    (3200, 320),
]

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
        try:
            mesh = afile.load_mesh(
                name="firedrake_default_extruded",
                reorder=False,
            )
        except KeyError:
            mesh_names = list(
                afile._get_mesh_name_topology_name_map().keys()
            )

            if not mesh_names:
                raise RuntimeError(
                    f"No mesh found in {filename}. "
                    "The checkpoint may be incomplete."
                )

            extruded_names = [
                name for name in mesh_names
                if "extruded" in name.lower()
            ]

            if extruded_names:
                mesh_name = extruded_names[0]
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


def MAE(H, Href, zs):
    """
    Mean absolute error in the horizontal thickness profile.
    Returns the error in metres.
    """

    Href_on_H = Function(H.function_space(), name="Href_on_H")
    Href_on_H.dat.data[:] = Href.dat.data_ro[:]

    n_z = 1.0 / sqrt(1.0 + zs.dx(0)**2)

    total_error = assemble(
        sqrt((H - Href_on_H)**2) * n_z * ds_t
    )

    domain_length = assemble(
        Constant(1.0) * n_z * ds_t
    )

    return float(total_error / domain_length)


def latex_number(x):
    if x == 0:
        return "$0$"

    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent

    return f"${mantissa:.2f}\\times10^{{{exponent}}}$"


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

for panel_index, (nx, nz) in enumerate(resolution_pairs):
    ax = axes_flat[panel_index]

    try:
        _, Href, _ = load_state(
            ref_dt,
            ref_theta,
            nx,
            nz,
        )
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

            err = MAE(H, Href, zs)

            error_rows.append(
                {
                    "T": T,
                    "nx": nx,
                    "nz": nz,
                    "dt": dt,
                    "theta": theta,
                    "MAE_m": err,
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
            label=fr"$\theta={theta:g}$",
        )

    ax.set_title(fr"$n_x={nx},\ n_z={nz}$")
    ax.grid(True, which="both")
    ax.legend(fontsize=9)

for ax in axes_flat[len(resolution_pairs):]:
    ax.set_visible(False)

fig.supxlabel(r"$\Delta t$ [a]")
fig.supylabel(r"Mean absolute error in ice thickness [m]")

plt.tight_layout()

os.makedirs("Simulations", exist_ok=True)

figure_filename = (
    "Simulations/thickness_MAE_vs_dt_by_resolution.png"
)

plt.savefig(
    figure_filename,
    dpi=700,
    bbox_inches="tight",
)

print(f"Saved figure to: {figure_filename}")

error_rows.sort(
    key=lambda row: (
        row["nx"],
        row["nz"],
        row["dt"],
        row["theta"],
    )
)

csv_filename = f"Simulations/thickness_MAE_T{T:g}.csv"

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
        "MAE_m",
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
                "MAE_m": f'{row["MAE_m"]:.8e}',
            }
        )

print(f"Saved CSV table to: {csv_filename}")

latex_filename = f"Simulations/thickness_MAE_T{T:g}.tex"

with open(
    latex_filename,
    "w",
    encoding="utf-8",
) as tex:

    tex.write("% Automatically generated\n")
    tex.write("% Requires \\usepackage{booktabs}\n\n")

    for nx, nz in resolution_pairs:
        lookup = {
            (row["dt"], row["theta"]): row["MAE_m"]
            for row in error_rows
            if row["nx"] == nx and row["nz"] == nz
        }

        tex.write("\\begin{table}[htbp]\n")
        tex.write("\\centering\n")
        tex.write(
            f"\\caption{{Mean absolute error in ice thickness at "
            f"$T={T:g}$ years for "
            f"$n_x={nx}$, $n_z={nz}$. "
            f"Errors are given in metres.}}\n"
        )
        tex.write(
            f"\\label{{tab:mae_nx{nx}_nz{nz}}}\n"
        )
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
                    table_row.append(latex_number(value))

            tex.write(" & ".join(table_row) + " \\\\\n")

        tex.write("\\bottomrule\n")
        tex.write("\\end{tabular}\n")
        tex.write("\\end{table}\n\n")

print(f"Saved LaTeX tables to: {latex_filename}")

plt.show()
