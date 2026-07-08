from firedrake import *
from firedrake import CheckpointFile
import numpy as np
import matplotlib.pyplot as plt
import os

T = 100
dts = [0.1, 1, 2, 5, 10, 20, 50]
theta_values = [0, 1]

ref_dt = 0.01
ref_theta = 0

resolution_pairs = [
    (100, 10),
#    (200, 20),
#    (400, 40),
#    (800, 80),
#    (1600, 160),
]

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

fig, ax = plt.subplots(figsize=(5, 4))

for theta in theta_values:

    for nx, nz in resolution_pairs:

        # Fixed reference: theta = 0, dt = 0.001
        _, Href, zs_ref = load_state(ref_dt, ref_theta, nx, nz)

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

            plot_dts.append(dt)
            errors.append(err)

        plot_dts = np.array(plot_dts)
        errors = np.array(errors)
        order = np.argsort(plot_dts)

        ax.loglog(plot_dts[order],errors[order],"o-",label=fr"$\theta={theta}$, $n_x={nx}$, $n_z={nz}$")

ax.set_xlabel(r"$\Delta t$ [years]")
ax.set_ylabel(r"$L_2$ error in ice thickness")
ax.grid(True, which="both")
ax.legend(fontsize=8)

plt.tight_layout()

os.makedirs("Simulations", exist_ok=True)
plt.savefig("Simulations/thickness_error_vs_dt_one_panel.png", dpi=700)
