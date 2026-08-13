from firedrake import CheckpointFile
import numpy as np


def save_restart(filename, t, step, dt, theta_out):
    from domain import mesh3D
    from fields import thick, zb, zs
    from spaces import w, uvec_out

    with CheckpointFile(filename, "w") as afile:

        # Save the actual mesh on which all restart fields live.
        afile.save_mesh(mesh3D)

        afile.save_function(
            thick,
            name="thick",
        )

        afile.save_function(
            zb,
            name="zb",
        )

        afile.save_function(
            zs,
            name="zs",
        )

        afile.save_function(
            w,
            name="w",
        )

        afile.save_function(
            uvec_out,
            name="uvec_out",
        )

    np.savez(
        filename.replace(".h5", "_meta.npz"),
        t=t,
        step=step,
        dt=dt,
        theta_out=theta_out,
    )


def load_restart(filename):
    from domain import mesh3D
    from fields import thick, zb, zs
    from spaces import w, uvec_out

    with CheckpointFile(filename, "r") as afile:

        # IMPORTANT:
        # mesh3D must have been loaded from THIS checkpoint by domain.py.
        # We do not load a second mesh here.

        thick_restart = afile.load_function(
            mesh3D,
            name="thick",
        )

        zb_restart = afile.load_function(
            mesh3D,
            name="zb",
        )

        zs_restart = afile.load_function(
            mesh3D,
            name="zs",
        )

        w_restart = afile.load_function(
            mesh3D,
            name="w",
        )

        uvec_out_restart = afile.load_function(
            mesh3D,
            name="uvec_out",
        )

        thick.assign(thick_restart)
        zb.assign(zb_restart)
        zs.assign(zs_restart)
        w.assign(w_restart)
        uvec_out.assign(uvec_out_restart)

    metadata_filename = filename.replace(
        ".h5",
        "_meta.npz",
    )

    with np.load(metadata_filename) as meta:

        return (
            float(meta["t"]),
            int(meta["step"]),
            float(meta["dt"]),
            float(meta["theta_out"]),
        )