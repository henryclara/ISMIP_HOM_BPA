from firedrake import *
import numpy as np

from firedrake import CheckpointFile

def save_restart(filename, t, step, dt, theta_out):
    from domain import mesh3D
    from fields import thick, zb, zs
    from spaces import w, uvec_out
    with CheckpointFile(filename, "w") as afile:
        afile.save_mesh(mesh3D)
        afile.save_function(thick, name="thick")
        afile.save_function(zb, name="zb")
        afile.save_function(zs, name="zs")
        afile.save_function(zs, name="bed")
        afile.save_function(w, name="w")
        afile.save_function(uvec_out, name="uvec_out")

    np.savez(filename.replace(".h5", "_meta.npz"), t=t, step=step, dt=dt, theta_out=theta_out)

def load_restart(filename):
    from fields import thick, zb, zs
    from spaces import w, uvec_out
    from domain import mesh3D

    with CheckpointFile(filename, "r") as afile:
        thick.assign(afile.load_function(mesh3D, name="thick"))
        zb.assign(afile.load_function(mesh3D, name="zb"))
        zs.assign(afile.load_function(mesh3D, name="zs"))
        w.assign(afile.load_function(mesh3D, name="w"))
        uvec_out.assign(afile.load_function(mesh3D, name="uvec_out"))

    meta = np.load(filename.replace(".h5", "_meta.npz"))

    return (
        float(meta["t"]),
        int(meta["step"]),
        float(meta["dt"]),
        float(meta["theta_out"]),
    )
