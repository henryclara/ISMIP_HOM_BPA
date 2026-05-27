from firedrake import *
import numpy as np

def save_restart(filename, t, step, dt, theta_out):
    from domain import mesh3D
    from fields import thick, zb, zs
    from spaces import w, uvec_out
    with CheckpointFile(filename, "w") as afile:
        afile.save_mesh(mesh3D)
        afile.save_function(thick, name="thick")
        afile.save_function(zb, name="zb")
        afile.save_function(zs, name="zs")
        afile.save_function(w, name="w")
        afile.save_function(uvec_out, name="uvec_out")

    np.savez(filename.replace(".h5", "_meta.npz"), t=t, step=step, dt=dt, theta_out=theta_out)
