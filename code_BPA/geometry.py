from firedrake import *
import numpy as np

from config import *


def mismip_bed(x, y):
    yc = y - Ly_full / 2.0
    X = x / xbar

    Bx = B0 + B2 * X**2 + B4 * X**4 + B6 * X**6
    By = dc * (1.0 / (1.0 + exp(-2.0 * (yc - wc) / fc))
        + 1.0 / (1.0 + exp( 2.0 * (yc + wc) / fc)))

    return max_value(Bx + By, zdeep)

def reset_state():
    from domain import x, y
    from fields import bed, zb, thick, zs
    from spaces import mesh3D, xref, yref, sigmaref, w, uvec_out, H, u_prev

    zb.interpolate(max_value(-90.0, bed))
    thick.interpolate(100.0)
    zs.interpolate(zb + thick)

    mesh3D.coordinates.interpolate(
        as_vector([xref, yref, zb + sigmaref * thick])
    )

    w.assign(0.0)
    uvec_out.assign(0.0)
    H.assign(0.0)
    u_prev.assign(0.0)

    u_init = as_vector([
        10.0 * (1.0 + 0.01 * sin(2.0 * pi * x / Lx) * sin(2.0 * pi * y / Ly_full)),
        10.0 * (0.01 * sin(2.0 * pi * x / Lx) * sin(2.0 * pi * y / Ly_full)),
    ])

    w.sub(0).interpolate(u_init)
    w.sub(1).interpolate(u_init)

    u0 = w.sub(0)
    ux0, uy0 = split(u0)
    w.sub(2).project(thick * (ux0.dx(0) + uy0.dx(1)))
