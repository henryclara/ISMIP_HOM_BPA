from firedrake import *
import numpy as np

from config import *


def mismip_bed(x, y):
    # Keep the 2D simulation on the requested fixed horizontal cross-section.
    # The desired slice is the y = 60000 line, so the bed is frozen in y and
    # retains the natural x dependence on that plane.
    slice_y = 60000.0
    yc = slice_y - Ly_full / 2.0
    X = x / xbar

    Bx = B0 + B2 * X**2 + B4 * X**4 + B6 * X**6
    By = dc * (1.0 / (1.0 + exp(-2.0 * (yc - wc) / fc))
        + 1.0 / (1.0 + exp( 2.0 * (yc + wc) / fc)))

    return max_value(Bx + By, zdeep)

def reset_state():
    from domain import x
    from fields import bed, zb, thick, zs
    from spaces import mesh3D, xref, sigmaref, w, uvec_out, H, u_prev

    zb.interpolate(max_value(-90.0, bed))
    thick.interpolate(100.0)
    zs.interpolate(zb + thick)

    mesh3D.coordinates.interpolate(
        as_vector([xref, zb + sigmaref * thick])
    )

    w.assign(0.0)
    uvec_out.assign(0.0)
    H.assign(0.0)
    u_prev.assign(0.0)

    u_init = 10.0 * (1.0 + 0.01 * sin(2.0 * pi * x / Lx))

    w.sub(0).interpolate(u_init)

    u0 = w.sub(0)
    w.sub(2).project(thick * u0.dx(0))
