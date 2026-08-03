from firedrake import *

from config import *


def mismip_bed(x):
    X = x / xbar
    Bx = B0 + B2 * X**2 + B4 * X**4 + B6 * X**6
    return max_value(Bx, zdeep)


def reset_state():
    from domain import x, z_ref
    from fields import bed, zb, thick, zs
    from spaces import mesh3D, xref, z_ref_split, uvec_out, H, u_prev

    zb.interpolate(max_value(-90.0, bed))
    # Hydrostatic equilibrium: rho_i * H = -rho_w * zb
    # => H = -zb * (rho_w / rho_i)
    thick.interpolate(max_value(-zb * (rhow / rhoi), 1.0))
    zs.interpolate(zb + thick)

    sigma = z_ref_split / Lz
    z_new = zb + sigma * thick
    mesh3D.coordinates.interpolate(as_vector([xref, z_new]))

    uvec_out.assign(0.0)
    H.assign(0.0)
    u_prev.assign(0.0)

    u_init = as_vector([
        10.0 * (1.0 + 0.01 * sin(2.0 * pi * x / Lx)),
    ])

    uvec_out.interpolate(u_init)
