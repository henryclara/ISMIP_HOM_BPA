from firedrake import *
from geometry import *
from spaces import *

beta2 = Function(Vbar, name="beta2")
beta2.interpolate(Constant(1.0e4))

bed = Function(Vbar, name="bed").interpolate(mismip_bed(xref, yref))
zb = Function(Vbar, name="zb").interpolate(max_value(-90.0, bed))
thick = Function(Vbar, name="thick").interpolate(100.0)
zs = Function(Vbar, name="zs").interpolate(zb + thick)

# ============================================================
# Horizontal geometry
# ============================================================

x_2D, y_2D = SpatialCoordinate(base)

bed_2D = Function(Q_H, name="bed_2D")
bed_2D.interpolate(mismip_bed(x_2D, y_2D))

thick_2D = Function(Q_H, name="thick_2D")
thick_2D.interpolate(Constant(100.0))

zb_2D = Function(Q_H, name="zb_2D")
zb_2D.interpolate(
    max_value(
        bed_2D,
        -rhoi / rhow * thick_2D,
    )
)

zs_2D = Function(Q_H, name="zs_2D")
zs_2D.interpolate(zb_2D + thick_2D)

# =============================================================

delta_zb = Constant(1.0)
grounded_out = Function(Vbar, name="grounded")
