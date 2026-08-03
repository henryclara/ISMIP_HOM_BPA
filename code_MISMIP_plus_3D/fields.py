from firedrake import *
from domain import *
from geometry import *
from spaces import *

beta2 = Function(Vbar, name="beta2")
beta2.interpolate(Constant(1.0e4))

bed = Function(Vbar, name="bed").interpolate(mismip_bed(xref, yref))
zb = Function(Vbar, name="zb").interpolate(max_value(-90.0, bed))
thick = Function(Vbar, name="thick").interpolate(100.0)
zs = Function(Vbar, name="zs").interpolate(zb + thick)

delta_zb = Constant(1.0)
grounded_out = Function(Vbar, name="grounded")
