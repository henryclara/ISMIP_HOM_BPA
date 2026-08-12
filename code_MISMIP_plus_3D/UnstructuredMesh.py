import gmsh

gmsh.initialize()
gmsh.model.add("refined_rectangle")

# --------------------------------------------------
# Domain dimensions [m]
# --------------------------------------------------
Lx = 640000.0
Ly = 40000.0

h_coarse = 2000.0
h_fine = 500.0

x_ref_min = 400000.0
x_ref_max = 500000.0

# --------------------------------------------------
# Geometry
# --------------------------------------------------
surface = gmsh.model.occ.addRectangle(
    0.0, 0.0, 0.0,
    Lx, Ly
)

gmsh.model.occ.synchronize()

# --------------------------------------------------
# Identify the four boundary curves
# --------------------------------------------------
boundary_curves = gmsh.model.getBoundary(
    [(2, surface)],
    oriented=False,
    recursive=False
)

left = []
right = []
bottom = []
top = []

tol = 1.0e-6

for dim, tag in boundary_curves:

    xcentre, ycentre, zcentre = gmsh.model.occ.getCenterOfMass(
        dim, tag
    )

    if abs(xcentre - 0.0) < tol:
        left.append(tag)

    elif abs(xcentre - Lx) < tol:
        right.append(tag)

    elif abs(ycentre - 0.0) < tol:
        bottom.append(tag)

    elif abs(ycentre - Ly) < tol:
        top.append(tag)


# --------------------------------------------------
# Physical boundary groups
#
# Keep the same numbering as Firedrake RectangleMesh:
#
# 1 = left
# 2 = right
# 3 = bottom
# 4 = top
# --------------------------------------------------
gmsh.model.addPhysicalGroup(
    1, left,
    tag=1,
    name="left"
)

gmsh.model.addPhysicalGroup(
    1, right,
    tag=2,
    name="right"
)

gmsh.model.addPhysicalGroup(
    1, bottom,
    tag=3,
    name="bottom"
)

gmsh.model.addPhysicalGroup(
    1, top,
    tag=4,
    name="top"
)

# IMPORTANT: also mark the 2D domain itself
gmsh.model.addPhysicalGroup(
    2, [surface],
    tag=10,
    name="domain"
)


# --------------------------------------------------
# Mesh refinement
# --------------------------------------------------
refinement = gmsh.model.mesh.field.add("Box")

gmsh.model.mesh.field.setNumber(
    refinement, "VIn", h_fine
)

gmsh.model.mesh.field.setNumber(
    refinement, "VOut", h_coarse
)

gmsh.model.mesh.field.setNumber(
    refinement, "XMin", x_ref_min
)

gmsh.model.mesh.field.setNumber(
    refinement, "XMax", x_ref_max
)

gmsh.model.mesh.field.setNumber(
    refinement, "YMin", 0.0
)

gmsh.model.mesh.field.setNumber(
    refinement, "YMax", Ly
)

gmsh.model.mesh.field.setNumber(
    refinement, "Thickness", 0.0
)

gmsh.model.mesh.field.setAsBackgroundMesh(
    refinement
)

gmsh.option.setNumber(
    "Mesh.MeshSizeExtendFromBoundary", 0
)

gmsh.option.setNumber(
    "Mesh.MeshSizeFromPoints", 0
)

gmsh.option.setNumber(
    "Mesh.MeshSizeFromCurvature", 0
)

gmsh.option.setNumber(
    "Mesh.Algorithm", 6
)

# --------------------------------------------------
# Generate mesh
# --------------------------------------------------
gmsh.model.mesh.generate(2)

gmsh.write("mesh_res_2000_500.msh")

gmsh.fltk.run()

gmsh.finalize()
