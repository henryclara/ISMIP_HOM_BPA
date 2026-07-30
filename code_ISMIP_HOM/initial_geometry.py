import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

Lx = 80_000.0
psi = np.deg2rad(2.0)
omega = 2.0 * np.pi / Lx

x = np.linspace(0.0, Lx, 800)
zs = 500.0 * np.sin(omega * x)
zb = -1000.0 + 500.0 * np.sin(omega * x)

fig, ax = plt.subplots(figsize=(7, 5))

ax.fill_between(x / 1000.0, zb, zs, alpha=0.1)
ax.fill_between(x / 1000.0,zb,-5000,color="0.9",zorder=0)
ax.plot(x / 1000.0, zs, linewidth=2.0, color="black",alpha=0.8)
ax.plot(x / 1000.0, zb, linewidth=2.0, color="black",alpha=0.8)

for sigma in [0.25, 0.50, 0.75]:
    z_layer = zb + sigma * (zs - zb)

    ax.plot(x / 1000.0,z_layer,color="black",lw=2,alpha=0.8,linestyle=(0, (0.5, 8)),dash_capstyle="round")

# Labels
ax.text(10, np.interp(10_000, x, zs) + 180, r"$z_s(x)$",fontsize=20)
ax.text(52, np.interp(52_000, x, zb) - 280, r"$z_b(x) = b(x)$",fontsize=20)

xm = 18_000.0
zsm = np.interp(xm, x, zs)
zbm = np.interp(xm, x, zb)

thickness_arrow = FancyArrowPatch((xm / 1000.0, zbm),(xm / 1000.0, zsm),arrowstyle="<->",mutation_scale=14,linewidth=1.6,)
ax.add_patch(thickness_arrow)
ax.text(xm / 1000.0 + 1.4, 0.5 * (zsm + zbm),
        r"$H(x)$", fontsize=20, va="center")

xg = 0.7
yg = 0.88

# Visual length of the gravity arrow
L_plot = 0.28

# Exaggerated visual angle from vertical
psi_plot = np.deg2rad(15.0)

# Arrow endpoint in axes coordinates
dx_plot = L_plot * np.sin(psi_plot)
dy_plot = L_plot * np.cos(psi_plot)

# Dashed vertical reference line
ax.plot(
    [xg, xg],
    [yg, yg - L_plot],
    color="black",
    linewidth=2,
    transform=ax.transAxes,
    clip_on=False,
)

# Gravity arrow along the diagonal
gravity = FancyArrowPatch(
    (xg, yg),
    (xg + dx_plot, yg - dy_plot),
    arrowstyle="-|>",
    mutation_scale=20,
    linewidth=2,
    color="black",
    transform=ax.transAxes,
    clip_on=False,
)
ax.add_patch(gravity)

# Gravity label
ax.text(
    xg + dx_plot + 0.015,
    yg - dy_plot,
    r"$\mathbf{g}$",
    fontsize=20,
    va="center",
    transform=ax.transAxes,
)

theta = np.linspace(-np.pi / 2,-np.pi / 2 + psi_plot,80,)

arc_radius = 0.075

arc_x = xg + arc_radius * np.cos(theta)
arc_y = yg + arc_radius * np.sin(theta)

ax.plot(arc_x,arc_y,color="black",linewidth=1.4,transform=ax.transAxes,clip_on=False)

# Angle label
label_theta = -np.pi / 2 + 0.5 * psi_plot

ax.text(
    xg + 0.14 * np.cos(label_theta),
    yg + 0.15 * np.sin(label_theta),
    r"$\psi$",
    fontsize=20,
    ha="center",
    va="center",
    transform=ax.transAxes,
)


# Angle arc between vertical and gravity
theta = np.linspace(-np.pi / 2,-np.pi / 2 + psi_plot,60)

arc_x = xg + 3.0 * np.cos(theta)
arc_y = yg + 160.0 * np.sin(theta)

ax.set_xlim(0, 80)
ax.set_ylim(zb.min() - 500, zs.max() + 850)
ax.set_xlabel(r"$x$", fontsize=25)
ax.set_ylabel(r"$z$", fontsize=25)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

fig.tight_layout()

output = Path("Figures/initial_geometry_schematic.png")
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print(output)

