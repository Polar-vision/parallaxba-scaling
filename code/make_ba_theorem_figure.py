#!/usr/bin/env python3
"""Figure for the 3D BA-level chart theorem."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "ijrr_3d_ba_theorem.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["experiment_E_near_singular"]["rows"]
om = np.array([r["omega"] for r in rows])
ratio = 1.0 / np.tan(om)
kX = np.array([r["kappa_IX"] for r in rows])
kq = np.array([r["kappa_Iq"] for r in rows])
kD = np.array([r["kappa_D"] for r in rows])
dX = np.array([abs(r["det_IX"]) for r in rows])
dq = np.array([abs(r["det_Iq"]) for r in rows])
depth = np.array([r["depth_information"] for r in rows])

plt.rcParams.update({
    "font.size": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "lines.markersize": 3.0,
})

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.45))

ax = axes[0]
ax.loglog(ratio, kX, "o-", color="#b2182b", label=r"$\kappa(I_X)$ XYZ")
ax.loglog(ratio, kq, "s-", color="#2166ac", label=r"$\kappa(I_q)$ par.")
ax.loglog(ratio, kD, "^--", color="#4d4d4d", label=r"$\kappa(D_q)$")
ax.loglog(ratio, ratio ** 2, ":", color="#b2182b", lw=1)
ax.loglog(ratio, ratio, ":", color="#4d4d4d", lw=1)
ax.set_xlabel(r"$d/B$")
ax.set_ylabel("condition number")
ax.set_title("(a) conditioning", loc="left")
ax.legend(frameon=False, loc="upper left", handlelength=1.6,
          borderpad=0.1, labelspacing=0.25)

ax = axes[1]
ax.loglog(ratio, dX, "o-", color="#b2182b", label=r"$|\det I_X|$")
ax.loglog(ratio, dq, "s-", color="#2166ac", label=r"$|\det I_q|$")
ax.set_xlabel(r"$d/B$")
ax.set_ylabel("determinant")
ax.set_title("(b) information volume", loc="left")
ax.legend(frameon=False, loc="lower left", handlelength=1.6,
          borderpad=0.1, labelspacing=0.25)

ax = axes[2]
ax.loglog(ratio, ratio, "--", color="#bbbbbb", lw=3.0, label=r"$d/B$")
ax.loglog(ratio, kD, "^-", color="#4d4d4d", lw=1.0, ms=3.0,
          label=r"$\sigma_{\max}/\sigma_{\min}$")
ax.set_xlabel(r"$d/B$")
ax.set_ylabel("aspect ratio")
ax.set_title("(c) trust region", loc="left")
ax.legend(frameon=False, loc="upper left", handlelength=1.6,
          borderpad=0.1, labelspacing=0.25)

fig.tight_layout()
out = os.path.join(ROOT, "figures", "ba_theorem.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
