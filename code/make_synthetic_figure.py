#!/usr/bin/env python3
"""Figure for the controlled synthetic chart experiment."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "synthetic_chart_experiment.json"),
          "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["S2_angle_sweep"]
beta = np.array([r["beta_deg"] for r in rows])
charts = ("xyz", "id", "ir", "pa")
styles = {
    "xyz": dict(color="#b2182b", marker="o", label="XYZ"),
    "id": dict(color="#2166ac", marker="s", label="inverse depth"),
    "ir": dict(color="#4d9221", marker="^", label="inverse range"),
    "pa": dict(color="#762a83", marker="D", label="parallax"),
}

plt.rcParams.update({
    "font.size": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

# (a) conditioning of the chart Hessian
ax = axes[0]
for c in charts:
    ax.loglog(beta, [r[c]["kappa_H"] for r in rows], "-", ms=3.0, lw=1.1, **styles[c])
ax.set_xscale("log")
ax.set_xlabel(r"$\beta$ (deg)")
ax.set_ylabel(r"$\kappa(H_q)$")
ax.set_title("(a) chart conditioning", loc="left")
ax.invert_xaxis()
ax.set_ylim(3e3, 6e12)

# (b) depth uncertainty: one curve for four charts, and the chain rule
ax = axes[1]
sd = np.array([r["xyz"]["sigma_depth_m"] for r in rows])
ax.loglog(beta, sd, "-", color="#333333", lw=3.2, alpha=0.30)
for c in charts:
    ax.loglog(beta, [r[c]["sigma_depth_m"] for r in rows], "o", ms=3.6,
              mfc="none", mec=styles[c]["color"], mew=0.9)
chain = np.array([r["pa"]["sigma_point_params"][2] * r["pa"]["dd_domega"] for r in rows])
ax.loglog(beta, chain, "--", color="#762a83", lw=0.9)
ax.set_xscale("log")
ax.set_xlabel(r"$\beta$ (deg)")
ax.set_ylabel(r"$\sigma_d$ (m)")
ax.set_title("(b) depth uncertainty", loc="left")
ax.invert_xaxis()
ax.set_ylim(4e-2, 4e2)

# (c) iterations of the same damped solver
ax = axes[2]
for c in charts:
    ax.semilogx(beta, [r[c]["lm_iter"] for r in rows], "-", ms=3.0, lw=1.1, **styles[c])
ax.set_xscale("log")
ax.set_xlabel(r"$\beta$ (deg)")
ax.set_ylabel("LM iterations")
ax.set_title("(c) damped solver", loc="left")
ax.invert_xaxis()

fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.13, 1, 1))

handles = [Line2D([], [], color=styles[c]["color"], marker=styles[c]["marker"],
                  ms=3.2, lw=1.1, label=styles[c]["label"]) for c in charts]
handles.append(Line2D([], [], color="#333333", lw=3.2, alpha=0.30,
                      label=r"$\sigma_d$ shared by all four"))
handles.append(Line2D([], [], color="#762a83", ls="--", lw=0.9,
                      label=r"$\sigma_\omega\,|\partial d/\partial\omega|$"))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=6.4, handlelength=1.5, columnspacing=1.4,
           labelspacing=0.35, bbox_to_anchor=(0.5, 0.005))

out = os.path.join(ROOT, "figures", "synthetic_chart.png")
fig.savefig(out)
print("wrote", out, os.path.getsize(out))
