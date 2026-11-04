#!/usr/bin/env python3
"""Figure for the real-sequence depth experiment (Section 12)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "real_depth_collapse.json"),
          "r", encoding="utf-8") as fh:
    data = json.load(fh)

s = data["solvers"]["pts_only"]
FX = 517.306408
sigma = data["sigma_px"]
charts = ("xyz", "id", "ir", "pa")
styles = {
    "xyz": dict(color="#b2182b", marker="o", label="XYZ"),
    "id": dict(color="#2166ac", marker="s", label="inverse depth"),
    "ir": dict(color="#4d9221", marker="^", label="inverse range"),
    "pa": dict(color="#762a83", marker="D", label="parallax"),
}

# every retained feature of the main solver, as a grey backdrop
beta_all, err_all, pred_all = [], [], []
for w in data["windows"]:
    if w.get("status") != "ok" or "crb" not in w:
        continue
    c = w["mc"].get("xyz", {})
    if "rel_std" not in c:
        continue
    b = np.asarray(w["beta"], float)
    e = np.asarray(c["rel_std"], float)
    p = np.asarray(w["crb"], float)
    beta_all.append(b)
    err_all.append(e)
    pred_all.append(p)
beta_all = np.concatenate(beta_all)
err_all = np.concatenate(err_all)
pred_all = np.concatenate(pred_all)
ok = np.isfinite(err_all) & np.isfinite(pred_all) & (err_all > 0)

plt.rcParams.update({
    "font.size": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

# ---------------------------------------------------------------- (a) geometry
ax = axes[0]
ax.scatter(pred_all[ok], err_all[ok], s=2.5, c="#999999", alpha=0.30,
           linewidths=0, rasterized=True, zorder=1)
pe = np.array(s["bins_pred_decades"])
centres = 10 ** (0.5 * (pe[:-1] + pe[1:]))
# the four medians land on top of each other; they are nudged horizontally by
# 4% so that all four rings stay visible, and by nothing else
nudge = {"xyz": 0.955, "id": 1.045, "ir": 0.925, "pa": 1.075}
for c in charts:
    med = np.array([np.nan if v is None else v
                    for v in s["curves_mc"][c]["median"]])
    ax.plot(centres * nudge[c], med, linestyle="none", mfc="none", mew=1.0,
            ms=4.4, zorder=3,
            **{k: v for k, v in styles[c].items() if k != "label"})
lim = np.array([7e-4, 0.5])
ax.plot(lim, lim, "-", color="#333333", lw=0.8, zorder=2)
rat = s["mc_ratio"]["xyz"]["median"]
ax.plot(lim, rat * lim, "--", color="#333333", lw=0.8, zorder=2)
ax.text(0.13, 0.30, r"$y=x$", fontsize=6.2, color="#333333")
ax.text(1.15e-3, rat * 1.15e-3 * 3.1, r"$y=%.2f\,x$" % rat,
        fontsize=6.2, color="#333333")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(7e-4, 0.5)
ax.set_xlabel(r"Fisher bound on $\sigma_d/d$")
ax.set_ylabel(r"measured $\sigma_{\hat d}/\hat d$")
ax.set_title("(a) the triangle sets the error", loc="left")

# ---------------------------------------------------------------- (b) equality
ax = axes[1]
wins = [w for w in data["windows"]
        if w.get("status") == "ok" and "depth_est" in w["charts"].get("xyz", {})]
for c in charts:
    xs, ys = [], []
    for w in wins:
        r = w["charts"].get(c, {})
        if "depth_est" not in r:
            continue
        a = np.asarray(r["depth_est"], float)
        b = np.asarray(w["charts"]["xyz"]["depth_est"], float)
        okk = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        if not np.any(okk):
            continue
        xs.append(np.median(w["beta"]))
        ys.append(np.max(np.abs(a[okk] - b[okk]) / np.abs(b[okk])))
    ax.semilogy(xs, ys, linestyle="none", ms=3.4, mfc="none", mew=0.9,
                zorder=3, **{k: v for k, v in styles[c].items() if k != "label"})
ax.axhline(1e-6, color="#333333", lw=0.8, ls="--", zorder=2)
ax.text(39.0, 1.35e-6, r"$10^{-6}$", fontsize=6.2, color="#333333", ha="right")
ax.set_xscale("log")
ax.set_xlim(0.1, 40)
ax.set_ylim(3e-13, 3e-6)
ax.set_xlabel(r"parallax angle $\beta$ (deg)")
ax.set_ylabel("relative depth difference vs XYZ")
ax.set_title("(b) the four depth estimates are one", loc="left")

fig.tight_layout(pad=0.3, w_pad=0.9, rect=(0, 0.16, 1, 1))
handles = [Line2D([], [], color=styles[c]["color"], marker=styles[c]["marker"],
                  mfc="none", ms=4.0, lw=0, mew=1.0, label=styles[c]["label"])
           for c in charts]
handles.append(Line2D([], [], color="#999999", marker="o", ms=2.4, lw=0,
                      label="individual features"))
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
           fontsize=6.4, handlelength=1.2, columnspacing=1.3,
           labelspacing=0.35, bbox_to_anchor=(0.5, 0.005))

out = os.path.join(ROOT, "figures", "real_depth.png")
fig.savefig(out)
print("wrote", out, os.path.getsize(out))
