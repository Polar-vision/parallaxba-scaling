"""Re-run Simulation 2 of Zhao et al. (IJRR 2015) with the control it omits.

The original (Section 6.1.2, Table 3) reports that on Simulation 2 ParallaxBA
converges under Gauss-Newton while inverse-depth BA is marked "Singular", and
attributes the failure to features lying along the direction of camera motion.
Table 3 is a two-column design: Simulation 1 compares ParallaxBA against XYZ,
Simulation 2 compares ParallaxBA against inverse depth.  **XYZ is never run on
Simulation 2**, so the experiment that is used to indict inverse depth has no
Euclidean control on the same geometry.

This script supplies that control and separates the three candidate causes:

  R1  conditioning at the true configuration      which quantity is chart
                                                 dependent and which is not
  R2  the completed Table 3                       all four charts, GN and LM,
                                                 from one shared initialization
  R3  initialization tolerance                    how much depth error each
                                                 chart's GN survives
  R4  motion direction                            the same sweep with a
                                                 lateral instead of forward
                                                 translation

Sensor and problem parameters follow Tables 1 and 2 of the original: focal
400 px, 800x800, principal point at the centre, FOV +-45 deg, image noise
sigma = 0.1 px, 21 poses on a straight line at 2 m per step.

The original feature box ([+-25,+-25,+-5] m) is rescaled to a slab ahead of the
traverse, because a 10 m deep box with a 40 m traverse puts most features
behind the last camera.  What is preserved is the property the original names:
a subset of features lies close to the motion axis, so their parallax is small.
The angle beta between each feature bearing and the motion axis is reported.

Usage:  python code/repro_sim2_ablation.py
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_chart_experiment as sce  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "repro_sim2_ablation.json")

# --- Table 1 / Table 2 of the original -------------------------------------
FOCAL = 400.0
SIGMA_PX = 0.1
SIGMA = SIGMA_PX / FOCAL
N_CAM = 21
STEP = 2.0
N_PT = 120
N_AXIAL = 24
AXIAL_RADIUS = (0.6, 5.0)      # metres off the optical axis
SLAB = (45.0, 75.0)            # depth range, metres
SEED = 11
CHARTS = ("xyz", "id", "ir", "pa")


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------
def project(R, c, X):
    y = R @ (X - c)
    if y[2] <= 1e-6:
        return None, None
    return y, np.array([y[0] / y[2], y[1] / y[2]])


def build_scene(motion="forward", slab=SLAB, seed=SEED, noise=SIGMA):
    rng = np.random.default_rng(seed)
    if motion == "forward":
        cams = [np.array([0.0, 0.0, STEP * k]) for k in range(N_CAM)]
    elif motion == "lateral":
        span = STEP * (N_CAM - 1)
        cams = [np.array([-0.5 * span + STEP * k, 0.0, 0.0])
                for k in range(N_CAM)]
    else:
        raise ValueError(motion)
    Rs = [np.eye(3) for _ in range(N_CAM)]

    pts, axial = [], []
    for i in range(N_PT):
        if i < N_AXIAL:
            rad = rng.uniform(*AXIAL_RADIUS)
            th = rng.uniform(0.0, 2.0 * math.pi)
            pts.append(np.array([rad * math.cos(th), rad * math.sin(th),
                                 rng.uniform(*slab)]))
            axial.append(True)
        else:
            pts.append(np.array([rng.uniform(-30.0, 30.0),
                                 rng.uniform(-30.0, 30.0),
                                 rng.uniform(*slab)]))
            axial.append(False)

    obs, u, index = [], [], {}
    for i in range(N_CAM):
        for j in range(len(pts)):
            y, uv = project(Rs[i], cams[i], pts[j])
            if y is None:
                continue
            if abs(uv[0]) > 1.0 or abs(uv[1]) > 1.0:
                continue
            if noise > 0:
                uv = uv + rng.normal(scale=noise, size=2)
            index[(i, j)] = len(obs)
            obs.append((i, j))
            u.append(uv)
    return {"cams": cams, "Rs": Rs, "pts": pts, "obs": obs,
            "u": np.concatenate(u), "axial": np.array(axial),
            "motion": motion, "index": index}


def triangulate(scene):
    """Two-view initialization from the widest observed pair, with cheirality."""
    cams, Rs, u, index = scene["cams"], scene["Rs"], scene["u"], scene["index"]
    per_pt = {}
    for (i, j) in scene["obs"]:
        per_pt.setdefault(j, []).append(i)
    out = []
    for j in range(len(scene["pts"])):
        seen = sorted(per_pt.get(j, []))
        if len(seen) < 2:
            out.append(scene["pts"][j].copy())
            continue
        i1, i2 = seen[0], seen[-1]
        k1, k2 = index[(i1, j)], index[(i2, j)]
        b1 = Rs[i1].T @ np.array([u[2 * k1], u[2 * k1 + 1], 1.0])
        b2 = Rs[i2].T @ np.array([u[2 * k2], u[2 * k2 + 1], 1.0])
        b1 /= np.linalg.norm(b1)
        b2 /= np.linalg.norm(b2)
        sol, *_ = np.linalg.lstsq(np.column_stack([b1, -b2]),
                                  cams[i2] - cams[i1], rcond=None)
        lam = float(sol[0])
        # keep the point in front of every camera that observed it
        need = max(float(np.dot(cams[i] - cams[i1], b1)) for i in seen) + 1.0
        if not np.isfinite(lam):
            lam = need
        lam = max(lam, need)
        out.append(cams[i1] + lam * b1)
    return np.array(out)


def scale_axial(scene, init, factor):
    """Move the near-axis features along their bearing from camera 0."""
    c0 = scene["cams"][0]
    out = np.array(init, dtype=float)
    ax = scene["axial"]
    out[ax] = c0 + factor * (out[ax] - c0)
    return out


def make_state(scene, init):
    return {"cams": [c.copy() for c in scene["cams"]],
            "Rs": [R.copy() for R in scene["Rs"]],
            "pts": [np.array(X, dtype=float).copy() for X in init],
            "obs": list(scene["obs"]), "u": scene["u"].copy()}


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run(state, chart, method, max_iter=60, lam0=1e-4):
    layout = sce.cam_layout(len(state["cams"]))
    n_cv = layout[1]
    params = sce.init_params(state, chart)
    if params is None:
        return {"status": "chart_undefined", "iter": 0}
    if method == "lm":
        return sce.solve(state, chart, method="lm", metric="chart",
                         max_iter=max_iter, lam0=lam0)

    cams = [c.copy() for c in state["cams"]]
    Rs = [R.copy() for R in state["Rs"]]
    cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
    cost = sce.cost_of(cur, chart, params)
    if not np.isfinite(cost):
        return {"status": "invalid_init", "iter": 0}
    cost0 = cost
    n_it, status = 0, "max_iter"
    while n_it < max_iter:
        A, B, r, _, _, _, ok = sce.assemble(cur, chart, params, layout)
        if not ok:
            status = "behind_camera"
            break
        H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
        g = np.concatenate([A.T @ r, B.T @ r])
        if np.max(np.abs(g)) < 1e-11:
            status = "converged"
            break
        dc, dp, _ = sce.schur_solve(H, g, n_cv, len(params))
        if dc is None:
            status = "singular"
            break
        step = float(np.linalg.norm(dc) + np.linalg.norm(dp))
        if not np.isfinite(step) or step > 1e5:
            status = "diverged"
            break
        cams, Rs, params = sce.apply_step(cams, Rs, params, layout, dc, dp)
        cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
        cost = sce.cost_of(cur, chart, params)
        n_it += 1
        if not np.isfinite(cost) or cost > 100.0 * max(cost0, 1e-30):
            status = "diverged"
            break
    X = np.array([sce.chart_forward(chart, p, cams, Rs) for p in params])
    return {"status": status, "iter": n_it, "cost": cost, "X": X}


def summarize(res, scene):
    out = {"status": res["status"], "iter": res["iter"]}
    if res.get("X") is None:
        return out
    c0 = scene["cams"][0]
    truth = np.array(scene["pts"])
    d_t = np.linalg.norm(truth - c0, axis=1)
    d_r = np.linalg.norm(np.asarray(res["X"]) - c0, axis=1)
    rel = np.abs(d_r - d_t) / d_t
    ax = scene["axial"]
    out["final_cost"] = float(res["cost"])
    out["rmse_px"] = float(FOCAL * math.sqrt(2.0 * res["cost"]
                                            / len(scene["obs"])))
    out["median_rel_depth_error"] = float(np.median(rel))
    out["median_rel_axial"] = float(np.median(rel[ax]))
    out["max_rel_axial"] = float(np.max(rel[ax]))
    out["median_rel_offaxis"] = float(np.median(rel[~ax]))
    return out


def conditioning(scene, chart):
    """Chart-dependent and problem-dependent conditioning at the truth."""
    st = make_state(scene, np.array(scene["pts"]))
    layout = sce.cam_layout(len(scene["cams"]))
    n_cv = layout[1]
    params = sce.init_params(st, chart)
    if params is None:
        return {"status": "chart_undefined"}
    A, B, _, _, _, _, ok = sce.assemble(st, chart, params, layout)
    if not ok:
        return {"status": "behind_camera"}
    H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
    Hcc, Hcp, Hpp = H[:n_cv, :n_cv], H[:n_cv, n_cv:], H[n_cv:, n_cv:]
    S = Hcc - Hcp @ np.linalg.solve(Hpp, Hcp.T)
    ev_h = np.linalg.eigvalsh(H)
    ev_s = np.linalg.eigvalsh(S)
    return {
        "cond_H": float(ev_h[-1] / max(ev_h[0], 1e-300)),
        "cond_point_block": float(np.linalg.cond(Hpp)),
        "cond_S": float(ev_s[-1] / ev_s[0]),
        "min_eig_S": float(ev_s[0]),
        "rank_S": int(np.sum(ev_s > 1e-10 * ev_s[-1])),
        "n_S": int(len(ev_s)),
    }


def beta_of(scene):
    c0 = scene["cams"][0]
    axis = scene["cams"][-1] - c0
    axis = axis / np.linalg.norm(axis)
    out = []
    for X in scene["pts"]:
        r = X - c0
        r = r / np.linalg.norm(r)
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, float(r @ axis))))))
    return np.array(out)


# --------------------------------------------------------------------------
def main():
    scene = build_scene("forward")
    truth = np.array(scene["pts"])
    init = triangulate(scene)
    b = beta_of(scene)
    ax = scene["axial"]
    c0 = scene["cams"][0]
    d_t = np.linalg.norm(truth - c0, axis=1)
    d_i = np.linalg.norm(init - c0, axis=1)
    rel_init = np.abs(d_i - d_t) / d_t

    result = {
        "protocol": {
            "source": "Zhao et al., IJRR 2015, Tables 1-3 and Section 6.1.2",
            "focal_px": FOCAL, "sigma_px": SIGMA_PX, "n_cam": N_CAM,
            "step_m": STEP, "n_pt": N_PT, "n_axial": N_AXIAL,
            "slab_m": list(SLAB), "axial_radius_m": list(AXIAL_RADIUS),
            "gauge": "camera 0 fixed, camera 1 translation fixed",
            "init": "two-view triangulation, widest pair, cheirality safe",
            "beta_deg": {"min": float(b.min()), "median": float(np.median(b)),
                         "axial_median": float(np.median(b[ax]))},
            "init_median_err": float(np.median(rel_init)),
            "init_median_err_axial": float(np.median(rel_init[ax])),
            "init_max_err_axial": float(np.max(rel_init[ax])),
            "observations": len(scene["obs"]),
        },
        "R1_conditioning": {}, "R2_completed_table": {},
        "R3_tolerance": {}, "R4_lateral": {},
    }

    print("== R1: conditioning at the true configuration ==")
    print("   beta min %.2f  median %.2f  axial %.2f deg"
          % (b.min(), np.median(b), np.median(b[ax])))
    print("   init error: median %.3g  axial median %.3g  axial max %.3g"
          % (np.median(rel_init), np.median(rel_init[ax]),
             np.max(rel_init[ax])))
    print("   %-4s %11s %11s %12s %s" % ("", "cond H", "cond Hpp", "cond S", "rank S"))
    for chart in CHARTS:
        c = conditioning(scene, chart)
        result["R1_conditioning"][chart] = c
        if c.get("status"):
            print("   %-4s %s" % (chart, c["status"]))
            continue
        print("   %-4s %11.3e %11.3e %12.6f %d/%d"
              % (chart, c["cond_H"], c["cond_point_block"], c["cond_S"],
                 c["rank_S"], c["n_S"]))

    print("== R2: the completed Table 3 (all four charts, shared init) ==")
    for chart in CHARTS:
        for method in ("gn", "lm"):
            row = summarize(run(make_state(scene, init), chart, method), scene)
            result["R2_completed_table"].setdefault(chart, {})[method] = row
            print("   %-4s %-2s %-12s it=%3d rmse=%8.4f px  depth=%.3g  axial=%.3g"
                  % (chart, method, row["status"], row["iter"],
                     row.get("rmse_px", float("nan")),
                     row.get("median_rel_depth_error", float("nan")),
                     row.get("median_rel_axial", float("nan"))))

    print("== R3: initialization tolerance (axial depth scaled) ==")
    scales = (1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0, 3.0, 5.0)
    for chart in CHARTS:
        result["R3_tolerance"][chart] = {}
        line = "   %-4s" % chart
        for s in scales:
            st = make_state(scene, scale_axial(scene, truth, s))
            row = summarize(run(st, chart, "gn"), scene)
            result["R3_tolerance"][chart]["%g" % s] = row
            line += "  %s:%s" % (s, "ok" if row["status"] == "converged"
                                 else row["status"][:5])
        print(line)

    print("== R4: same sweep, lateral translation ==")
    lat = build_scene("lateral")
    lat_truth = np.array(lat["pts"])
    for chart in CHARTS:
        result["R4_lateral"][chart] = {}
        line = "   %-4s" % chart
        for s in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            st = make_state(lat, scale_axial(lat, lat_truth, s))
            row = summarize(run(st, chart, "gn"), lat)
            result["R4_lateral"][chart]["%g" % s] = row
            line += "  %s:%s" % (s, "ok" if row["status"] == "converged"
                                 else row["status"][:5])
        print(line)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
