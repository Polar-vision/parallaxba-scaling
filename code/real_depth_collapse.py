#!/usr/bin/env python3
"""Experiment B: does depth error collapse onto the parallax angle, or onto the chart?

Experiment A tests the equality of the four charts at *one* configuration and
needs no ground truth.  This one asks a different question: once each chart is
driven to convergence on the same real window, is the *depth error* explained by
the chart, or by the geometry of the triangle?

Prediction under test
---------------------
The paper claims the chart is a re-parameterisation and nothing more.  If so:

  B1 (equality)  For one window and one noise realisation, all four charts must
                 return the same depth for the same feature.  Measured as a
                 paired relative difference; anything above 1e-6 falsifies the
                 theorem, exactly as in Experiment A.
  B2 (collapse)  The depth error, binned by the parallax angle beta, must be one
                 curve, not four.  The chart must carry no information about the
                 error once beta is known.

B2 is the claim that matters for the literature: a chart that looks better in a
table is a chart that was measured at a better beta.

Data
----
TUM RGB-D fr1/xyz.  Poses from the motion-capture track, GT depth from the
registered 16-bit Kinect images that ship with the sequence.  Depth is sampled
only at the anchor frame, and only where the 5x5 neighbourhood is dense and
fronto-parallel enough for the measurement to mean anything.

Usage
-----
    python real_depth_collapse.py --quick      # smoke test, two configs
    python real_depth_collapse.py              # full sweep
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_chart_experiment import (  # noqa: E402
    CHARTS, apply_step, assemble, build_T, cam_layout, chart_forward, cost_of,
    dproj, init_params, schur_solve, solve,
)
from real_invariance import (  # noqa: E402
    FX, FY, N_CAM, SEQ, pose_at, read_gt, read_index, to_normalized,
    track_window, triangulate,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "real_depth_collapse.json")

SIGMA_PX = 0.5           # KLT tracking accuracy on this sequence
DEPTH_SCALE = 5000.0     # TUM 16-bit PNG -> metres
MIN_DEPTH, MAX_DEPTH = 0.4, 8.0
MAX_REPROJ_PX = 3.0      # same gate as Experiment A
GT_REPROJ_PX = 5.0       # GT depth + GT pose must agree this well
N_BIN = 9
N_MC = 6               # noise realisations per window for the sensitivity


# --------------------------------------------------------------------------
# depth images
# --------------------------------------------------------------------------
def read_depth_index(seq):
    return read_index(os.path.join(seq, "depth.txt"))


def associate(a, b, max_diff=0.02):
    """Nearest-timestamp association, one-to-one, TUM style."""
    out = {}
    j = 0
    for t, name in a:
        best, bd = None, max_diff
        k = j
        while k < len(b) and b[k][0] < t + max_diff:
            d = abs(b[k][0] - t)
            if d < bd:
                best, bd = b[k][1], d
                j = k
            k += 1
        if best is not None:
            out[name] = best
    return out


def load_depth(seq, rel):
    img = cv2.imread(os.path.join(seq, rel), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.astype(np.float64) / DEPTH_SCALE


def sample_depth(dmap, uv, rad=2):
    """Depth at a pixel, rejected at holes and at depth discontinuities."""
    h, w = dmap.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (rad <= u < w - rad and rad <= v < h - rad):
        return None
    patch = dmap[v - rad:v + rad + 1, u - rad:u + rad + 1]
    valid = patch[patch > 0]
    if valid.size < 0.6 * patch.size:
        return None
    med = float(np.median(valid))
    if med < MIN_DEPTH or med > MAX_DEPTH:
        return None
    if (valid.max() - valid.min()) > 0.15 * med:
        return None
    return med


def gt_point(xn, z, R, c):
    """Back-project a registered-depth pixel; z is measured along the optical axis."""
    return c + R.T @ (np.array([xn[0], xn[1], 1.0]) * z)


def reproj_px_one(xn, X, cams, Rs):
    err = []
    for i in range(len(cams)):
        y = Rs[i] @ (X - cams[i])
        if y[2] <= 1e-9:
            return float("inf")
        err.append(np.hypot((y[0] / y[2] - xn[i][0]) * FX,
                            (y[1] / y[2] - xn[i][1]) * FY))
    return float(np.sqrt(np.mean(np.square(err))))


def parallax_deg(X, ca, cb):
    """Angle subtended at X by the two end cameras (the pa chart's beta).

    Measured in the world frame: rotating each ray into its own camera frame
    before taking the angle would fold the change of orientation between the
    two cameras into the parallax, which on a handheld sequence is of the same
    order as the parallax itself.
    """
    va = X - ca
    vb = X - cb
    cosang = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def depth_crb(X, cams, Rs, sigma_px, anchor=0):
    """Relative standard deviation of the depth of one point, from its own rays.

    The Fisher information of the eight bearing measurements, restricted to the
    depth direction.  Rotation contributes: a camera that turns between frames
    changes where a fixed point appears even when it does not translate, and
    that constrains depth as surely as parallax does.  On this sequence it
    dominates at the smallest baselines, which is why the parallax angle alone
    does not predict the error (Figure~\ref{fig:real-depth}).
    """
    F = np.zeros((3, 3))
    for i in range(len(cams)):
        y = Rs[i] @ (X - cams[i])
        if y[2] <= 1e-9:
            return None
        J = dproj(y) @ Rs[i]
        F += J.T @ J
    # J is d(normalised image)/dX, so the noise must be converted to the
    # normalised plane before it can be compared with it
    F /= (sigma_px / FX) ** 2
    try:
        C = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return None
    r = X - cams[anchor]
    dist = float(np.linalg.norm(r))
    u = r / dist
    var = float(u @ C @ u)
    return math.sqrt(max(var, 0.0)) / dist if var > 0 else None


# --------------------------------------------------------------------------
# undamped Gauss-Newton with a divergence guard
# --------------------------------------------------------------------------
def gn_solve(state, chart, max_iter=40):
    """Same step as the shared solver, but stops as soon as the cost explodes.

    The paper's theorem is about the linearisation, so the honest solver here is
    undamped Gauss-Newton: four identical steps must produce four identical
    orbits.  Without a guard, a low-parallax window can spend the whole budget
    walking away from the minimum, which is slow and tells us nothing new.
    """
    layout = cam_layout(len(state["cams"]))
    n_cv = layout[1]
    params = init_params(state, chart)
    if params is None:
        return {"status": "chart_undefined", "iter": 0, "X": None}
    cams = [c.copy() for c in state["cams"]]
    Rs = [R.copy() for R in state["Rs"]]
    cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
    cost = cost_of(cur, chart, params)
    if not np.isfinite(cost):
        return {"status": "invalid_init", "iter": 0, "X": None}
    worst, prev, it = cost, cost, 0
    for it in range(1, max_iter + 1):
        A, B, r, _, _, _, ok = assemble(cur, chart, params, layout)
        if not ok:
            return {"status": "behind_camera", "iter": it, "X": None}
        H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
        g = np.concatenate([A.T @ r, B.T @ r])
        if np.max(np.abs(g)) < 1e-11:
            break
        dc, dp, _ = schur_solve(H, g, n_cv, len(params))
        if dc is None:
            return {"status": "singular", "iter": it, "X": None}
        nc, nR, np_ = apply_step(cams, Rs, params, layout, dc, dp)
        trial = {"cams": nc, "Rs": nR, "obs": state["obs"], "u": state["u"]}
        new = cost_of(trial, chart, np_)
        if not np.isfinite(new) or new > 4.0 * worst:
            return {"status": "diverged", "iter": it, "X": None,
                    "cost": float(cost)}
        cams, Rs, params, cur, cost = nc, nR, np_, trial, new
        worst = max(worst, cost)
        if abs(prev - cost) <= 1e-13 * max(1.0, cost):
            break
        prev = cost
    X = np.array([chart_forward(chart, p, cams, Rs) for p in params])
    return {"status": "converged", "iter": it, "X": X, "cams": cams,
            "cost": float(cost)}


# --------------------------------------------------------------------------
# Levenberg-Marquardt, same rule as the shared solver but faster
# --------------------------------------------------------------------------
def lm_solve(state, chart, metric="physical", max_iter=100, max_backtrack=12,
             lam0=1e-4):
    """Identical to the shared LM, with two changes that only affect cost.

    The shared solver rebuilds the damping basis -- which contains a dense
    inverse of the full total-space map -- on every rejected trial step.  Here
    it is built once per iteration and scaled by lambda during backtracking.
    The cost is also read off the residual vector returned by ``assemble``
    instead of being recomputed from scratch.
    """
    layout = cam_layout(len(state["cams"]))
    n_cv = layout[1]
    params = init_params(state, chart)
    if params is None:
        return {"status": "chart_undefined", "iter": 0, "X": None}
    cams = [c.copy() for c in state["cams"]]
    Rs = [R.copy() for R in state["Rs"]]
    cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
    cost = cost_of(cur, chart, params)
    if not np.isfinite(cost):
        return {"status": "invalid_init", "iter": 0, "X": None}
    lam, status, it = lam0, "max_iter", 0
    for it in range(1, max_iter + 1):
        A, B, r, _, _, _, ok = assemble(cur, chart, params, layout)
        if not ok:
            status = "behind_camera"
            break
        cost = 0.5 * float(r @ r)
        H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
        g = np.concatenate([A.T @ r, B.T @ r])
        if np.max(np.abs(g)) < 1e-11:
            status = "converged"
            break
        if metric == "chart":
            basis = np.eye(H.shape[0])
        else:
            T = build_T(chart, params, cams, Rs, layout, len(params))
            Ti = np.linalg.inv(T)
            basis = Ti.T @ Ti
        accepted = False
        for _ in range(max_backtrack):
            dc, dp, _ = schur_solve(H, g, n_cv, len(params), lam * basis)
            if dc is None:
                lam *= 10.0
                continue
            nc, nR, np_ = apply_step(cams, Rs, params, layout, dc, dp)
            trial = {"cams": nc, "Rs": nR, "obs": state["obs"], "u": state["u"]}
            new = cost_of(trial, chart, np_)
            if np.isfinite(new) and new < cost:
                cams, Rs, params, cur, cost = nc, nR, np_, trial, new
                lam = max(lam / 3.0, 1e-12)
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            status = "stalled"
            break
        # a scale-free stopping rule: the fixed 1e-13 threshold of the shared
        # solver is unreachable for chart coordinates whose own magnitude is
        # O(1) in radians, so the parallax chart never reports convergence even
        # when it has stopped moving.  All four charts see the same rule here.
        if np.linalg.norm(dc) + np.linalg.norm(dp) < 1e-10 * (1.0 + np.linalg.norm(params)):
            status = "converged"
            break
    X = np.array([chart_forward(chart, p, cams, Rs) for p in params])
    return {"status": status, "iter": it, "X": X, "cams": cams,
            "cost": float(cost)}


# --------------------------------------------------------------------------
# points only, poses held at the motion-capture solution
# --------------------------------------------------------------------------
def solve_points_only(state, chart, max_iter=40):
    """Minimise the reprojection error over the points with the cameras fixed.

    With the camera block free, bundle adjustment can trade a point's depth
    against a small displacement of the observing poses and leave the residual
    almost unchanged -- the depth-blindness of Section~\ref{sec:depth-blind}.
    That is a real property of BA, but it is not the object of this experiment:
    here the question is whether the chart changes the depth a *given*
    geometry supports.  Fixing the poses removes the trade and leaves the point
    block alone, which is exactly what Theorem~\ref{thm:main} speaks about.
    """
    layout = cam_layout(len(state["cams"]))
    params = init_params(state, chart)
    if params is None:
        return {"status": "chart_undefined", "iter": 0, "X": None}
    cams, Rs = state["cams"], state["Rs"]
    cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
    status, it = "max_iter", 0
    for it in range(1, max_iter + 1):
        A, B, r, _, _, _, ok = assemble(cur, chart, params, layout)
        if not ok:
            return {"status": "behind_camera", "iter": it, "X": None}
        cost = 0.5 * float(r @ r)
        gp = B.T @ r
        if np.max(np.abs(gp)) < 1e-12:
            status = "converged"
            break
        try:
            dp = -np.linalg.solve(B.T @ B, gp)
        except np.linalg.LinAlgError:
            return {"status": "singular", "iter": it, "X": None}
        step, accepted = 1.0, False
        for _ in range(14):
            trial = params + step * dp.reshape(-1, 3)
            new = cost_of(cur, chart, trial)
            if np.isfinite(new) and new < cost:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "stalled"
            break
        params = trial
        if np.linalg.norm(step * dp) < 1e-12 * (1.0 + np.linalg.norm(params)):
            status = "converged"
            break
    X = np.array([chart_forward(chart, p, cams, Rs) for p in params])
    return {"status": status, "iter": it, "X": X, "cams": list(cams),
            "cost": float(cost)}


# --------------------------------------------------------------------------
# one configuration: a window plus one noise realisation, four charts
# --------------------------------------------------------------------------
def run_window(seq, rgb, gt, dep_of, idx, sigma, rng, max_points=100,
               solver="lm_phys", n_mc=1):
    gray = []
    for i in idx:
        im = cv2.imread(os.path.join(seq, rgb[i][1]), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return {"status": "read_failed"}
        gray.append(im)
    tracks = track_window(gray)
    if tracks is None:
        return {"status": "tracking_failed"}
    xn = to_normalized(tracks.reshape(-1, 2)).reshape(tracks.shape)

    cams, Rs = [], []
    for i in idx:
        R, c = pose_at(gt, rgb[i][0])
        Rs.append(R)
        cams.append(c)

    # ---- ground-truth depth at the anchor frame -------------------------
    dmap = load_depth(seq, dep_of[rgb[idx[0]][1]])
    if dmap is None:
        return {"status": "no_depth_map"}

    Xg, keep, zs = [], [], []
    for j in range(tracks.shape[0]):
        z = sample_depth(dmap, tracks[j, 0])
        if z is None:
            continue
        X = gt_point(xn[j, 0], z, Rs[0], cams[0])
        if not np.all(np.isfinite(X)):
            continue
        # GT depth and GT pose must agree with the observations, otherwise
        # either the depth edge or the pose is wrong and the "truth" is not one
        if reproj_px_one(xn[j], X, cams, Rs) > GT_REPROJ_PX:
            continue
        bg = all((Rs[i] @ (X - cams[i]))[2] > 0 for i in range(N_CAM))
        if not bg:
            continue
        Xg.append(X)
        zs.append(z)
        keep.append(j)
    if len(Xg) < 10:
        return {"status": "too_few_gt", "n_track": int(tracks.shape[0])}
    Xg = np.array(Xg)
    zs = np.array(zs)
    beta = np.array([parallax_deg(X, cams[0], cams[-1]) for X in Xg])
    crb = np.array([depth_crb(X, cams, Rs, sigma) or np.nan for X in Xg])

    # ---- noisy observations --------------------------------------------
    n_pt = len(keep)
    u0 = np.array([xn[j, i] for i in range(N_CAM) for j in keep]).ravel()
    u = u0 + rng.normal(0.0, sigma / FX, u0.shape)
    um = u.reshape(N_CAM, n_pt, 2)          # [camera, feature, xy]

    # common physical initialisation: triangulate the noisy rays
    pts = np.array([triangulate(um[:, j, :], cams, Rs) for j in range(n_pt)])

    # A linear triangulation at a small baseline can put a point behind a
    # camera.  Such a point has no chart coordinates, so the iteration never
    # starts.  Drop them, count them, and report the count: this is itself part
    # of the collapse -- all four charts fail to start on the same features.
    good = np.all(np.isfinite(pts), axis=1)
    good &= np.array([min((Rs[i] @ (p - cams[i]))[2] for i in range(N_CAM)) > 0.05
                      for p in pts])
    good &= np.array([np.linalg.norm(p - cams[0]) < 50.0 for p in pts])
    n_dropped = int((~good).sum())
    if good.sum() < 30:
        return {"status": "too_few_points", "n_dropped_init": n_dropped,
                "n_gt": n_pt, "n_kept": int(good.sum())}
    pts = pts[good]
    beta = beta[good]
    crb = crb[good]
    depth_gt = np.array([(Rs[0] @ (X - cams[0]))[2] for X in Xg])[good]
    um = um[:, good, :]

    # thin the window, stratified in beta so that the shape of the distribution
    # survives; the cost of one Gauss-Newton step grows as n_pt^3
    if len(pts) > max_points:
        order = np.argsort(beta)
        take = np.linspace(0, len(order) - 1, max_points).round().astype(int)
        sel = np.sort(order[np.unique(take)])
        pts, beta, crb, depth_gt, um = (pts[sel], beta[sel], crb[sel],
                                        depth_gt[sel], um[:, sel, :])
    u = um.reshape(-1)
    obs = [(i, j) for i in range(N_CAM) for j in range(len(pts))]

    state = {"cams": cams, "Rs": Rs, "pts": pts, "obs": obs, "u": u}
    B = float(np.linalg.norm(cams[-1] - cams[0]))

    out = {"beta": beta, "depth_gt": depth_gt, "n_dropped_init": n_dropped,
           "n_gt": n_pt, "solver": solver, "baseline_m": B, "crb": crb,
           "charts": {}}
    u_clean = um.reshape(-1)
    Z = {c: [] for c in CHARTS}
    for m in range(n_mc):
        if m == 0:
            u_m, pts_m = u, pts
        else:
            u_m = u_clean + rng.normal(0.0, sigma / FX, u_clean.shape)
            um_m = u_m.reshape(N_CAM, len(pts), 2)
            pts_m = np.array([triangulate(um_m[:, j, :], cams, Rs)
                              for j in range(len(pts))])
        st = {"cams": cams, "Rs": Rs, "pts": pts_m, "obs": obs, "u": u_m}
        for chart in CHARTS:
            t0 = time.time()
            if solver == "gn":
                res = gn_solve(st, chart)
            elif solver == "lm_chart":
                res = lm_solve(st, chart, metric="chart")
            elif solver == "pts_only":
                res = solve_points_only(st, chart)
            else:
                res = lm_solve(st, chart, metric="physical")
            el = time.time() - t0
            if res.get("X") is None:
                Z[chart].append(np.full(len(pts), np.nan))
                if m == 0:
                    out["charts"][chart] = {"status": res["status"],
                                            "iter": res["iter"]}
                continue
            X = np.array(res["X"])
            z = np.array([(Rs[0] @ (x - cams[0]))[2] for x in X])
            ok = np.isfinite(z) & (z > 1e-6)
            Z[chart].append(np.where(ok, z, np.nan))
            if m:
                continue
            rel = np.full(len(z), np.nan)
            rel[ok] = np.abs(z[ok] - depth_gt[ok]) / depth_gt[ok]
            # camera displacement away from the motion-capture solution
            dcam = max(float(np.linalg.norm(res["cams"][i] - cams[i]))
                       for i in range(2, N_CAM))
            out["charts"][chart] = {
                "status": res["status"], "iter": res["iter"], "sec": round(el, 3),
                "depth_est": z, "rel_err": rel, "n_flipped": int((~ok).sum()),
                "reprojection_rmse": math.sqrt(2.0 * res["cost"] / len(u)) * FX,
                "cam_drift_m": dcam,
            }
    # Monte-Carlo spread: the systematic part of the depth error (Kinect bias,
    # pose bias, tracker bias) is common to all realisations and cancels in the
    # standard deviation, which leaves the geometric sensitivity alone.
    out["mc"] = {}
    for chart in CHARTS:
        if not Z[chart]:
            continue
        A = np.vstack(Z[chart])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(A, axis=0)
            sd = (np.nanstd(A, axis=0, ddof=1) if n_mc > 1
                  else np.full(len(pts), np.nan))
        out["mc"][chart] = {"mean": mean, "std": sd,
                            "rel_std": sd / np.where(np.abs(mean) > 1e-9, mean, np.nan),
                            "n_valid": np.sum(np.isfinite(A), axis=0)}
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
def summarize(records, strict=True):
    """Paired differences and beta-binned medians.

    strict=True keeps only the windows on which every chart produced a depth,
    so that the four curves are drawn over the same set of features.  Without
    that, a chart that diverges early silently drops its hard cases and looks
    better than it is.
    """
    per = {c: {"beta": [], "err": [], "pred": [], "status": [], "pairv": [],
               "relstd": []} for c in CHARTS}
    pair = {c: [] for c in CHARTS}
    status = {c: {} for c in CHARTS}
    kept, n_flip = 0, {c: 0 for c in CHARTS}

    for rec in records:
        if rec.get("status") != "ok":
            continue
        for c in CHARTS:
            st = rec["charts"].get(c, {}).get("status", rec.get("status"))
            status[c][st] = status[c].get(st, 0) + 1
        if strict and not all("rel_err" in rec["charts"].get(c, {}) for c in CHARTS):
            continue
        kept += 1
        # first-order propagation of the tracking noise through the triangle,
        # from the Fisher information of this point's own eight rays.  No chart
        # appears in it; rotation enters as well as translation.
        pred = np.asarray(rec.get("crb"), float)
        ref = rec["charts"].get("xyz", {})
        for c in CHARTS:
            r = rec["charts"].get(c, {})
            if "rel_err" not in r:
                continue
            per[c]["beta"].append(rec["beta"])
            per[c]["err"].append(r["rel_err"])
            per[c]["pred"].append(pred)
            per[c]["status"].append(r["status"])
            rs = rec.get("mc", {}).get(c, {}).get("rel_std")
            per[c]["relstd"].append(rs if rs is not None
                                    else np.full(len(rec["beta"]), np.nan))
            n_flip[c] += int(r.get("n_flipped", 0))
        for c in CHARTS:
            r = rec["charts"].get(c, {})
            if "depth_est" not in r or "depth_est" not in ref:
                continue
            a = np.asarray(r["depth_est"], float)
            b = np.asarray(ref["depth_est"], float)
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
            if not np.any(ok):
                continue
            scale = np.maximum(np.abs(b[ok]), 1e-30)
            pair[c].append(float(np.max(np.abs(a[ok] - b[ok]) / scale)))
            pv = np.full(a.shape, np.nan)
            pv[ok] = np.abs(a[ok] - b[ok]) / scale
            per[c]["pairv"].append(pv)

    edges = np.concatenate([[0.0], np.geomspace(0.15, 30.0, N_BIN - 1), [90.0]])
    pedges = np.arange(-5.0, 0.51, 0.5)
    out = {"bins_deg": edges.tolist(), "bins_pred_decades": pedges.tolist(),
           "curves": {}, "curves_pred": {}, "n_points": {}, "pair": {},
           "status": status, "n_windows_used": kept, "n_flipped": n_flip,
           "calibration": {}}
    for c in CHARTS:
        beta = np.concatenate(per[c]["beta"]) if per[c]["beta"] else np.array([])
        err = np.concatenate(per[c]["err"]) if per[c]["err"] else np.array([])
        pred = np.concatenate(per[c]["pred"]) if per[c]["pred"] else np.array([])
        out["n_points"][c] = int(beta.size)
        if beta.size == 0:
            continue
        med, q1, q3, n = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (beta >= lo) & (beta < hi) & np.isfinite(err)
            if m.sum() < 3:
                med.append(None); q1.append(None); q3.append(None); n.append(0)
                continue
            med.append(float(np.median(err[m])))
            q1.append(float(np.percentile(err[m], 25)))
            q3.append(float(np.percentile(err[m], 75)))
            n.append(int(m.sum()))
        out["curves"][c] = {"median": med, "q1": q1, "q3": q3, "n": n}

        # the same statistic against the geometric prediction
        lp = np.log10(np.maximum(pred, 1e-12))
        pv = (np.concatenate(per[c]["pairv"]) if per[c]["pairv"]
              else np.full(err.shape, np.nan))
        ok = np.isfinite(err) & np.isfinite(lp)
        pm, pn, pp = [], [], []
        for lo, hi in zip(pedges[:-1], pedges[1:]):
            m = ok & (lp >= lo) & (lp < hi)
            pm.append(float(np.median(err[m])) if m.sum() >= 3 else None)
            pn.append(int(m.sum()))
            pvv = pv[m] if pv.shape[0] == err.shape[0] else np.array([np.nan])
            pvv = pvv[np.isfinite(pvv)]
            pp.append(float(np.max(pvv)) if pvv.size else None)
        out["curves_pred"][c] = {"median": pm, "n": pn, "pairmax": pp}

        # Monte-Carlo depth sensitivity against the same prediction
        rs = (np.concatenate(per[c]["relstd"]) if per[c]["relstd"]
              else np.full(err.shape, np.nan))
        sm, sn = [], []
        for lo, hi in zip(pedges[:-1], pedges[1:]):
            m = ok & (lp >= lo) & (lp < hi) & np.isfinite(rs)
            sm.append(float(np.median(rs[m])) if m.sum() >= 3 else None)
            sn.append(int(m.sum()))
        out.setdefault("curves_mc", {})[c] = {"median": sm, "n": sn}
        rr = rs[ok] / pred[ok]
        rr = rr[np.isfinite(rr)]
        out.setdefault("mc_ratio", {})[c] = {
            "median": float(np.median(rr)) if rr.size else None,
            "q25": float(np.percentile(rr, 25)) if rr.size else None,
            "q75": float(np.percentile(rr, 75)) if rr.size else None}
        # measured / predicted: one number per chart, the "calibration" of the
        # first-order formula against this sequence
        r = err[ok] / pred[ok]
        r = r[np.isfinite(r)]
        out["calibration"][c] = {"median": float(np.median(r)) if r.size else None,
                                 "q25": float(np.percentile(r, 25)) if r.size else None,
                                 "q75": float(np.percentile(r, 75)) if r.size else None}

        p = np.array(pair[c]) if pair[c] else np.array([0.0])
        out["pair"][c] = {"max": float(np.max(p)), "median": float(np.median(p)),
                          "n_windows": int(p.size)}
    # collapse metric: spread of the four medians inside each bin
    spread = []
    for k in range(len(pedges) - 1):
        vals = [out["curves_pred"][c]["median"][k] for c in CHARTS
                if out["curves_pred"].get(c) and out["curves_pred"][c]["median"][k]]
        if len(vals) >= 2:
            spread.append(float((max(vals) - min(vals)) / max(max(vals), 1e-30)))
    out["cross_chart_spread_max"] = max(spread) if spread else None
    return out


# --------------------------------------------------------------------------
def report(s, tag):
    print("\n=== %s ===" % tag)
    print("  %-4s %6s %10s %10s  %s" %
          ("chart", "pts", "pair max", "pair med", "status"))
    for c in CHARTS:
        cu = s["curves"].get(c)
        if not cu:
            print("  %-4s  (no point)" % c)
            continue
        st = ", ".join("%s:%d" % kv for kv in sorted(s["status"][c].items()))
        print("  %-4s %6d %10.2e %10.2e  %s"
              % (c, s["n_points"][c], s["pair"][c]["max"], s["pair"][c]["median"], st))
    print("\n  beta bin (deg) |  median relative depth error")
    edges = s["bins_deg"]
    for k in range(len(edges) - 1):
        row = []
        for c in CHARTS:
            v = s["curves"].get(c, {}).get("median", [None] * 99)[k]
            row.append("   --   " if v is None else "%.2e" % v)
        n = s["curves"].get("xyz", {}).get("n", [0] * 99)[k]
        print("  %5.2f-%6.2f n=%-4d | %s" % (edges[k], edges[k + 1], n, " ".join(row)))
    print("\n  sigma_px / (f tan beta)  |  median relative depth error")
    pe = s["bins_pred_decades"]
    for k in range(len(pe) - 1):
        row = []
        for c in CHARTS:
            v = s["curves_pred"].get(c, {}).get("median", [None] * 99)[k]
            row.append("   --   " if v is None else "%.2e" % v)
        n = s["curves_pred"].get("xyz", {}).get("n", [0] * 99)[k]
        print("  1e%+.1f..1e%+.1f   n=%-4d | %s" % (pe[k], pe[k + 1], n, " ".join(row)))
    print("  measured/predicted, per chart:")
    for c in CHARTS:
        cc = s["calibration"].get(c)
        if cc and cc["median"] is not None:
            print("    %-4s  %.2f  [%.2f, %.2f]" % (c, cc["median"], cc["q25"], cc["q75"]))
    print("  cross-chart spread of binned medians: %.3e"
          % (s["cross_chart_spread_max"] or 0.0))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seq", default=SEQ)
    args = ap.parse_args()

    if cv2 is None:
        raise SystemExit("OpenCV is required")
    seq = args.seq
    rgb = read_index(os.path.join(seq, "rgb.txt"))
    gt = read_gt(os.path.join(seq, "groundtruth.txt"))
    dep = read_depth_index(seq)
    dep_of = associate(rgb, dep)
    print("sequence :", seq)
    print("frames   : %d   gt %d   depth %d   associated %d"
          % (len(rgb), len(gt), len(dep), len(dep_of)))

    # the main solver carries the sweep; the others are ablations on a subset
    plan = {"pts_only": ([1, 2, 3, 4, 6] if not args.quick else [1, 4],
                         4 if not args.quick else 1),
            "lm_phys": ([2, 6] if not args.quick else [1],
                        2 if not args.quick else 1),
            "lm_chart": ([2, 6] if not args.quick else [1],
                         2 if not args.quick else 1),
            "gn": ([2, 6] if not args.quick else [1],
                   2 if not args.quick else 1)}
    solvers = ["pts_only"] if args.quick else ["pts_only", "lm_phys", "lm_chart",
                                               "gn"]
    strides = plan[solvers[0]][0]
    rng = np.random.default_rng(20261104)

    records = []
    t_start = time.time()
    for solver in solvers:
        s_strides, n_win = plan[solver]
        print("\n-- solver %s  (strides %s) --" % (solver, s_strides), flush=True)
        for stride in s_strides:
            span = stride * (N_CAM - 1)
            starts = [int(x) for x in
                      np.linspace(0, max(0, len(rgb) - span - 2), n_win + 1)][:n_win]
            for s0 in starts:
                idx = [s0 + k * stride for k in range(N_CAM)]
                if rgb[idx[0]][1] not in dep_of or not all(i < len(rgb) for i in idx):
                    records.append({"status": "no_depth", "solver": solver})
                    continue
                t0 = time.time()
                rec = run_window(seq, rgb, gt, dep_of, idx, SIGMA_PX, rng,
                                 solver=solver,
                                 n_mc=(N_MC if solver == "pts_only" else 1))
                rec["stride"] = stride
                rec["start"] = s0
                rec["solver"] = solver
                records.append(rec)
                if rec.get("status") == "ok":
                    xr = rec["charts"].get("xyz", {})
                    med = (float(np.nanmedian(xr["rel_err"]))
                           if "rel_err" in xr else float("nan"))
                    okc = sum(1 for c in CHARTS
                              if "rel_err" in rec["charts"].get(c, {}))
                    print("  stride %d start %4d  n=%3d (drop %2d)  beta %.2f-%.2f  "
                          "ok %d/4  err(xyz) %.2e  %.1fs"
                          % (stride, s0, len(rec["beta"]), rec["n_dropped_init"],
                             np.min(rec["beta"]), np.max(rec["beta"]), okc, med,
                             time.time() - t0), flush=True)
                else:
                    print("  stride %d start %4d  %s  %.1fs"
                          % (stride, s0, rec.get("status"), time.time() - t0),
                          flush=True)

    print("\ntotal %.1f s" % (time.time() - t_start))
    out = {"sequence": os.path.basename(seq), "sigma_px": SIGMA_PX,
           "n_cam": N_CAM, "strides": strides, "solvers": {}}
    for solver in solvers:
        recs = [r for r in records if r.get("solver") == solver]
        ok = [r for r in recs if r.get("status") == "ok"]
        s = summarize(ok)
        s["n_windows_ok"] = len(ok)
        s["n_windows"] = len(recs)
        s["n_dropped_init"] = int(sum(r.get("n_dropped_init", 0) for r in ok))
        s["n_gt_candidates"] = int(sum(r.get("n_gt", 0) for r in ok))
        out["solvers"][solver] = s
        report(s, solver)

    # per-point data only for the main solver: the other two are ablations
    out["windows"] = []
    for r in records:
        if r.get("solver") != solvers[0]:
            continue
        if r.get("status") != "ok":
            out["windows"].append({"stride": r.get("stride"), "start": r.get("start"),
                                   "status": r.get("status")})
            continue
        w = {"stride": r["stride"], "start": r["start"], "status": "ok",
             "beta": r["beta"].tolist(), "depth_gt": r["depth_gt"].tolist(),
             "crb": np.asarray(r["crb"]).tolist(),
             "baseline_m": r["baseline_m"],
             "n_dropped_init": r["n_dropped_init"], "charts": {}}
        for c, v in r["charts"].items():
            w["charts"][c] = {k: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                              for k, vv in v.items()}
        w["mc"] = {c: {"rel_std": v["rel_std"].tolist()}
                   for c, v in r.get("mc", {}).items()}
        out["windows"].append(w)

    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, default=float)
    print("\nwrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
