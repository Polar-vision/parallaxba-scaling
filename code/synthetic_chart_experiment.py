#!/usr/bin/env python3
"""Controlled synthetic study of point-chart parameterizations in BA.

The companion theorem says that on a regular chart the physical Gauss-Newton
step, the Fisher information, and the attainable minima are chart invariants,
whereas the damping metric and the domain are not.  This script tests that
statement on synthetic data with a known ground truth.

Gauge.  Bearing-only BA has a seven-dimensional gauge group: six rigid-body
degrees of freedom plus one global scale.  Camera 0 is held fixed (six
coordinates) and so is the translation of camera 1 (three coordinates): nine
coordinates in all, the seven gauge freedoms plus the two directions of the
first baseline, which are held at their true values.  Every chart is
therefore compared at the same physical operating point.

Experiments
    S1  undamped GN, regular geometry: physical iterates must coincide
    S2  alignment: what "convergence" means when depth is unobservable
    S3  angle sweep: conditioning, step size, iterations, final depth error
    S4  isotropic per-chart damping versus the pullback metric T^{-T}T^{-1}
    S5  depth blindness of the reprojection metric

Charts
    xyz  global Euclidean point
    id   anchored inverse depth, optical-axis form (Civera-type)
    ir   anchored inverse range, world bearing
    pa   parallax angle with main/associate anchors (baseline-adapted
         coordinates: angle to and azimuth about the baseline)
    po   parallax angle in the original coordinates of Zhao et al. (2015),
         Eqs. (2)-(9): azimuth/elevation of the main bearing in the frame of
         the first pose, plus omega.  Not part of CHARTS; used by
         original_chart_experiment.py.

Only numpy is used.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "synthetic_chart_experiment.json")

CHARTS = ("xyz", "id", "ir", "pa")
ANCHOR = 0
ASSOCIATE = -1


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def expm_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def unit(v):
    return v / np.linalg.norm(v)


def dproj(y):
    iz = 1.0 / y[2]
    return np.array([[iz, 0.0, -y[0] * iz * iz],
                     [0.0, iz, -y[1] * iz * iz]])


def transverse_frame(e1):
    a = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = unit(a - np.dot(a, e1) * e1)
    e3 = np.cross(e1, e2)
    return e2, e3


# --------------------------------------------------------------------------
# chart maps: p -> X, with dX/dp and dX/d(anchor pose)
# a pose increment is ordered (rotation, translation)
# --------------------------------------------------------------------------
def chart_forward(chart, p, cams, Rs):
    if chart == "xyz":
        return np.asarray(p, dtype=float).copy()
    ca, Ra = cams[ANCHOR], Rs[ANCHOR]
    if chart == "id":
        u, v, rho = p
        return ca + (1.0 / rho) * (Ra.T @ np.array([u, v, 1.0]))
    if chart == "ir":
        th, ph, rho = p
        n = np.array([math.cos(ph) * math.cos(th),
                      math.cos(ph) * math.sin(th),
                      math.sin(ph)])
        return ca + (1.0 / rho) * n
    if chart == "pa":
        al, be, om = p
        cb = cams[ASSOCIATE]
        b = cb - ca
        B = np.linalg.norm(b)
        e1 = b / B
        e2, e3 = transverse_frame(e1)
        n = (math.cos(be) * e1
             + math.sin(be) * (math.cos(al) * e2 + math.sin(al) * e3))
        d = B * math.sin(be + om) / math.sin(om)
        return ca + d * n
    if chart == "po":
        ps, th, om = p
        cb = cams[ASSOCIATE]
        b = cb - ca
        B = np.linalg.norm(b)
        n = Ra.T @ np.array([math.sin(ps) * math.cos(th), math.sin(th),
                             math.cos(ps) * math.cos(th)])
        phi = math.acos(max(-1.0, min(1.0, float(np.dot(n, b / B)))))
        d = B * math.sin(phi + om) / math.sin(om)
        return ca + d * n
    raise ValueError(chart)


def chart_jacobian(chart, p, cams, Rs):
    """Return (X, dX/dp, [(camera, dX/dpose 3x6), ...])."""
    if chart == "xyz":
        return np.asarray(p, dtype=float).copy(), np.eye(3), []
    ca, Ra = cams[ANCHOR], Rs[ANCHOR]
    if chart == "id":
        u, v, rho = p
        m = np.array([u, v, 1.0])
        X = ca + (1.0 / rho) * (Ra.T @ m)
        Jx = np.column_stack([(1.0 / rho) * (Ra.T @ np.array([1.0, 0.0, 0.0])),
                              (1.0 / rho) * (Ra.T @ np.array([0.0, 1.0, 0.0])),
                              -(1.0 / rho ** 2) * (Ra.T @ m)])
        G = np.hstack([(1.0 / rho) * (Ra.T @ skew(m)), np.eye(3)])
        return X, Jx, [(ANCHOR, G)]
    if chart == "ir":
        th, ph, rho = p
        n = np.array([math.cos(ph) * math.cos(th),
                      math.cos(ph) * math.sin(th),
                      math.sin(ph)])
        X = ca + (1.0 / rho) * n
        dth = np.array([-math.cos(ph) * math.sin(th),
                        math.cos(ph) * math.cos(th), 0.0])
        dph = np.array([-math.sin(ph) * math.cos(th),
                        -math.sin(ph) * math.sin(th), math.cos(ph)])
        Jx = np.column_stack([(1.0 / rho) * dth, (1.0 / rho) * dph,
                              -(1.0 / rho ** 2) * n])
        G = np.hstack([np.zeros((3, 3)), np.eye(3)])
        return X, Jx, [(ANCHOR, G)]
    if chart == "pa":
        al, be, om = p
        cb = cams[ASSOCIATE]
        b = cb - ca
        B = np.linalg.norm(b)
        e1 = b / B
        e2, e3 = transverse_frame(e1)
        n = (math.cos(be) * e1
             + math.sin(be) * (math.cos(al) * e2 + math.sin(al) * e3))
        u = (-math.sin(be) * e1
             + math.cos(be) * (math.cos(al) * e2 + math.sin(al) * e3))
        w = math.sin(be) * (-math.sin(al) * e2 + math.cos(al) * e3)
        d = B * math.sin(be + om) / math.sin(om)
        db = B * math.cos(be + om) / math.sin(om)
        dw = -B * math.sin(be) / math.sin(om) ** 2
        X = ca + d * n
        Jx = np.column_stack([d * w, db * n + d * u, dw * n])
        h = 1e-7
        blocks = []
        for cam in (ANCHOR, len(cams) - 1):
            Gt = np.zeros((3, 3))
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                cp = [c.copy() for c in cams]
                cm = [c.copy() for c in cams]
                cp[cam] = cp[cam] + e
                cm[cam] = cm[cam] - e
                Gt[:, k] = (chart_forward(chart, p, cp, Rs)
                            - chart_forward(chart, p, cm, Rs)) / (2 * h)
            blocks.append((cam, np.hstack([np.zeros((3, 3)), Gt])))
        return X, Jx, blocks
    if chart == "po":
        # X = c_m + d n(psi, theta), d = B sin(phi + omega) / sin(omega),
        # phi = arccos(n . b_hat): phi is the geodesic distance from b_hat, so
        # d is not differentiable at phi = 0 (a cone), and dphi/dn has unit norm.
        ps, th, om = p
        cb = cams[ASSOCIATE]
        b = cb - ca
        B = np.linalg.norm(b)
        bh = b / B
        sp, cp, st, ct = math.sin(ps), math.cos(ps), math.sin(th), math.cos(th)
        n = Ra.T @ np.array([sp * ct, st, cp * ct])
        n_ps = Ra.T @ np.array([cp * ct, 0.0, -sp * ct])
        n_th = Ra.T @ np.array([-sp * st, ct, -cp * st])
        c = max(-1.0, min(1.0, float(np.dot(n, bh))))
        phi = math.acos(c)
        s_ = math.sin(phi)
        d = B * math.sin(phi + om) / math.sin(om)
        dd_phi = B * math.cos(phi + om) / math.sin(om)
        dd_om = -B * s_ / math.sin(om) ** 2
        dphi_ps = -float(np.dot(bh, n_ps)) / s_
        dphi_th = -float(np.dot(bh, n_th)) / s_
        X = ca + d * n
        Jx = np.column_stack([d * n_ps + dd_phi * dphi_ps * n,
                              d * n_th + dd_phi * dphi_th * n,
                              dd_om * n])
        eperp = (n - c * bh) / s_
        gd = (d / B) * bh - (dd_phi / B) * eperp          # d(d)/d(c_associate)
        # rotation of the anchor frame, R <- exp([w]x) R, gives dn = R^T [m]x dw,
        # and the sine law moves d through phi = arccos(n . b_hat)
        m = np.array([sp * ct, st, cp * ct])
        Grot = (d * np.eye(3) - (dd_phi / s_) * np.outer(n, bh)) @ (Ra.T @ skew(m))
        blocks = [(ANCHOR, np.hstack([Grot, np.eye(3) - np.outer(n, gd)])),
                  (len(cams) - 1, np.hstack([np.zeros((3, 3)),
                                             np.outer(n, gd)]))]
        return X, Jx, blocks
    raise ValueError(chart)


def chart_inverse(chart, X, cams, Rs):
    if chart == "xyz":
        return np.asarray(X, dtype=float).copy()
    ca, Ra = cams[ANCHOR], Rs[ANCHOR]
    if chart == "id":
        y = Ra @ (X - ca)
        return np.array([y[0] / y[2], y[1] / y[2], 1.0 / y[2]])
    if chart == "ir":
        r = X - ca
        d = np.linalg.norm(r)
        n = r / d
        return np.array([math.atan2(n[1], n[0]),
                         math.atan2(n[2], math.hypot(n[0], n[1])), 1.0 / d])
    if chart == "pa":
        cb = cams[ASSOCIATE]
        b = cb - ca
        B = np.linalg.norm(b)
        e1 = b / B
        e2, e3 = transverse_frame(e1)
        r = X - ca
        d = np.linalg.norm(r)
        n = r / d
        cb_ = float(np.dot(n, e1))
        perp = n - cb_ * e1
        be = math.atan2(np.linalg.norm(perp), cb_)
        al = math.atan2(float(np.dot(perp, e3)), float(np.dot(perp, e2)))
        ra = X - cb
        cross = np.linalg.norm(np.cross(r, ra))
        om = math.atan2(cross, float(np.dot(r, ra)))
        return np.array([al, be, om])
    if chart == "po":
        cb = cams[ASSOCIATE]
        r = X - ca
        nc = Ra @ (r / np.linalg.norm(r))
        ra = X - cb
        om = math.atan2(float(np.linalg.norm(np.cross(r, ra))),
                        float(np.dot(r, ra)))
        return np.array([math.atan2(nc[0], nc[2]),
                         math.atan2(nc[1], math.hypot(nc[0], nc[2])), om])
    raise ValueError(chart)


# --------------------------------------------------------------------------
# gauge layout: camera 0 fixed, camera 1 rotation only (known baseline)
# --------------------------------------------------------------------------
def cam_layout(n_cam):
    entries = []
    col = 0
    for i in range(1, n_cam):
        dofs = (0, 1, 2) if i == 1 else (0, 1, 2, 3, 4, 5)
        for d in dofs:
            entries.append((i, d, col))
            col += 1
    per_cam = {}
    for i, d, c in entries:
        per_cam.setdefault(i, []).append((d, c))
    return entries, col, per_cam


# --------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------
def _project(R, c, X):
    y = R @ (X - c)
    return y, np.array([y[0] / y[2], y[1] / y[2]])


def scene_regular(n_cam=6, n_pt=24, spacing=0.5, depth=5.0, seed=3, noise=0.0):
    """Sideways rig: transverse baseline, beta close to pi/2."""
    rng = np.random.default_rng(seed)
    cams = [np.array([k * spacing, 0.0, 0.0]) for k in range(n_cam)]
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    Rs = [R.copy() for _ in range(n_cam)]
    pts = []
    while len(pts) < n_pt:
        pts.append(np.array([rng.uniform(-2.5, 2.5),
                             depth + rng.uniform(-1.0, 1.0),
                             rng.uniform(-2.0, 2.0)]))
    obs, u = [], []
    for i in range(n_cam):
        for j in range(n_pt):
            y, uv = _project(Rs[i], cams[i], pts[j])
            if y[2] <= 0.1:
                continue
            if noise > 0:
                uv = uv + rng.normal(scale=noise, size=2)
            obs.append((i, j))
            u.append(uv)
    return {"cams": cams, "Rs": Rs, "pts": pts, "obs": obs,
            "u": np.concatenate(u), "probe": len(pts) - 1}


def scene_forward(n_cam=6, n_bg=20, spacing=0.4, probe_dir=None,
                  probe_depth=12.0, seed=5, noise=0.0):
    """Forward motion along +z; background points stay in front of camera 0."""
    rng = np.random.default_rng(seed)
    cams = [np.array([0.0, 0.0, k * spacing]) for k in range(n_cam)]
    Rs = [np.eye(3) for _ in range(n_cam)]
    base = cams[-1]
    pts = []
    for _ in range(n_bg):
        r = rng.uniform(6.0, 16.0)
        th = rng.uniform(0.0, 2 * math.pi)
        el = math.radians(rng.uniform(15.0, 40.0))
        pts.append(base + r * np.array([math.cos(el) * math.cos(th),
                                        math.cos(el) * math.sin(th),
                                        math.sin(el)]))
    if probe_dir is None:
        probe_dir = np.array([0.0, 0.0, 1.0])
    probe = probe_depth * unit(probe_dir)
    pts.append(probe)
    obs, u = [], []
    for i in range(n_cam):
        for j in range(len(pts)):
            y, uv = _project(Rs[i], cams[i], pts[j])
            if y[2] <= 0.1:
                continue
            if noise > 0:
                uv = uv + rng.normal(scale=noise, size=2)
            obs.append((i, j))
            u.append(uv)
    return {"cams": cams, "Rs": Rs, "pts": pts, "obs": obs,
            "u": np.concatenate(u), "probe": len(pts) - 1}


def perturb_shared(prob, rng, rel=0.05):
    pts = [X + rng.normal(scale=rel * max(1.0, np.linalg.norm(X)), size=3)
           for X in prob["pts"]]
    return {"cams": [c.copy() for c in prob["cams"]],
            "Rs": [R.copy() for R in prob["Rs"]],
            "pts": pts, "obs": list(prob["obs"]), "u": prob["u"].copy(),
            "probe": prob.get("probe")}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def assemble(state, chart, params, layout):
    entries, n_cv, per_cam = layout
    cams, Rs, obs, u = state["cams"], state["Rs"], state["obs"], state["u"]
    K, n_pt = len(obs), len(params)
    A = np.zeros((2 * K, n_cv))
    B = np.zeros((2 * K, 3 * n_pt))
    r = np.zeros(2 * K)
    Xs, Jxs, Gs = [], [], []
    for j in range(n_pt):
        X, Jx, gb = chart_jacobian(chart, params[j], cams, Rs)
        Xs.append(X)
        Jxs.append(Jx)
        Gs.append(gb)
    ok = True
    zero6 = np.zeros((3, 6))
    for k, (i, j) in enumerate(obs):
        y, uv = _project(Rs[i], cams[i], Xs[j])
        if y[2] <= 1e-9:
            ok = False
        P = dproj(y)
        r[2 * k:2 * k + 2] = uv - u[2 * k:2 * k + 2]
        B[2 * k:2 * k + 2, 3 * j:3 * j + 3] = P @ Rs[i] @ Jxs[j]
        # d u / d(observing pose) plus, for every anchor of this point,
        # d u / d(anchor pose) = P R_i G, whichever camera observes it
        blocks = {i: np.hstack([-skew(y), -Rs[i]])}
        for cam, G in Gs[j]:
            if cam in per_cam:
                blocks[cam] = blocks.get(cam, zero6.copy()) + Rs[i] @ G
        for cam, blk in blocks.items():
            if cam not in per_cam:
                continue
            for d, col in per_cam[cam]:
                A[2 * k:2 * k + 2, col] = P @ blk[:, d]
    return A, B, r, Xs, Jxs, Gs, ok


def build_T(chart, params, cams, Rs, layout, n_pt):
    entries, n_cv, per_cam = layout
    T = np.zeros((n_cv + 3 * n_pt, n_cv + 3 * n_pt))
    T[:n_cv, :n_cv] = np.eye(n_cv)
    for j in range(n_pt):
        _, Jx, gb = chart_jacobian(chart, params[j], cams, Rs)
        rows = slice(n_cv + 3 * j, n_cv + 3 * j + 3)
        T[rows, n_cv + 3 * j:n_cv + 3 * j + 3] = Jx
        for cam, G in gb:
            if cam in per_cam:
                for d, col in per_cam[cam]:
                    T[rows, col] += G[:, d]
    return T


def apply_step(cams, Rs, params, layout, dc, dp):
    entries, n_cv, _ = layout
    n_cam = len(cams)
    rot = [np.zeros(3) for _ in range(n_cam)]
    tra = [np.zeros(3) for _ in range(n_cam)]
    for cam, d, col in entries:
        if d < 3:
            rot[cam][d] = dc[col]
        else:
            tra[cam][d - 3] = dc[col]
    cams = [cams[i] + tra[i] for i in range(n_cam)]
    Rs = [expm_so3(rot[i]) @ Rs[i] for i in range(n_cam)]
    params = np.asarray(params, dtype=float) + dp.reshape(-1, 3)
    return cams, Rs, params


def physical_step(chart, params, dp, cams, Rs, layout=None, dc=None):
    """Map a chart step to physical point displacements, including dX/d(anchor)."""
    per_cam = layout[2] if layout is not None else {}
    out = []
    for j, p in enumerate(params):
        _, Jx, gb = chart_jacobian(chart, p, cams, Rs)
        dX = Jx @ dp[3 * j:3 * j + 3]
        if dc is not None:
            for cam, G in gb:
                if cam in per_cam:
                    for d, col in per_cam[cam]:
                        dX = dX + G[:, d] * dc[col]
        out.append(dX)
    return np.array(out)


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def schur_solve(H, g, n_cv, n_pt, damp=None):
    Hcc, Hcp = H[:n_cv, :n_cv], H[:n_cv, n_cv:]
    Hpp, gc, gp = H[n_cv:, n_cv:], g[:n_cv], g[n_cv:]
    if damp is not None:
        Hcc = Hcc + damp[:n_cv, :n_cv]
        Hcp = Hcp + damp[:n_cv, n_cv:]
        Hpp = Hpp + damp[n_cv:, n_cv:]
    try:
        Hpp_inv_gp = np.linalg.solve(Hpp, gp)
        Hpp_inv_Hpc = np.linalg.solve(Hpp, Hcp.T)
        S = Hcc - Hcp @ Hpp_inv_Hpc
        dc = np.linalg.solve(S, -(gc - Hcp @ Hpp_inv_gp))
        dp = -(Hpp_inv_gp + Hpp_inv_Hpc @ dc)
        return dc, dp, float(np.linalg.cond(S))
    except np.linalg.LinAlgError:
        return None, None, float("inf")


def cost_of(state, chart, params):
    cams, Rs, obs, u = state["cams"], state["Rs"], state["obs"], state["u"]
    r = np.zeros_like(u)
    for k, (i, j) in enumerate(obs):
        X = chart_forward(chart, params[j], cams, Rs)
        y, uv = _project(Rs[i], cams[i], X)
        if y[2] <= 0:
            return float("inf")
        r[2 * k:2 * k + 2] = uv - u[2 * k:2 * k + 2]
    return 0.5 * float(r @ r)


def init_params(state, chart):
    try:
        p = np.array([chart_inverse(chart, X, state["cams"], state["Rs"])
                      for X in state["pts"]], dtype=float)
    except (ZeroDivisionError, ValueError, FloatingPointError):
        return None
    if not np.all(np.isfinite(p)):
        return None
    if chart in ("id", "ir") and np.any(p[:, 2] <= 0):
        return None
    if chart in ("pa", "po") and np.any(np.abs(np.sin(p[:, 2])) < 1e-13):
        return None
    return p


def solve(state, chart, method="lm", metric="chart", max_iter=100, lam0=1e-4):
    layout = cam_layout(len(state["cams"]))
    n_cv = layout[1]
    params = init_params(state, chart)
    if params is None:
        return {"status": "chart_undefined", "iter": 0}
    cams = [c.copy() for c in state["cams"]]
    Rs = [R.copy() for R in state["Rs"]]
    cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
    cost = cost_of(cur, chart, params)
    if not np.isfinite(cost):
        return {"status": "invalid_init", "iter": 0}
    lam = lam0
    n_it = 0
    status = "max_iter"
    for _ in range(max_iter):
        A, B, r, _, _, _, ok = assemble(cur, chart, params, layout)
        if not ok:
            status = "behind_camera"
            break
        H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
        g = np.concatenate([A.T @ r, B.T @ r])
        if np.max(np.abs(g)) < 1e-11:
            status = "converged"
            break
        if method == "gn":
            dc, dp, _ = schur_solve(H, g, n_cv, len(params))
            if dc is None:
                status = "singular"
                break
            cams, Rs, params = apply_step(cams, Rs, params, layout, dc, dp)
            cur = {"cams": cams, "Rs": Rs, "obs": state["obs"], "u": state["u"]}
            cost = cost_of(cur, chart, params)
            n_it += 1
            continue
        accepted = False
        for _ in range(30):
            if metric == "chart":
                damp = lam * np.eye(H.shape[0])
            else:
                T = build_T(chart, params, cams, Rs, layout, len(params))
                Ti = np.linalg.inv(T)
                damp = lam * (Ti.T @ Ti)
            dc, dp, _ = schur_solve(H, g, n_cv, len(params), damp)
            if dc is None:
                lam *= 10.0
                continue
            nc, nR, np_ = apply_step(cams, Rs, params, layout, dc, dp)
            trial = {"cams": nc, "Rs": nR, "obs": state["obs"], "u": state["u"]}
            new_cost = cost_of(trial, chart, np_)
            if np.isfinite(new_cost) and new_cost < cost:
                cams, Rs, params, cost, cur = nc, nR, np_, new_cost, trial
                accepted = True
                lam = max(lam / 3.0, 1e-12)
                n_it += 1
                break
            lam *= 10.0
        if not accepted:
            status = "stalled"
            break
        if np.linalg.norm(dc) + np.linalg.norm(dp) < 1e-13:
            status = "converged"
            break
    Xs = np.array([chart_forward(chart, p, cams, Rs) for p in params])
    # gradient and damping at exit, in chart units: the "converged" flag above is
    # set by a step-norm test, which says nothing about either of them
    gmax = None
    A, B, r, _, _, _, ok = assemble(cur, chart, params, layout)
    if ok:
        gmax = float(np.max(np.abs(np.concatenate([A.T @ r, B.T @ r]))))
    return {"status": status, "iter": n_it, "cost": cost, "params": params,
            "X": Xs, "cams": cams, "Rs": Rs, "lam": lam, "gmax": gmax}


def summarize(res, probe, dtrue, n_res):
    out = {"status": res["status"], "iter": res["iter"]}
    if res.get("X") is None:
        return out
    Xp = res["X"][probe]
    out["cost"] = res["cost"]
    out["reprojection_rmse"] = math.sqrt(2.0 * res["cost"] / n_res)
    dhat = float(np.linalg.norm(Xp))
    out["depth_estimate"] = dhat
    out["depth_rel_error"] = abs(dhat - dtrue) / dtrue
    return out

# --------------------------------------------------------------------------
# Cramer-Rao bound at a given linearization point
# --------------------------------------------------------------------------
NOISE = 1e-3          # standard deviation per image coordinate
FOCAL = 500.0         # pixels, used only to quote reprojection errors


def position_map(chart, params, cams, Rs, layout):
    """U with X = U [camera variables; point parameters]."""
    n_cv = layout[1]
    per_cam = layout[2]
    n_pt = len(params)
    U = np.zeros((3 * n_pt, n_cv + 3 * n_pt))
    for j, p in enumerate(params):
        _, Jx, gb = chart_jacobian(chart, p, cams, Rs)
        U[3 * j:3 * j + 3, n_cv + 3 * j:n_cv + 3 * j + 3] = Jx
        for cam, G in gb:
            if cam in per_cam:
                for d, col in per_cam[cam]:
                    U[3 * j:3 * j + 3, col] += G[:, d]
    return U


def hessian_at(chart, state, params, layout):
    A, B, r, _, _, _, ok = assemble(state, chart, params, layout)
    H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
    return H, np.concatenate([A.T @ r, B.T @ r]), ok


def crb(chart, state, layout, probe, direction):
    """Marginal standard deviation of the probe depth on the given chart."""
    p = init_params(state, chart)
    if p is None:
        return None
    H, g, ok = hessian_at(chart, state, p, layout)
    if not ok:
        return None
    sv = np.linalg.svd(H, compute_uv=False)
    rcond = 1e-14
    out = {"kappa_H": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
           "min_singular_value": float(sv[-1]),
           "nullity": int((sv < sv[0] * rcond).sum())}
    # reduced camera matrix: its rank is chart invariant
    A, B, r, _, _, _, _ = assemble(state, chart, p, layout)
    S = A.T @ (A - B @ np.linalg.pinv(B.T @ B, rcond=rcond) @ (B.T @ A))
    svs = np.linalg.svd(S, compute_uv=False)
    out["rank_S"] = int((svs > svs[0] * 1e-10).sum())
    out["nullity_S"] = int(S.shape[0] - out["rank_S"])
    if out["nullity"] > 0:
        out["sigma_depth_m"] = None
        return out
    Sigma = NOISE ** 2 * np.linalg.inv(H)
    U = position_map(chart, p, state["cams"], state["Rs"], layout)
    SX = U @ Sigma @ U.T
    blk = SX[3 * probe:3 * probe + 3, 3 * probe:3 * probe + 3]
    out["sigma_depth_m"] = float(math.sqrt(max(0.0, direction @ blk @ direction)))
    n_cv = layout[1]
    out["sigma_point_params"] = [float(math.sqrt(max(0.0, Sigma[n_cv + 3 * probe + k,
                                                              n_cv + 3 * probe + k])))
                                 for k in range(3)]
    if chart == "pa":
        _, Jx, _ = chart_jacobian(chart, p[probe], state["cams"], state["Rs"])
        out["dd_domega"] = float(abs(Jx[:, 2] @ direction))
    return out


# --------------------------------------------------------------------------
# S1: undamped GN is chart invariant on a regular problem
# --------------------------------------------------------------------------
def experiment_s1():
    prob = scene_regular()
    init = perturb_shared(prob, np.random.default_rng(101), rel=0.04)
    rows, diffs, base = {}, [], None
    dtrue = np.linalg.norm(np.array(prob["pts"]), axis=1)
    for chart in CHARTS:
        res = solve(init, chart, method="gn", max_iter=25)
        X = res.get("X")
        rows[chart] = {"status": res["status"], "iter": res["iter"],
                       "final_cost": res.get("cost")}
        if X is not None:
            rows[chart]["max_depth_rel_error"] = float(
                np.max(np.abs(np.linalg.norm(X, axis=1) - dtrue) / dtrue))
            if base is None:
                base = X
            else:
                diffs.append(float(np.max(np.abs(X - base))))
    return {"charts": rows,
            "max_abs_position_difference_m": max(diffs) if diffs else None}


# --------------------------------------------------------------------------
# S2: angle sweep -- conditioning, the Cramer-Rao bound, and the solver
# --------------------------------------------------------------------------
def experiment_s2():
    rows = []
    for bdeg in (80.0, 45.0, 20.0, 8.0, 3.0, 1.0, 0.3, 0.1):
        b = math.radians(bdeg)
        direction = np.array([math.sin(b), 0.0, math.cos(b)])
        prob = scene_forward(probe_dir=direction, probe_depth=12.0, noise=NOISE)
        probe = prob["probe"]
        n = unit(prob["pts"][probe])
        dtrue = float(np.linalg.norm(prob["pts"][probe]))
        layout = cam_layout(len(prob["cams"]))
        entry = {"beta_deg": bdeg}
        for chart in CHARTS:
            c = crb(chart, prob, layout, probe, n)
            entry[chart] = c if c is not None else {"status": "chart_undefined"}
        init = perturb_shared(prob, np.random.default_rng(303), rel=0.05)
        for chart in CHARTS:
            res = solve(init, chart, method="lm", max_iter=150)
            d = entry[chart]
            d["lm_status"] = res["status"]
            d["lm_iter"] = res["iter"]
            if res.get("X") is not None:
                dhat = float(np.linalg.norm(res["X"][probe]))
                rel = abs(dhat - dtrue) / dtrue
                d["lm_depth_rel_error"] = rel
                d["lm_depth_bounded"] = bool(rel < 10.0)
                d["lm_rmse_px"] = FOCAL * math.sqrt(2.0 * res["cost"] / (2 * len(prob["obs"])))
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------
# S3: exact alignment -- every chart loses one rank; the parallax chart is
#     not even defined
# --------------------------------------------------------------------------
def experiment_s3():
    b = 0.0
    direction = np.array([0.0, 0.0, 1.0])
    prob = scene_forward(probe_dir=direction, probe_depth=12.0, noise=NOISE)
    probe = prob["probe"]
    n = unit(prob["pts"][probe])
    layout = cam_layout(len(prob["cams"]))
    rows = {}
    for chart in CHARTS:
        c = crb(chart, prob, layout, probe, n)
        rows[chart] = c if c is not None else {"status": "chart_undefined",
                                               "reason": "sin(omega)=0 at exact alignment"}
    # what a direct undamped GN step does from the true point
    for chart in CHARTS:
        p = init_params(prob, chart)
        if p is None:
            continue
        H, g, ok = hessian_at(chart, prob, p, layout)
        dc, dp, cond_s = schur_solve(H, g, layout[1], len(p))
        rows[chart]["gn_direct_solve"] = ("ok" if dc is not None else "singular")
        rows[chart]["cond_S"] = cond_s if np.isfinite(cond_s) else None
    return {"probe_depth_true": float(np.linalg.norm(prob["pts"][probe])),
            "rows": rows}


# --------------------------------------------------------------------------
# S4: one damped step -- isotropic per-chart damping versus the pullback
# --------------------------------------------------------------------------
def experiment_s4():
    """Hold the physical damping ellipsoid fixed and compare the chart step.

    Theorem C1: an isotropic step in chart coordinates equals a physical step
    damped by lambda T^{-T} T^{-1}.  The naive comparison, the same lambda in
    every chart, is reported alongside as the failure mode.
    """
    b = math.radians(1.0)
    direction = np.array([math.sin(b), 0.0, math.cos(b)])
    prob = scene_forward(probe_dir=direction, probe_depth=12.0, noise=NOISE)
    init = perturb_shared(prob, np.random.default_rng(404), rel=0.04)
    layout = cam_layout(len(init["cams"]))
    px = init_params(init, "xyz")
    Hx, gx, _ = hessian_at("xyz", init, px, layout)
    dx_iso = -np.linalg.solve(Hx + 1e-12 * np.eye(Hx.shape[0]), gx)
    rows = []
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        dx_naive = -np.linalg.solve(Hx + lam * np.eye(Hx.shape[0]), gx)
        rec = {"lambda": lam,
               "gn_step_norm_xyz": float(np.linalg.norm(dx_naive))}
        naive, exact, norms = {}, {}, {"xyz": float(np.linalg.norm(dx_naive))}
        for chart in ("id", "ir", "pa"):
            p = init_params(init, chart)
            Hq, gq, _ = hessian_at(chart, init, p, layout)
            T = build_T(chart, p, init["cams"], init["Rs"], layout, len(p))
            # naive: the same lambda in each chart's own coordinates
            dq = -np.linalg.solve(Hq + lam * np.eye(Hq.shape[0]), gq)
            dx_q = T @ dq
            norms[chart] = float(np.linalg.norm(dx_q))
            naive[chart] = float(np.linalg.norm(dx_q - dx_naive)
                                 / max(1e-300, np.linalg.norm(dx_naive)))
            # exact: isotropic in the chart equals the pulled-back metric
            Ti = np.linalg.inv(T)
            metric = Ti.T @ Ti
            dx_ref = -np.linalg.solve(Hx + lam * metric, gx)
            exact[chart] = float(np.linalg.norm(dx_q - dx_ref)
                                 / max(1e-300, np.linalg.norm(dx_ref)))
        rec["same_lambda_everywhere"] = naive
        rec["pullback_metric"] = exact
        rec["step_norm"] = norms
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# S5: the reprojection metric is nearly blind to depth error
# --------------------------------------------------------------------------
def experiment_s5():
    spacing, n_cam = 0.4, 6
    cams = [np.array([0.0, 0.0, k * spacing]) for k in range(n_cam)]
    Rs = [np.eye(3) for _ in range(n_cam)]
    rows = []
    for bdeg in (3.0, 1.0, 0.3, 0.1):
        b = math.radians(bdeg)
        n = np.array([math.sin(b), 0.0, math.cos(b)])
        d = 12.0
        u0 = np.array([_project(Rs[i], cams[i], d * n)[1] for i in range(n_cam)])
        for fac in (1.01, 1.05, 1.2, 1.5, 2.0, 5.0):
            du = np.array([_project(Rs[i], cams[i], d * fac * n)[1] - u0[i]
                           for i in range(n_cam)])
            rows.append({"beta_deg": bdeg, "depth_ratio": fac,
                         "rmse_px": float(FOCAL * np.sqrt(np.mean(np.sum(du ** 2, axis=1)))),
                         "max_px": float(FOCAL * np.max(np.abs(du)))})
    return rows


def main():
    result = {
        "protocol": {
            "noise_per_coordinate": NOISE,
            "gauge": "camera 0 fixed, camera 1 baseline fixed (scale)",
            "charts": list(CHARTS),
            "parallax_anchors": "camera 0 and the last camera",
            "camera_model": "pinhole, focal length 1 in normalized coordinates",
        },
        "S1_gn_invariance": experiment_s1(),
        "S2_angle_sweep": experiment_s2(),
        "S3_exact_alignment": experiment_s3(),
        "S4_damping_metric": experiment_s4(),
        "S5_depth_blindness": experiment_s5(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("S1", json.dumps(result["S1_gn_invariance"], indent=2))
    print("\nS3", json.dumps(result["S3_exact_alignment"], indent=2))
    print("\nS2")
    for r in result["S2_angle_sweep"]:
        print(" beta=%6.2f" % r["beta_deg"])
        for ch in CHARTS:
            d = r[ch]
            if d.get("status") == "chart_undefined":
                print("   %-4s undefined" % ch)
                continue
            sd = d.get("sigma_depth_m")
            print("   %-4s kappa=%9.2e nH=%d nS=%d sigma_d=%s it=%3s relerr=%.3g%s"
                  % (ch, d["kappa_H"], d["nullity"], d["nullity_S"],
                     "None" if sd is None else "%.4e" % sd, d.get("lm_iter"),
                     d.get("lm_depth_rel_error", float("nan")),
                     ("  sigma_omega=%.2e |dd/dw|=%.2e" % (d["sigma_point_params"][2],
                                                           d["dd_domega"])) if ch == "pa" else ""))
    print("\nS4")
    for r in result["S4_damping_metric"]:
        print(" lam=%.0e naive=%s exact=%s" % (r["lambda"], r["same_lambda_everywhere"], r["pullback_metric"]))
    print("\nS5")
    for r in result["S5_depth_blindness"]:
        print("  beta=%5.1f ratio=%4.1f rmse=%.3e px" % (r["beta_deg"], r["depth_ratio"], r["rmse_px"]))


if __name__ == "__main__":
    main()
