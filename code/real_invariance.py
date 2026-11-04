#!/usr/bin/env python3
"""Experiment A: chart invariance on a real monocular sequence.

The paper's claim is an equality between charts, not an accuracy claim, so this
test needs no ground-truth depth: at one physical configuration every chart must
produce the same physical Gauss-Newton step and the same reduced camera system.
A relative disagreement above 1e-8 falsifies the theorem.

Sequence: TUM RGB-D (default fr1/xyz).  Poses from the motion-capture track,
points from two-view triangulation of KLT tracks.  No depth image is read.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_chart_experiment import (  # noqa: E402
    CHARTS, assemble, cam_layout, init_params, physical_step,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = os.environ.get("TUM_SEQ", os.path.join("data", "rgbd_dataset_freiburg1_xyz"))

# TUM fr1 calibration, as published with the benchmark
FX, FY, CX, CY = 517.306408, 516.469215, 318.643040, 255.313989
DIST = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])
K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)

N_CAM = 8          # frames per window
STRIDE = 6         # frames between window members
N_WIN = 10         # windows to analyse
MAX_REPROJ_PX = 3.0


# --------------------------------------------------------------------------
# TUM file formats
# --------------------------------------------------------------------------
def read_index(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            out.append((float(parts[0]), parts[1]))
    return out


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def read_gt(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            out.append((v[0], quat_to_R(*v[4:8]), np.array(v[1:4])))
    return out


def pose_at(gt, t):
    """Nearest motion-capture pose, returned as world-to-camera (R, c)."""
    best = min(gt, key=lambda g: abs(g[0] - t))
    return best[1].T.copy(), best[2].copy()


def to_normalized(uv):
    p = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K_MAT, DIST).reshape(-1, 2)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def track_window(gray):
    p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=900, qualityLevel=0.01, minDistance=7)
    if p0 is None:
        return None
    pts = p0.reshape(-1, 1, 2)
    cols, valid = [pts.copy()], np.ones(len(pts), bool)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    for k in range(1, len(gray)):
        fwd, s1, _ = cv2.calcOpticalFlowPyrLK(gray[k - 1], gray[k], pts, None,
                                             winSize=(21, 21), maxLevel=3, criteria=crit)
        back, s2, _ = cv2.calcOpticalFlowPyrLK(gray[k], gray[k - 1], fwd, None,
                                              winSize=(21, 21), maxLevel=3, criteria=crit)
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        valid &= (s1.ravel() == 1) & (s2.ravel() == 1) & (fb < 0.7)
        pts = fwd.copy()
        cols.append(pts.copy())
    ok = np.where(valid)[0]
    if len(ok) < 8:
        return None
    return np.stack([c[ok, 0, :] for c in cols], axis=1)   # (m, K, 2) pixels


def triangulate(xn, cams, Rs):
    """Linear triangulation from normalised observations xn[i] = (x, y)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(len(cams)):
        d = Rs[i].T @ np.array([xn[i][0], xn[i][1], 1.0])
        d /= np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ cams[i]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def reproj_px(xn, X, cams, Rs):
    """RMS reprojection error in pixels over the window."""
    err = []
    for i in range(len(cams)):
        y = Rs[i] @ (X - cams[i])
        if y[2] <= 1e-9:
            return float("inf")
        err.append(np.hypot((y[0] / y[2] - xn[i][0]) * FX, (y[1] / y[2] - xn[i][1]) * FY))
    return float(np.sqrt(np.mean(np.square(err))))


# --------------------------------------------------------------------------
# the invariance measurement
# --------------------------------------------------------------------------
def analyze(state):
    layout = cam_layout(len(state["cams"]))
    n_cv = layout[1]
    out = {"n_cam": len(state["cams"]), "n_pt": len(state["pts"]), "charts": {}}
    ref = None
    for chart in CHARTS:
        params = init_params(state, chart)
        if params is None:
            out["charts"][chart] = {"status": "chart_undefined"}
            continue
        A, B, r, _, _, _, ok = assemble(state, chart, params, layout)
        if not ok:
            out["charts"][chart] = {"status": "state_behind_camera"}
            continue
        H = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
        g = np.concatenate([A.T @ r, B.T @ r])
        ev = np.linalg.eigvalsh(H)
        top = max(abs(ev[-1]), 1e-300)
        nullity = int(np.sum(ev < 1e-12 * top))
        Hcc, Hcp, Hpp = H[:n_cv, :n_cv], H[:n_cv, n_cv:], H[n_cv:, n_cv:]
        S = Hcc - Hcp @ np.linalg.pinv(Hpp, rcond=1e-14) @ Hcp.T
        d = -np.linalg.lstsq(H, g, rcond=None)[0]
        dx = physical_step(chart, params, d[n_cv:], state["cams"], state["Rs"],
                           layout, dc=d[:n_cv])
        rec = {"status": "ok", "kappa_H": float(top / max(ev[0], 1e-300)),
               "nullity_H": nullity, "kappa_S": float(np.linalg.cond(S)),
               "S": S, "dx": dx, "dc": d[:n_cv]}
        out["charts"][chart] = rec
        if chart == "xyz":
            ref = rec
    if ref is None:
        out["status"] = "no_reference_chart"
        return out
    for chart, rec in out["charts"].items():
        if rec.get("status") != "ok":
            continue
        rec["S_rel"] = float(np.linalg.norm(rec["S"] - ref["S"])
                             / max(np.linalg.norm(ref["S"]), 1e-300))
        rec["step_rel"] = float(np.linalg.norm(rec["dx"] - ref["dx"])
                                / max(np.linalg.norm(ref["dx"]), 1e-300))
        rec["camstep_rel"] = float(np.linalg.norm(rec["dc"] - ref["dc"])
                                   / max(np.linalg.norm(ref["dc"]), 1e-300))
    for chart, rec in out["charts"].items():
        if rec.get("status") != "ok":
            continue
        for k in ("S", "dx", "dc"):
            rec.pop(k)
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
def main():
    if cv2 is None:
        raise SystemExit("OpenCV is required")
    rgb = read_index(os.path.join(SEQ, "rgb.txt"))
    gt = read_gt(os.path.join(SEQ, "groundtruth.txt"))
    print("sequence :", SEQ)
    print("frames   :", len(rgb), " gt poses:", len(gt))

    span = STRIDE * (N_CAM - 1)
    starts = [int(x) for x in np.linspace(0, max(0, len(rgb) - span - 1), N_WIN)]
    result = {"sequence": os.path.basename(SEQ), "n_cam": N_CAM, "stride": STRIDE,
              "windows": []}

    for w, s0 in enumerate(starts):
        idx = [s0 + k * STRIDE for k in range(N_CAM)]
        gray = []
        for i in idx:
            im = cv2.imread(os.path.join(SEQ, rgb[i][1]), cv2.IMREAD_GRAYSCALE)
            if im is None:
                break
            gray.append(im)
        if len(gray) != N_CAM:
            continue
        tracks = track_window(gray)
        if tracks is None:
            result["windows"].append({"start": s0, "status": "tracking_failed"})
            print("window %2d  start=%4d  tracking failed" % (w, s0), flush=True)
            continue
        xn = to_normalized(tracks.reshape(-1, 2)).reshape(tracks.shape)
        cams, Rs = [], []
        for i in idx:
            R, c = pose_at(gt, rgb[i][0])
            Rs.append(R)
            cams.append(c)
        pts, keep, reps = [], [], []
        for j in range(tracks.shape[0]):
            X = triangulate(xn[j], cams, Rs)
            if not np.all(np.isfinite(X)):
                continue
            if not all((Rs[i] @ (X - cams[i]))[2] > 0 for i in range(N_CAM)):
                continue
            e = reproj_px(xn[j], X, cams, Rs)
            if e > MAX_REPROJ_PX:
                continue
            pts.append(X)
            keep.append(j)
            reps.append(e)
        if len(pts) < 8:
            result["windows"].append({"start": s0, "status": "too_few_points",
                                      "n_track": int(tracks.shape[0])})
            print("window %2d  start=%4d  only %d points" % (w, s0, len(pts)), flush=True)
            continue
        u = np.array([xn[j, i] for i in range(N_CAM) for j in keep]).ravel()
        state = {"cams": cams, "Rs": Rs, "pts": np.array(pts),
                 "obs": [(i, j) for i in range(N_CAM) for j in range(len(pts))],
                 "u": u}
        rec = analyze(state)
        rec.update({"start": s0, "frames": idx,
                    "baseline_m": float(np.linalg.norm(cams[-1] - cams[0])),
                    "median_depth_m": float(np.median(
                        np.linalg.norm(np.array(pts) - cams[0], axis=1))),
                    "median_reproj_px": float(np.median(reps)),
                    "n_track": int(tracks.shape[0]), "n_pt_used": len(pts)})
        result["windows"].append(rec)
        print("window %2d  start=%4d  pts=%3d  base=%.3f m  depth=%.2f m  %s"
              % (w, s0, len(pts), rec["baseline_m"], rec["median_depth_m"],
                 rec["status"]), flush=True)

    good = [r for r in result["windows"] if r.get("status") == "ok"]
    summary = {"n_windows": len(result["windows"]), "n_ok": len(good)}
    for chart in CHARTS:
        okc = [r["charts"][chart] for r in good
               if r["charts"].get(chart, {}).get("status") == "ok"]
        summary[chart] = {
            "n": len(okc),
            "step_rel_max": float(max(c["step_rel"] for c in okc)) if okc else None,
            "S_rel_max": float(max(c["S_rel"] for c in okc)) if okc else None,
            "camstep_rel_max": float(max(c["camstep_rel"] for c in okc)) if okc else None,
            "kappa_H_min": float(min(c["kappa_H"] for c in okc)) if okc else None,
            "kappa_H_max": float(max(c["kappa_H"] for c in okc)) if okc else None,
        }
    kap = [summary[c]["kappa_H_max"] / summary[c]["kappa_H_min"]
           for c in CHARTS if summary[c]["kappa_H_max"]]
    summary["kappa_spread_across_charts"] = float(max(kap)) if kap else None
    result["summary"] = summary

    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    out = os.path.join(HERE, "..", "results", "real_invariance.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    print("\nsummary")
    print("  windows ok : %d / %d" % (summary["n_ok"], summary["n_windows"]))
    for chart in CHARTS:
        s = summary[chart]
        if not s["n"]:
            print("  %-4s : undefined" % chart)
            continue
        print("  %-4s : step_rel=%.2e  camstep_rel=%.2e  S_rel=%.2e  kappa %.2e..%.2e"
              % (chart, s["step_rel_max"], s["camstep_rel_max"], s["S_rel_max"],
                 s["kappa_H_min"], s["kappa_H_max"]))
    print("  wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
