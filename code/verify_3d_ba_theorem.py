#!/usr/bin/env python3
"""Numerical verification of the three-dimensional BA-level chart theorem.

A. total-space congruence, reduced-camera-matrix invariance
B. point-block singularity versus BA rank (exact collinear feature)
C. Jacobian density and reduced-system block sparsity
D. damped LM: metric identity, trust-region semi-axes, step comparison
E. near-singular asymptotics (far point, beta = pi/2)

Only numpy is used.  Every reported "theorem" is proved analytically in
ijrr_3d_ba_theorem.md; this script checks the algebra and the finite-
difference Jacobians, and supplies the asymptotic constants.
"""
import json
import math
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
OUT = os.path.join(RESULTS, "ijrr_3d_ba_theorem.json")


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def unit(v):
    return v / np.linalg.norm(v)


def look_at(c, target, up=np.array([0.0, 0.0, 1.0])):
    z = unit(target - c)
    x = unit(np.cross(up, z))
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def dproj(y):
    iz = 1.0 / y[2]
    return np.array([[iz, 0.0, -y[0] * iz * iz],
                     [0.0, iz, -y[1] * iz * iz]])


def transverse_frame(e1):
    a = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e2 = unit(a - np.dot(a, e1) * e1)
    e3 = np.cross(e1, e2)
    return e2, e3


def parallax_chart(q, c0, c1):
    alpha, beta, omega = q
    b = c1 - c0
    B = np.linalg.norm(b)
    e1 = b / B
    e2, e3 = transverse_frame(e1)
    n = np.cos(beta) * e1 + np.sin(beta) * (np.cos(alpha) * e2 + np.sin(alpha) * e3)
    d = B * np.sin(beta + omega) / np.sin(omega)
    return c0 + d * n


def parallax_inverse(X, c0, c1):
    b = c1 - c0
    B = np.linalg.norm(b)
    e1 = b / B
    e2, e3 = transverse_frame(e1)
    r = X - c0
    d = np.linalg.norm(r)
    n = r / d
    cb = np.clip(np.dot(n, e1), -1.0, 1.0)
    beta = np.arccos(cb)
    perp = n - cb * e1
    alpha = np.arctan2(np.dot(perp, e3), np.dot(perp, e2))
    ra = X - c1
    cosw = np.dot(r, ra) / (d * np.linalg.norm(ra))
    omega = np.arccos(np.clip(cosw, -1.0, 1.0))
    return np.array([alpha, beta, omega]), d


def parallax_jacobian(q, c0, c1):
    alpha, beta, omega = q
    b = c1 - c0
    B = np.linalg.norm(b)
    e1 = b / B
    e2, e3 = transverse_frame(e1)
    n = np.cos(beta) * e1 + np.sin(beta) * (np.cos(alpha) * e2 + np.sin(alpha) * e3)
    u = -np.sin(beta) * e1 + np.cos(beta) * (np.cos(alpha) * e2 + np.sin(alpha) * e3)
    w = np.sin(beta) * (-np.sin(alpha) * e2 + np.cos(alpha) * e3)
    d = B * np.sin(beta + omega) / np.sin(omega)
    db = B * np.cos(beta + omega) / np.sin(omega)
    dw = -B * np.sin(beta) / np.sin(omega) ** 2
    return np.column_stack([d * w, db * n + d * u, dw * n])


# --------------------------------------------------------------------------
# linear algebra helpers
# --------------------------------------------------------------------------
def svd_rank(M, rcond=1e-10):
    if M.size == 0:
        return 0, np.array([])
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > s[0] * rcond).sum()), s


def reduced_matrix(A, B, rcond=1e-10):
    """S = A^T (I - P_B) A and rank(B), via SVD of B."""
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    r = int((s > s[0] * rcond).sum())
    Ub = U[:, :r]
    return A.T @ (A - Ub @ (Ub.T @ A)), r


def null_space(M, rcond=1e-10):
    _, s, Vt = np.linalg.svd(M)
    r = int((s > s[0] * rcond).sum())
    return Vt[r:].T


def inertia(M, rcond=1e-9):
    w = np.linalg.eigvalsh(M)
    tol = rcond * max(1.0, float(np.max(np.abs(w))))
    return (int((w > tol).sum()), int((w < -tol).sum()),
            int((np.abs(w) <= tol).sum()), w)


def build_ba(cams, Rs, pts, obs):
    A = np.zeros((2 * len(obs), 6 * len(cams)))
    B = np.zeros((2 * len(obs), 3 * len(pts)))
    for k, (i, j) in enumerate(obs):
        y = Rs[i] @ (pts[j] - cams[i])
        P = dproj(y)
        A[2 * k:2 * k + 2, 6 * i:6 * i + 6] = P @ np.hstack([-skew(y), -Rs[i]])
        B[2 * k:2 * k + 2, 3 * j:3 * j + 3] = P @ Rs[i]
    return A, B


# --------------------------------------------------------------------------
# A. generic BA: congruence and reduced-system invariance
# --------------------------------------------------------------------------
def experiment_a():
    rng = np.random.default_rng(20261104)
    n_cam, n_pt = 5, 15
    angles = np.linspace(0.0, 2.0 * np.pi, n_cam, endpoint=False)
    cams = [np.array([6.0 * np.cos(a), 6.0 * np.sin(a), 1.5]) for a in angles]
    Rs = [look_at(c, np.zeros(3)) for c in cams]
    pts = rng.normal(scale=0.7, size=(n_pt, 3))
    obs = [(i, j) for i in range(n_cam) for j in range(n_pt)]
    A, B = build_ba(cams, Rs, pts, obs)
    Hx = np.block([[A.T @ A, A.T @ B], [B.T @ A, B.T @ B]])
    Sx, rankB = reduced_matrix(A, B)

    m_anchor, a_anchor = 0, 1
    Dblock = np.zeros((3 * n_pt, 3 * n_pt))
    G = np.zeros((3 * n_pt, 6 * n_cam))
    max_recon = 0.0
    max_jac = 0.0
    for j in range(n_pt):
        q, _ = parallax_inverse(pts[j], cams[m_anchor], cams[a_anchor])
        Xr = parallax_chart(q, cams[m_anchor], cams[a_anchor])
        max_recon = max(max_recon, float(np.max(np.abs(Xr - pts[j]))))
        Dj = parallax_jacobian(q, cams[m_anchor], cams[a_anchor])
        Dblock[3 * j:3 * j + 3, 3 * j:3 * j + 3] = Dj
        h = 1e-7
        Dfd = np.zeros((3, 3))
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = h
            Dfd[:, k] = (parallax_chart(q + dq, cams[m_anchor], cams[a_anchor])
                         - parallax_chart(q - dq, cams[m_anchor], cams[a_anchor])) / (2 * h)
        max_jac = max(max_jac, float(np.max(np.abs(Dj - Dfd))))
        for cam_idx in (m_anchor, a_anchor):
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                c0p = cams[m_anchor] + (e if cam_idx == m_anchor else 0.0)
                c1p = cams[a_anchor] + (e if cam_idx == a_anchor else 0.0)
                c0m = cams[m_anchor] - (e if cam_idx == m_anchor else 0.0)
                c1m = cams[a_anchor] - (e if cam_idx == a_anchor else 0.0)
                G[3 * j:3 * j + 3, 6 * cam_idx + 3 + k] = (
                    parallax_chart(q, c0p, c1p) - parallax_chart(q, c0m, c1m)) / (2 * h)

    Aq = A + B @ G
    Bq = B @ Dblock
    Hq = np.block([[Aq.T @ Aq, Aq.T @ Bq], [Bq.T @ Aq, Bq.T @ Bq]])
    T = np.zeros((6 * n_cam + 3 * n_pt, 6 * n_cam + 3 * n_pt))
    T[:6 * n_cam, :6 * n_cam] = np.eye(6 * n_cam)
    T[6 * n_cam:, :6 * n_cam] = G
    T[6 * n_cam:, 6 * n_cam:] = Dblock

    rankHx, _ = svd_rank(Hx)
    rankHq, _ = svd_rank(Hq)
    rankSx, _ = svd_rank(Sx)
    Sq, _ = reduced_matrix(Aq, Bq)
    rankSq, _ = svd_rank(Sq)
    cong = float(np.max(np.abs(Hq - T.T @ Hx @ T)))
    sinv = float(np.max(np.abs(Sq - Sx)))
    sinv_rel = sinv / float(np.max(np.abs(Sx)))

    # gradients: perturb the points so that r != 0
    pts2 = pts + rng.normal(scale=0.01, size=pts.shape)
    r = []
    for k, (i, j) in enumerate(obs):
        y = Rs[i] @ (pts2[j] - cams[i])
        u = np.array([y[0] / y[2], y[1] / y[2]])
        r.append(u)
    r = np.concatenate(r)

    # A and B are evaluated at the same linearization point but the residual
    # is nonzero; recompute Jacobians at pts2 for a consistent gradient test
    A2, B2 = build_ba(cams, Rs, pts2, obs)
    gx = np.concatenate([A2.T @ r, B2.T @ r])
    gq = np.concatenate([(A2 + B2 @ G).T @ r, (B2 @ Dblock).T @ r])
    gq_expected = T.T @ gx
    gerr = float(np.max(np.abs(gq - gq_expected)))
    red_x = A2.T @ r - A2.T @ B2 @ np.linalg.pinv(B2.T @ B2, rcond=1e-10) @ (B2.T @ r)
    red_q = ((A2 + B2 @ G).T @ r
             - (A2 + B2 @ G).T @ (B2 @ Dblock)
             @ np.linalg.pinv((B2 @ Dblock).T @ (B2 @ Dblock), rcond=1e-10)
             @ ((B2 @ Dblock).T @ r))
    red_err = float(np.max(np.abs(red_q - red_x)))

    # damped full-BA identity: q-damping equals the metric T^{-T}T^{-1} in x
    lam = 1e-3
    dq = -np.linalg.solve(Hq + lam * np.eye(Hq.shape[0]), gq)
    dx_from_q = T @ dq
    metric = np.linalg.inv(T).T @ np.linalg.inv(T)
    dx_metric = -np.linalg.solve(Hx + lam * metric, gx)
    dx_isotropic = -np.linalg.solve(Hx + lam * np.eye(Hx.shape[0]), gx)
    damped_identity = float(np.max(np.abs(dx_from_q - dx_metric)))
    damped_difference = float(np.linalg.norm(dx_from_q - dx_isotropic)
                              / np.linalg.norm(dx_isotropic))

    return {
        "cameras": n_cam,
        "points": n_pt,
        "observations": len(obs),
        "variables": int(Hx.shape[0]),
        "rank_Hx": rankHx,
        "nullity_Hx": int(Hx.shape[0] - rankHx),
        "rank_Hq": rankHq,
        "nullity_Hq": int(Hq.shape[0] - rankHq),
        "rank_JX": rankB,
        "rank_Sx": rankSx,
        "nullity_Sx": int(Sx.shape[0] - rankSx),
        "rank_Sq": rankSq,
        "nullity_Sq": int(Sq.shape[0] - rankSq),
        "rank_identity_Hx_minus_rank_JX": int(rankHx - rankB),
        "max_reconstruction_error": max_recon,
        "max_analytic_vs_fd_chart_jacobian": max_jac,
        "max_Hq_minus_TtHxT": cong,
        "max_Sq_minus_Sx": sinv,
        "relative_Sq_minus_Sx": sinv_rel,
        "max_gradient_congruence_error": gerr,
        "max_reduced_gradient_difference": red_err,
        "damped_full_ba_metric_identity_error": damped_identity,
        "damped_full_ba_relative_step_difference": damped_difference,
        "condition_D": float(np.linalg.cond(Dblock)),
        "inertia_Hx": inertia(Hx)[:3],
        "inertia_Hq": inertia(Hq)[:3],
    }


# --------------------------------------------------------------------------
# B. exact collinear feature: point-block rank versus BA rank
# --------------------------------------------------------------------------
def experiment_b():
    cams = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    target = np.array([2.0, 0.15, 0.1])
    Rs = [look_at(c, target) for c in cams]
    generic = [np.array([2.0, 0.3, 0.15]), np.array([2.3, -0.15, 0.1]),
               np.array([1.8, 0.1, -0.2]), np.array([2.15, -0.25, -0.1]),
               np.array([2.4, 0.2, 0.25])]
    base_obs = [(i, j) for i in range(4) for j in range(5)]
    A0, B0 = build_ba(cams, Rs, generic, base_obs)
    H0 = np.block([[A0.T @ A0, A0.T @ B0], [B0.T @ A0, B0.T @ B0]])
    S0, rankB0 = reduced_matrix(A0, B0)
    rankH0, _ = svd_rank(H0)
    rankS0, _ = svd_rank(S0)

    pts = generic + [np.array([2.0, 0.0, 0.0])]
    aug_obs = base_obs + [(0, 5), (1, 5)]
    A1, B1 = build_ba(cams, Rs, pts, aug_obs)
    H1 = np.block([[A1.T @ A1, A1.T @ B1], [B1.T @ A1, B1.T @ B1]])
    S1, rankB1 = reduced_matrix(A1, B1)
    rankH1, _ = svd_rank(H1)
    rankS1, _ = svd_rank(S1)

    # point block of the collinear feature in XYZ
    B5 = B1[:, 15:18]
    pb_rank, pb_s = svd_rank(B5)

    # extra null vector of H1 must have zero pose components
    nsH1 = null_space(H1)
    pose_part_rank, _ = svd_rank(nsH1[:24, :])

    # null spaces of S0 and S1 must coincide
    ns0 = null_space(S0)
    ns1 = null_space(S1)
    Q0, _ = np.linalg.qr(ns0)
    Q1, _ = np.linalg.qr(ns1)
    principal_cos = np.linalg.svd(Q0.T @ Q1, compute_uv=False)

    diff = S1 - S0
    eig_diff = np.linalg.eigvalsh(diff)

    # the collinear ray: every point X=(t,0,0), t>1, has (beta,omega)=(0,0)
    ray = []
    for t in (1.5, 2.0, 3.0, 5.0, 8.0, 20.0):
        q, d = parallax_inverse(np.array([t, 0.0, 0.0]), cams[0], cams[1])
        ray.append({"t": t, "beta": float(q[1]), "omega": float(q[2]),
                    "reconstruction_numerator": float(1.0 * math.sin(q[1] + q[2])),
                    "reconstruction_denominator": float(math.sin(q[2])),
                    "distance": float(d)})

    return {
        "base": {
            "rank_H": rankH0, "nullity_H": int(H0.shape[0] - rankH0),
            "rank_JX": rankB0, "rank_S": rankS0,
            "nullity_S": int(S0.shape[0] - rankS0),
            "rank_identity": int(rankH0 - rankB0),
        },
        "augmented_with_collinear_point": {
            "rank_H": rankH1, "nullity_H": int(H1.shape[0] - rankH1),
            "rank_JX": rankB1, "rank_S": rankS1,
            "nullity_S": int(S1.shape[0] - rankS1),
            "rank_identity": int(rankH1 - rankB1),
            "point_block_rank": pb_rank,
            "point_block_singular_values": [float(x) for x in pb_s],
            "pose_rank_of_full_null_space": pose_part_rank,
        },
        "S_null_space_principal_cosines_min": float(np.min(principal_cos)),
        "S_augmented_minus_base_min_eigenvalue": float(np.min(eig_diff)),
        "S_augmented_minus_base_max_abs": float(np.max(np.abs(diff))),
        "collinear_ray": ray,
    }


# --------------------------------------------------------------------------
# C. Jacobian density and block sparsity
# --------------------------------------------------------------------------
def _density_case(n_cam, n_pt, obs, anchors_fn, rng):
    angles = np.linspace(0.0, 2.0 * np.pi, n_cam, endpoint=False)
    cams = [np.array([6.0 * np.cos(a), 6.0 * np.sin(a), 1.5]) for a in angles]
    Rs = [look_at(c, np.zeros(3)) for c in cams]
    pts = rng.normal(scale=0.7, size=(n_pt, 3))
    A, B = build_ba(cams, Rs, pts, obs)

    Dblock = np.zeros((3 * n_pt, 3 * n_pt))
    G = np.zeros((3 * n_pt, 6 * n_cam))
    h = 1e-7
    for j in range(n_pt):
        m_anchor, a_anchor = anchors_fn(j, n_cam)
        q, _ = parallax_inverse(pts[j], cams[m_anchor], cams[a_anchor])
        Dblock[3 * j:3 * j + 3, 3 * j:3 * j + 3] = parallax_jacobian(
            q, cams[m_anchor], cams[a_anchor])
        for cam_idx in (m_anchor, a_anchor):
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                c0p = cams[m_anchor] + (e if cam_idx == m_anchor else 0.0)
                c1p = cams[a_anchor] + (e if cam_idx == a_anchor else 0.0)
                c0m = cams[m_anchor] - (e if cam_idx == m_anchor else 0.0)
                c1m = cams[a_anchor] - (e if cam_idx == a_anchor else 0.0)
                G[3 * j:3 * j + 3, 6 * cam_idx + 3 + k] = (
                    parallax_chart(q, c0p, c1p) - parallax_chart(q, c0m, c1m)) / (2 * h)
    Aq = A + B @ G

    tol = 1e-9 * max(1.0, float(np.max(np.abs(Aq))))
    nnz_A = int((np.abs(A) > tol).sum())
    nnz_Aq = int((np.abs(Aq) > tol).sum())

    Sx, _ = reduced_matrix(A, B)
    Sq, _ = reduced_matrix(Aq, B @ Dblock)

    def block_pattern(S):
        pattern = set()
        scale = float(np.max(np.abs(S))) if S.size else 1.0
        for i in range(n_cam):
            for k in range(i, n_cam):
                if np.linalg.norm(S[6 * i:6 * i + 6, 6 * k:6 * k + 6]) > 1e-9 * scale:
                    pattern.add((i, k))
        return pattern

    px = block_pattern(Sx)
    pq = block_pattern(Sq)
    return {
        "nonzeros_A_xyz": nnz_A,
        "nonzeros_A_parallax": nnz_Aq,
        "density_ratio": nnz_Aq / nnz_A,
        "nonzero_pose_blocks_S_xyz": len(px),
        "nonzero_pose_blocks_S_parallax": len(pq),
        "block_patterns_identical": px == pq,
        "possible_upper_blocks": n_cam * (n_cam + 1) // 2,
    }


def experiment_c():
    rng = np.random.default_rng(7)
    full = _density_case(5, 15,
                         [(i, j) for i in range(5) for j in range(15)],
                         lambda j, n: (0, 1), rng)
    partial = _density_case(6, 14,
                            [(cam, j) for j in range(14)
                             for cam in ((j % 6), ((j + 2) % 6), ((j + 4) % 6))],
                            lambda j, n: (j % n, (j + 2) % n), rng)
    return {"full_visibility_5cam": full, "partial_visibility_6cam": partial}


# --------------------------------------------------------------------------
# D. damped LM: metric identity, trust-region semi-axes, step comparison
# --------------------------------------------------------------------------
def experiment_d():
    rng = np.random.default_rng(11)
    errs = []
    for _ in range(30):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        M = Q @ np.diag(rng.uniform(0.1, 10.0, size=3)) @ Q.T
        while True:
            D = rng.normal(size=(3, 3))
            if abs(np.linalg.det(D)) > 1e-3:
                break
        for lam in (1e-6, 1e-3, 1e-1, 1.0, 10.0):
            Dinv = np.linalg.inv(D)
            Mx = Dinv.T @ M @ Dinv
            lhs = D @ np.linalg.solve(M + lam * np.eye(3), D.T)
            rhs = np.linalg.inv(Mx + lam * np.linalg.inv(D @ D.T))
            errs.append(float(np.max(np.abs(lhs - rhs))))
    metric_identity_error = max(errs)

    # trust-region semi-axes of the parallax chart for a far point
    c0 = np.array([0.0, 0.0, 0.0])
    c1 = np.array([1.0, 0.0, 0.0])
    axes = []
    for om in (0.1, 0.01, 0.001):
        beta = math.pi / 2.0
        d = 1.0 / math.tan(om)
        D = parallax_jacobian(np.array([0.0, beta, om]), c0, c1)
        s = np.linalg.svd(D, compute_uv=False)
        axes.append({
            "omega": om,
            "depth_over_baseline": d,
            "singular_values": [float(x) for x in s],
            "aspect_ratio_max_min": float(s[0] / s[-1]),
            "predicted_depth_over_baseline": float(d),
        })

    # single-point LM step: XYZ versus parallax, same lambda
    om = 0.02
    d = 1.0 / math.tan(om)
    Xtrue = np.array([0.0, d, 0.0])
    X0 = np.array([0.0, d * 1.05, 0.0])
    Rs = [look_at(c0, Xtrue), look_at(c1, Xtrue)]
    cams = [c0, c1]
    obs = [(0, 0), (1, 0)]
    A, B = build_ba(cams, Rs, [Xtrue], obs)
    # residuals at X0
    r = []
    for i in range(2):
        y = Rs[i] @ (X0 - cams[i])
        r.append(np.array([y[0] / y[2], y[1] / y[2]]))
    r = np.concatenate(r)
    A0, B0 = build_ba(cams, Rs, [X0], obs)
    q0, _ = parallax_inverse(X0, c0, c1)
    D0 = parallax_jacobian(q0, c0, c1)
    Jq = B0 @ D0
    step_rows = []
    for lam in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        dx = -np.linalg.solve(B0.T @ B0 + lam * np.eye(3), B0.T @ r)
        dq = -np.linalg.solve(Jq.T @ Jq + lam * np.eye(3), Jq.T @ r)
        dxq = D0 @ dq
        step_rows.append({
            "lambda": lam,
            "step_xyz": [float(x) for x in dx],
            "step_parallax_mapped": [float(x) for x in dxq],
            "relative_difference": float(np.linalg.norm(dx - dxq) / np.linalg.norm(dx)),
            "depth_ratio": float(abs(dxq[1]) / abs(dx[1])),
        })

    return {
        "metric_identity_max_error": metric_identity_error,
        "trust_region_axes": axes,
        "single_point_lm_steps": step_rows,
    }


# --------------------------------------------------------------------------
# E. near-singular asymptotics at beta = pi/2
# --------------------------------------------------------------------------
def experiment_e():
    rows = []
    for k in range(1, 8):
        om = 10.0 ** (-k)
        d = 1.0 / math.tan(om)
        c0 = np.array([0.0, 0.0, 0.0])
        c1 = np.array([1.0, 0.0, 0.0])
        X = np.array([0.0, d, 0.0])
        Rs = [look_at(c0, X), look_at(c1, X)]
        cams = [c0, c1]
        A, B = build_ba(cams, Rs, [X], [(0, 0), (1, 0)])
        IX = B.T @ B
        D = parallax_jacobian(np.array([0.0, math.pi / 2.0, om]), c0, c1)
        Iq = D.T @ IX @ D
        n = np.array([0.0, 1.0, 0.0])
        rows.append({
            "omega": om,
            "depth": d,
            "kappa_IX": float(np.linalg.cond(IX)),
            "kappa_Iq": float(np.linalg.cond(Iq)),
            "kappa_D": float(np.linalg.cond(D)),
            "depth_information": float(n @ IX @ n),
            "det_IX": float(np.linalg.det(IX)),
            "det_Iq": float(np.linalg.det(Iq)),
            "Iq": [[float(x) for x in row] for row in Iq],
        })
    arr = np.array([[r["omega"], r["kappa_IX"], r["kappa_Iq"], r["kappa_D"],
                     r["depth_information"], r["det_IX"], r["det_Iq"]]
                    for r in rows])
    slopes = {}
    names = ["kappa_IX", "kappa_Iq", "kappa_D", "depth_information",
             "det_IX", "det_Iq"]
    for idx, name in enumerate(names, start=1):
        slopes[name] = float(np.polyfit(np.log(arr[:4, 0]), np.log(arr[:4, idx]), 1)[0])
    return {"rows": rows, "loglog_slopes": slopes}


def main():
    result = {
        "experiment_A_generic_ba": experiment_a(),
        "experiment_B_collinear_point": experiment_b(),
        "experiment_C_sparsity": experiment_c(),
        "experiment_D_damped_lm": experiment_d(),
        "experiment_E_near_singular": experiment_e(),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
