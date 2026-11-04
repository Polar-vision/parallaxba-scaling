"""Deterministic checks for the parametrization audit.

Only Python's standard library is used. The script verifies:
1. The exact rank change in Zhao et al.'s aligned 2-D example is accompanied by
   a non-injective chart, not by newly identifiable Euclidean depth.
2. Zhao et al.'s original 3-D scaled-ray equation degenerates to 0/0 at the
   same collinear, zero-parallax configuration.
3. On a common open chart, inverse-depth and depth GN steps are identical in
   physical coordinates, and LM steps are identical under the pullback metric.
4. The sine floor that a Ceres-based implementation adds defines a non-unique
   numerical continuation near the degenerate associate-view ray.
5. The normalized-parallax chart nu=tan(omega)/sin(beta) agrees with the
   original chart on their common regular domain and covers aligned finite
   points where the original chart is singular.
"""

import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "controlled_results.json")


def det2(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def inv2(a):
    d = det2(a)
    if abs(d) < 1e-30:
        raise ValueError("singular 2x2 matrix")
    return [[a[1][1] / d, -a[0][1] / d],
            [-a[1][0] / d, a[0][0] / d]]


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def mv(a, x):
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def tr(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def jt_j(j):
    return mm(tr(j), j)


def jt_r(j, r):
    return mv(tr(j), r)


def eig2_sym(a):
    t = a[0][0] + a[1][1]
    d = math.sqrt(max(0.0, (a[0][0] - a[1][1]) ** 2 + 4 * a[0][1] ** 2))
    return (t - d) / 2, (t + d) / 2


def cond2(a):
    lo, hi = eig2_sym(a)
    if lo <= 0:
        return float("inf")
    return hi / lo


def corr2(a):
    s0 = math.sqrt(a[0][0])
    s1 = math.sqrt(a[1][1])
    return [[1.0, a[0][1] / (s0 * s1)], [a[1][0] / (s0 * s1), 1.0]]


def model(theta, scalar, phi_t, phi_2, inverse=False, baseline=1.0):
    if inverse:
        rho = scalar
        y = math.sin(theta) - rho * baseline * math.sin(phi_t)
        x = math.cos(theta) - rho * baseline * math.cos(phi_t)
    else:
        d = scalar
        y = d * math.sin(theta) - baseline * math.sin(phi_t)
        x = d * math.cos(theta) - baseline * math.cos(phi_t)
    return math.atan2(y, x) - phi_2


def jacobian(theta, scalar, phi_t, phi_2, inverse=False, baseline=1.0):
    if inverse:
        rho = scalar
        y = math.sin(theta) - rho * baseline * math.sin(phi_t)
        x = math.cos(theta) - rho * baseline * math.cos(phi_t)
        den = x * x + y * y
        y_t, x_t = math.cos(theta), -math.sin(theta)
        y_s, x_s = -baseline * math.sin(phi_t), -baseline * math.cos(phi_t)
    else:
        d = scalar
        y = d * math.sin(theta) - baseline * math.sin(phi_t)
        x = d * math.cos(theta) - baseline * math.cos(phi_t)
        den = x * x + y * y
        y_t, x_t = d * math.cos(theta), -d * math.sin(theta)
        y_s, x_s = math.sin(theta), math.cos(theta)
    return [[1.0, 0.0],
            [(x * y_t - y * x_t) / den, (x * y_s - y * x_s) / den]]


def aligned_2d_example():
    theta = 0.3
    phi_t = theta  # phi = theta - phi_t = 0
    phi_2 = 0.0
    d = 4.0
    j_xyz = jacobian(theta, d, phi_t, phi_2, inverse=False)
    j_idp = jacobian(theta, 1.0 / d, phi_t, phi_2, inverse=True)
    omega_values = []
    for depth in (1.0, 4.0, 100.0):
        # Exact collinearity: both bearing rays are identical for every depth.
        omega_values.append(0.0 * depth)
    return {
        "configuration": {"theta": theta, "phi_t": phi_t, "phi": 0.0},
        "xyz_jacobian": j_xyz,
        "xyz_normal_matrix": jt_j(j_xyz),
        "idp_jacobian": j_idp,
        "idp_normal_matrix": jt_j(j_idp),
        "parallax_jacobian": [[1.0, 0.0], [1.0, 1.0]],
        "parallax_normal_matrix": [[2.0, 1.0], [1.0, 1.0]],
        "omega_at_depths_1_4_100": omega_values,
        "chart_derivative_theta_d_to_theta_omega": [[1.0, 0.0], [0.0, 0.0]],
        "interpretation": "Parallax coordinates remain informative while depth is not identifiable; the chart is non-injective at phi=0."
    }


def scaled_ray_original(beta, omega, baseline=1.0):
    # Main bearing and baseline are both +Z. Current camera is the associate.
    # Eq. (8) becomes B*sin(beta+omega)*u - sin(omega)*B*u.
    scale = baseline * (math.sin(beta + omega) - math.sin(omega))
    return [0.0, 0.0, scale]


def projection_2d_over_z(v):
    if v[2] == 0.0:
        return [float("nan"), float("nan")]
    return [v[0] / v[2], v[1] / v[2]]


def degenerate_3d_example():
    exact = scaled_ray_original(0.0, 0.0)
    backward = scaled_ray_original(math.pi, 0.0)
    # The additional floor sin(beta) >= 1e-12 changes the exact beta=0 ray.
    cep_scale = 1e-12
    cep_ray = [0.0, 0.0, cep_scale]
    near = []
    for beta in (0.0, 1e-14, 1e-12, 1e-10, 1e-8):
        original = scaled_ray_original(beta, 0.0)
        regularized = [0.0, 0.0, math.sqrt(math.sin(beta) ** 2 + 1e-24)]
        near.append({
            "beta_rad": beta,
            "original_scaled_ray": original,
            "original_projection": projection_2d_over_z(original),
            "cep_regularized_projection": projection_2d_over_z(regularized)
        })
    return {
        "original_forward_scaled_ray": exact,
        "original_forward_projection": projection_2d_over_z(exact),
        "original_backward_scaled_ray": backward,
        "cep_regularized_ray_at_exact_zero": cep_ray,
        "cep_regularized_projection_at_exact_zero": projection_2d_over_z(cep_ray),
        "near_zero_sweep": near,
        "interpretation": "The original 3-D residual is undefined at beta=omega=0; the floor adds a regularization that the original equation does not contain."
    }


def associate_projection_reduced(beta, omega, regularization=0.0, stable_original=False):
    """Reduced associate-view Eq. (8) in the baseline/main-ray plane.

    beta >= 0 is the unsigned angle phi between baseline and main bearing.
    The returned scalar is atan2 of the two scaled-ray components.  With
    stable_original=True, sin(beta) is evaluated directly to separate the
    mathematical formula from floating-point cancellation in 1-cos(beta)^2.
    """
    cos_beta = math.cos(beta)
    if stable_original:
        sin_beta = math.sin(beta)
    else:
        sin_beta = math.sqrt(max(0.0, 1.0 - cos_beta * cos_beta) + regularization)
    sin_beta_plus_omega = sin_beta * math.cos(omega) + cos_beta * math.sin(omega)
    along = sin_beta_plus_omega * cos_beta - math.sin(omega)
    # The sine floor regularizes the triangulation factor only; it does not
    # alter the main-bearing unit vector used for the transverse component.
    perpendicular = sin_beta_plus_omega * math.sin(beta)
    if along == 0.0 and perpendicular == 0.0:
        return float("nan")
    return math.atan2(perpendicular, along)


def regularization_continuation():
    omega = 0.1
    rows = []
    for beta in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6):
        original = associate_projection_reduced(beta, omega, stable_original=True)
        cep = associate_projection_reduced(beta, omega, regularization=1e-24)
        rows.append({
            "beta_rad": beta,
            "omega_rad": omega,
            "original_projection_angle_rad": original,
            "cep_regularized_projection_angle_rad": cep,
            "cep_minus_original_rad": cep - original if math.isfinite(original) else None,
        })
    physical_paths = []
    for beta in (1e-14, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6):
        for c in (0.01, 0.5, 0.9):
            # To first order for a distant point beyond the associate camera
            # on a nearly forward baseline, omega/beta = B/(Z-B), hence 0<c<1.
            omega_physical = c * beta
            original = associate_projection_reduced(
                beta, omega_physical, stable_original=True)
            cep = associate_projection_reduced(
                beta, omega_physical, regularization=1e-24)
            physical_paths.append({
                "beta_rad": beta, "omega_over_beta": c,
                "omega_rad": omega_physical,
                "original_projection_angle_rad": original,
                "cep_regularized_projection_angle_rad": cep,
                "absolute_cep_error_rad": abs(cep - original),
            })
    return {
        "configuration": "associate view; main ray nearly parallel to the anchor baseline",
        "analytic_transverse_over_longitudinal_limit_as_beta_to_0": math.tan(omega),
        "analytic_projection_bearing_limit_as_beta_to_0": omega,
        "note": "The limiting transverse/longitudinal ratio is tan(omega), so the projected bearing angle tends to omega.",
        "fixed_omega_auxiliary_coordinate_rows": rows,
        "fixed_omega_caveat": "beta near zero with fixed omega=0.1 is generally not a regular finite-point triangulation; it tests only the optimizer's auxiliary-coordinate domain.",
        "physically_consistent_forward_paths": physical_paths,
        "numerical_caveat": "The regularized model evaluates 1-cos(beta)^2 in IEEE double precision, matching the cancellation-prone form of the implementation's formula; it is a reduced numerical experiment, not a measured error on real data.",
        "interpretation": "The original scaled ray is zero on the entire beta=0 line in this reduced associate-view model. The floor defines a finite continuation, but double-precision cancellation can change the residual near collinearity. Absolute errors must be reported alongside these relative effects.",
    }


def gn_lm_invariance():
    random.seed(20261103)
    rows = []
    max_gn = 0.0
    max_lm = 0.0
    for phi in (0.01, 0.05, 0.2, 0.5):
        for d in (2.0, 10.0, 100.0):
            theta = 0.35
            phi_t = theta - phi
            phi_2 = -0.1
            # Noisy measurements define the same residual in both charts.
            z1 = theta - 0.0 + 0.002
            truth_theta = theta + 0.03
            truth_d = d * 1.1
            z2 = model(truth_theta, truth_d, phi_t, phi_2) - 0.001
            r = [theta - z1, model(theta, d, phi_t, phi_2) - z2]
            jx = jacobian(theta, d, phi_t, phi_2, inverse=False)
            rho = 1.0 / d
            # D = d(theta,d)/d(theta,rho).
            D = [[1.0, 0.0], [0.0, -1.0 / (rho * rho)]]
            jq = mm(jx, D)
            dx = mv(inv2(jt_j(jx)), [-v for v in jt_r(jx, r)])
            dq = mv(inv2(jt_j(jq)), [-v for v in jt_r(jq, r)])
            mapped = mv(D, dq)
            gn_err = max(abs(dx[i] - mapped[i]) for i in range(2))
            max_gn = max(max_gn, gn_err)
            lam = 1e-3
            dd_t = mm(D, tr(D))
            metric_x = inv2(dd_t)
            hx = jt_j(jx)
            hq = jt_j(jq)
            hx_lm = [[hx[i][j] + lam * metric_x[i][j] for j in range(2)] for i in range(2)]
            hq_lm = [[hq[i][j] + lam * (1.0 if i == j else 0.0) for j in range(2)] for i in range(2)]
            dx_lm = mv(inv2(hx_lm), [-v for v in jt_r(jx, r)])
            dq_lm = mv(inv2(hq_lm), [-v for v in jt_r(jq, r)])
            mapped_lm = mv(D, dq_lm)
            lm_err = max(abs(dx_lm[i] - mapped_lm[i]) for i in range(2))
            max_lm = max(max_lm, lm_err)
            rows.append({
                "phi_rad": phi,
                "depth": d,
                "condition_xyz": cond2(jt_j(jx)),
                "condition_inverse_depth": cond2(jt_j(jq)),
                "condition_xyz_after_jacobi_scaling": cond2(corr2(jt_j(jx))),
                "condition_idp_after_jacobi_scaling": cond2(corr2(jt_j(jq))),
                "gn_physical_step_error": gn_err,
                "lm_pullback_metric_step_error": lm_err
            })
    return {
        "cases": rows,
        "max_gn_physical_step_error": max_gn,
        "max_lm_pullback_metric_step_error": max_lm
    }


def vdot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def vscale(a, s):
    return [s * x for x in a]


def vnorm(a):
    return math.sqrt(vdot(a, a))


def vunit(a):
    n = vnorm(a)
    return [x / n for x in a]


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def det3(a):
    return (a[0][0] * det2([[a[1][1], a[1][2]], [a[2][1], a[2][2]]])
            - a[0][1] * det2([[a[1][0], a[1][2]], [a[2][0], a[2][2]]])
            + a[0][2] * det2([[a[1][0], a[1][1]], [a[2][0], a[2][1]]]))


def raw_parallax_point(center_main, baseline, bearing, omega):
    """Original chart: X=C_m+(n.b+B*sin(beta)*cot(omega))*n."""
    baseline_norm = vnorm(baseline)
    longitudinal = vdot(bearing, baseline)
    transverse = vsub(baseline, vscale(bearing, longitudinal))
    sin_beta = vnorm(transverse) / baseline_norm
    depth = longitudinal + baseline_norm * sin_beta / math.tan(omega)
    return vadd(center_main, vscale(bearing, depth))


def normalized_parallax_point(center_main, baseline, bearing, nu):
    """Proposed complementary chart: X=C_m+(n.b+B/nu)*n."""
    baseline_norm = vnorm(baseline)
    depth = vdot(bearing, baseline) + baseline_norm / nu
    return vadd(center_main, vscale(bearing, depth))


def tangent_basis(bearing):
    reference = [1.0, 0.0, 0.0] if abs(bearing[0]) < 0.8 else [0.0, 1.0, 0.0]
    e1 = vunit(vcross(reference, bearing))
    e2 = vcross(bearing, e1)
    return e1, e2


def retract_bearing(bearing, e1, e2, x, y):
    rho = math.hypot(x, y)
    if rho == 0.0:
        return bearing[:]
    tangent = vadd(vscale(e1, x / rho), vscale(e2, y / rho))
    return vadd(vscale(bearing, math.cos(rho)), vscale(tangent, math.sin(rho)))


def numeric_chart_determinant(center_main, baseline, bearing, scalar, normalized):
    """Determinant in an orthonormal [e1,e2,n] basis by central differences."""
    e1, e2 = tangent_basis(bearing)
    h = 1e-5
    scalar_h = h * max(1.0, abs(scalar))

    def point(x, y, value):
        n = retract_bearing(bearing, e1, e2, x, y)
        if normalized:
            return normalized_parallax_point(center_main, baseline, n, value)
        return raw_parallax_point(center_main, baseline, n, value)

    columns = []
    for axis in range(3):
        if axis == 0:
            plus, minus = point(h, 0.0, scalar), point(-h, 0.0, scalar)
        elif axis == 1:
            plus, minus = point(0.0, h, scalar), point(0.0, -h, scalar)
        else:
            plus = point(0.0, 0.0, scalar + scalar_h)
            minus = point(0.0, 0.0, scalar - scalar_h)
        step = scalar_h if axis == 2 else h
        delta = vscale(vsub(plus, minus), 1.0 / (2.0 * step))
        columns.append(delta)
    matrix = [[vdot(columns[j], basis) for j in range(3)]
              for basis in (e1, e2, bearing)]
    return det3(matrix)


def normalized_parallax_atlas():
    """Check equivalence, determinant formulas, and aligned-chart coverage."""
    random.seed(20261104)
    max_point_error = 0.0
    max_nu_error = 0.0
    max_raw_det_relative_error = 0.0
    max_normalized_det_relative_error = 0.0
    max_raw_det_absolute_error = 0.0
    max_normalized_det_absolute_error = 0.0
    determinant_cases = 0
    cases = 0
    for _ in range(200):
        center_main = [random.uniform(-2.0, 2.0) for _ in range(3)]
        baseline = [random.uniform(-1.0, 1.0) for _ in range(3)]
        if vnorm(baseline) < 0.2:
            continue
        bearing = vunit([random.uniform(-1.0, 1.0) for _ in range(3)])
        baseline_norm = vnorm(baseline)
        cos_beta = max(-1.0, min(1.0, vdot(bearing, baseline) / baseline_norm))
        beta = math.acos(cos_beta)
        omega = random.uniform(0.08, 1.35)
        signed_depth = (vdot(bearing, baseline) +
                        baseline_norm * math.sin(beta) / math.tan(omega))
        if signed_depth <= 0.0:
            continue
        point = raw_parallax_point(center_main, baseline, bearing, omega)
        depth_vector = vsub(point, center_main)
        depth = vnorm(depth_vector)
        if depth < 1e-6:
            continue
        reconstructed_bearing = vunit(depth_vector)
        sin_beta = math.sin(beta)
        nu_from_transition = math.tan(omega) / sin_beta
        associate_normal_distance = vdot(reconstructed_bearing, vsub(point, vadd(center_main, baseline)))
        nu_from_xyz = baseline_norm / associate_normal_distance
        point_normalized = normalized_parallax_point(
            center_main, baseline, reconstructed_bearing, nu_from_xyz)
        point_error = vnorm(vsub(point, point_normalized))
        max_point_error = max(max_point_error, point_error)
        max_nu_error = max(max_nu_error, abs(nu_from_xyz - nu_from_transition))

        # Finite-difference determinant checks are restricted to a moderately
        # scaled domain; near-zero determinants make relative differences
        # meaningless even when the analytic formula is correct.
        if (0.5 < depth < 20.0 and sin_beta > 0.2 and
                abs(math.sin(omega)) > 0.2 and 0.1 < abs(nu_from_xyz) < 10.0):
            raw_det = numeric_chart_determinant(
                center_main, baseline, bearing, omega, normalized=False)
            raw_formula = depth * depth * baseline_norm * sin_beta / (math.sin(omega) ** 2)
            raw_absolute_error = abs(abs(raw_det) - raw_formula)
            max_raw_det_absolute_error = max(max_raw_det_absolute_error, raw_absolute_error)
            max_raw_det_relative_error = max(
                max_raw_det_relative_error, raw_absolute_error / raw_formula)
            normalized_det = numeric_chart_determinant(
                center_main, baseline, bearing, nu_from_xyz, normalized=True)
            normalized_formula = depth * depth * baseline_norm / (nu_from_xyz ** 2)
            normalized_absolute_error = abs(abs(normalized_det) - normalized_formula)
            max_normalized_det_absolute_error = max(
                max_normalized_det_absolute_error, normalized_absolute_error)
            max_normalized_det_relative_error = max(
                max_normalized_det_relative_error,
                normalized_absolute_error / normalized_formula)
            determinant_cases += 1
        cases += 1

    aligned = []
    center_main = [0.0, 0.0, 0.0]
    baseline = [0.0, 0.0, 1.0]
    bearing = [0.0, 0.0, 1.0]
    for depth in (1.5, 4.0, 100.0):
        point = [0.0, 0.0, depth]
        nu = 1.0 / (depth - 1.0)
        reconstructed = normalized_parallax_point(center_main, baseline, bearing, nu)
        aligned.append({
            "depth": depth,
            "raw_omega": 0.0,
            "normalized_nu": nu,
            "normalized_reconstruction": reconstructed,
            "reconstruction_error": vnorm(vsub(point, reconstructed)),
            "original_scaled_ray_at_beta_zero": scaled_ray_original(0.0, 0.0),
        })
    return {
        "definition": "nu=tan(omega)/sin(beta)=B/(n.(X-C_a)); X=C_m+(n.b+B/nu)n",
        "regular_cases": cases,
        "moderately_scaled_determinant_cases": determinant_cases,
        "max_point_equivalence_error": max_point_error,
        "max_transition_nu_error": max_nu_error,
        "max_raw_determinant_relative_error": max_raw_det_relative_error,
        "max_normalized_determinant_relative_error": max_normalized_det_relative_error,
        "max_raw_determinant_absolute_error": max_raw_det_absolute_error,
        "max_normalized_determinant_absolute_error": max_normalized_det_absolute_error,
        "raw_abs_determinant": "d^2*B*abs(sin(beta))/sin(omega)^2 in an orthonormal tangent basis",
        "normalized_abs_determinant": "d^2*B/nu^2 in an orthonormal tangent basis",
        "aligned_finite_point_coverage": aligned,
        "interpretation": "The normalized chart is exactly equivalent on the common regular domain, but unlike the raw chart it reconstructs aligned finite points without sin(beta), sin(omega), or a scaled-ray zero. It is an inverse-depth-like complementary chart, not new image information.",
    }


def main():
    results = {
        "aligned_2d": aligned_2d_example(),
        "degenerate_3d": degenerate_3d_example(),
        "regularization_continuation": regularization_continuation(),
        "gn_lm_invariance": gn_lm_invariance(),
        "normalized_parallax_atlas": normalized_parallax_atlas()
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, allow_nan=True)
    print("wrote", OUT)
    print("max GN physical-step error:", results["gn_lm_invariance"]["max_gn_physical_step_error"])
    print("max LM pullback-step error:", results["gn_lm_invariance"]["max_lm_pullback_metric_step_error"])
    print("original 3-D ray at beta=omega=0:", results["degenerate_3d"]["original_forward_scaled_ray"])
    atlas = results["normalized_parallax_atlas"]
    print("normalized-parallax cases:", atlas["regular_cases"])
    print("normalized-parallax max point equivalence:", atlas["max_point_equivalence_error"])
    print("normalized-parallax max determinant errors:", atlas["max_raw_determinant_relative_error"], atlas["max_normalized_determinant_relative_error"])


if __name__ == "__main__":
    main()
