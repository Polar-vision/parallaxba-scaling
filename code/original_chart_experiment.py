#!/usr/bin/env python3
"""The parallax chart in the coordinates of the original ParallaxBA paper.

The manuscript analyses a *baseline-adapted* parallax chart q = (alpha, beta,
omega): beta is the angle between the main bearing and the baseline, alpha the
azimuth about the baseline.  Zhao et al. (IJRR 2015, Eqs. 2-3, 7-9) use instead
the azimuth and elevation (psi, theta) of the main bearing in the frame of the
first pose, plus the parallax angle omega; the angle phi between the baseline
and the bearing, which enters the sine law, is then *derived* (Eq. 9).  This
script repeats the chart-dependent experiments of the manuscript with that
original chart ("po") next to the baseline-adapted one ("pa").

Everything chart independent (Fisher information, Cramer-Rao bound, reduced
camera matrix, rank) must agree; everything chart dependent (kappa(H), the
singular values of the chart Jacobian, the shape of the trust region, the
second-order behaviour of finite steps) is allowed to differ and is reported.

  O1  regular rig, undamped GN: iterations, first-step linearised agreement,
      second-order mismatch of the actual update (O1b: cameras perturbed too,
      so that the camera step is nonzero and the anchor coupling G acts)
  O2  angle sweep: kappa(H), Cramer-Rao bound, singular values of D_q, LM
  O2b LM at small angles from the manuscript's isotropic initialization and
      from a depth-only initialization that keeps the measured bearing
  O3  exact alignment
  O6  one damped step: shared lambda versus the pullback metric (experiment S4)
  O4  Simulation 2 re-creation: conditioning, completed Table 3, thresholds
  O5  the aligned limit of the chart Jacobian: baseline-adapted (a polar
      coordinate singularity) versus original (a cone)

Usage:  python code/original_chart_experiment.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_chart_experiment as sce  # noqa: E402
import repro_sim2_ablation as sim2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "original_chart_experiment.json")
SHIPPED_SYN = os.path.join(ROOT, "results", "synthetic_chart_experiment.json")
SHIPPED_SIM2 = os.path.join(ROOT, "results", "repro_sim2_ablation.json")

CH = ("xyz", "id", "ir", "pa", "po")
OLD = ("xyz", "id", "ir", "pa")


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(1e-300, np.linalg.norm(b)))


# --------------------------------------------------------------------------
# O1: undamped GN on the regular rig
# --------------------------------------------------------------------------
def first_step(init, chart):
    """Linearised physical step and the actual chart update from `init`."""
    layout = sce.cam_layout(len(init["cams"]))
    n_cv = layout[1]
    p = sce.init_params(init, chart)
    H, g, _ = sce.hessian_at(chart, init, p, layout)
    dc, dp, _ = sce.schur_solve(H, g, n_cv, len(p))
    dX = sce.physical_step(chart, p, dp, init["cams"], init["Rs"], layout, dc)
    cams2, Rs2, p2 = sce.apply_step(init["cams"], init["Rs"], p, layout, dc, dp)
    Xn = np.array([sce.chart_forward(chart, q, cams2, Rs2) for q in p2])
    X0 = np.array([sce.chart_forward(chart, q, init["cams"], init["Rs"]) for q in p])
    return dc, dX, Xn, X0


def perturb_cameras(state, rng, rot=0.01, tra=0.03):
    """Perturb the free cameras (gauge: camera 0 and the translation of camera 1 stay)."""
    cams = [c.copy() for c in state["cams"]]
    Rs = [R.copy() for R in state["Rs"]]
    for i in range(1, len(cams)):
        Rs[i] = sce.expm_so3(rng.normal(scale=rot, size=3)) @ Rs[i]
        if i >= 2:
            cams[i] = cams[i] + rng.normal(scale=tra, size=3)
    return {"cams": cams, "Rs": Rs, "pts": [X.copy() for X in state["pts"]],
            "obs": list(state["obs"]), "u": state["u"].copy(), "probe": state.get("probe")}


def _run_o1(init, dtrue):
    rows, base, steps = {}, None, {}
    for chart in CH:
        res = sce.solve(init, chart, method="gn", max_iter=25)
        X = res.get("X")
        rows[chart] = {"status": res["status"], "iter": res["iter"],
                       "final_cost": res.get("cost")}
        if X is not None:
            rows[chart]["max_depth_rel_error"] = float(
                np.max(np.abs(np.linalg.norm(X, axis=1) - dtrue) / dtrue))
            if base is None:
                base = X
            else:
                rows[chart]["position_difference_vs_xyz_m"] = float(np.max(np.abs(X - base)))
        steps[chart] = first_step(init, chart)
    dc_x, dX_x, Xn_x, X0_x = steps["xyz"]
    full_x = np.concatenate([np.ravel(dc_x), np.ravel(dX_x)])
    for chart in CH:
        dc, dX, Xn, X0 = steps[chart]
        rows[chart]["first_step_camera_norm"] = float(np.linalg.norm(dc))
        rows[chart]["first_step_point_norm"] = float(np.linalg.norm(dX))
        rows[chart]["first_step_physical_rel_diff"] = rel(
            np.concatenate([np.ravel(dc), np.ravel(dX)]), full_x)
        rows[chart]["first_step_camera_rel_diff"] = rel(dc, dc_x)
        rows[chart]["first_step_point_rel_diff"] = rel(dX, dX_x)
        rows[chart]["second_order_mismatch"] = float(
            np.linalg.norm(Xn - (X0_x + dX_x)) / np.linalg.norm(dX_x))
    return rows


def experiment_o1():
    """Cameras at the truth (as in the manuscript): the camera step is rounding noise."""
    prob = sce.scene_regular()
    init = sce.perturb_shared(prob, np.random.default_rng(101), rel=0.04)
    return _run_o1(init, np.linalg.norm(np.array(prob["pts"]), axis=1))


def experiment_o1b():
    """Points and free cameras perturbed: nonzero camera step, anchor coupling G != 0."""
    prob = sce.scene_regular()
    init = sce.perturb_shared(prob, np.random.default_rng(101), rel=0.04)
    init = perturb_cameras(init, np.random.default_rng(202))
    return _run_o1(init, np.linalg.norm(np.array(prob["pts"]), axis=1))


# --------------------------------------------------------------------------
# O2: angle sweep
# --------------------------------------------------------------------------
def sweep_row(bdeg):
    b = math.radians(bdeg)
    direction = np.array([math.sin(b), 0.0, math.cos(b)])
    prob = sce.scene_forward(probe_dir=direction, probe_depth=12.0, noise=sce.NOISE)
    probe = prob["probe"]
    n = sce.unit(prob["pts"][probe])
    dtrue = float(np.linalg.norm(prob["pts"][probe]))
    layout = sce.cam_layout(len(prob["cams"]))
    entry = {"beta_deg": bdeg}
    for chart in CH:
        c = sce.crb(chart, prob, layout, probe, n)
        entry[chart] = c if c is not None else {"status": "chart_undefined"}
        p = sce.init_params(prob, chart)
        if p is not None:
            _, Jx, _ = sce.chart_jacobian(chart, p[probe], prob["cams"], prob["Rs"])
            sv = np.linalg.svd(Jx, compute_uv=False)
            entry[chart]["svd_D"] = [float(x) for x in sv]
            entry[chart]["cond_D"] = float(sv[0] / sv[-1])
    init = sce.perturb_shared(prob, np.random.default_rng(303), rel=0.05)
    for chart in CH:
        res = sce.solve(init, chart, method="lm", max_iter=150)
        d = entry[chart]
        d["lm_status"] = res["status"]
        d["lm_iter"] = res["iter"]
        if res.get("X") is not None:
            dhat = float(np.linalg.norm(res["X"][probe]))
            d["lm_depth_rel_error"] = abs(dhat - dtrue) / dtrue
            d["lm_rmse_px"] = sce.FOCAL * math.sqrt(2.0 * res["cost"] / (2 * len(prob["obs"])))
    return entry


def experiment_o2():
    return [sweep_row(b) for b in (80.0, 45.0, 20.0, 8.0, 3.0, 1.0, 0.3, 0.1)]


# --------------------------------------------------------------------------
# O2b: LM from two initializations at small angles
# --------------------------------------------------------------------------
def depth_only_init(prob, rng, spread):
    """Move every point along its own bearing from camera 0: depth error only."""
    c0 = prob["cams"][0]
    pts = [c0 + (1.0 + rng.uniform(-spread, spread)) * (X - c0) for X in prob["pts"]]
    return {"cams": [c.copy() for c in prob["cams"]], "Rs": [R.copy() for R in prob["Rs"]],
            "pts": pts, "obs": list(prob["obs"]), "u": prob["u"].copy(),
            "probe": prob.get("probe")}


def _angle_deg(u, v):
    u, v = sce.unit(u), sce.unit(v)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def experiment_o2b():
    """The shared isotropic perturbation of the manuscript moves the probe bearing by
    about 3 degrees; below that angle the optimiser has to pass the cone tip phi = 0 of
    the original chart.  A depth-only perturbation keeps the (measured) bearing."""
    rows = []
    for bdeg in (8.0, 3.0, 1.0, 0.3, 0.1):
        b = math.radians(bdeg)
        prob = sce.scene_forward(probe_dir=np.array([math.sin(b), 0.0, math.cos(b)]),
                                 probe_depth=12.0, noise=sce.NOISE)
        probe = prob["probe"]
        dtrue = float(np.linalg.norm(prob["pts"][probe]))
        n_res = 2 * len(prob["obs"])
        axis = np.array([0.0, 0.0, 1.0])
        iso = sce.perturb_shared(prob, np.random.default_rng(303), rel=0.05)
        dep = depth_only_init(prob, np.random.default_rng(404), 0.20)
        entry = {"beta_deg": bdeg,
                 "isotropic_probe_bearing_error_deg": _angle_deg(iso["pts"][probe], prob["pts"][probe]),
                 "isotropic_probe_angle_to_axis_deg": _angle_deg(iso["pts"][probe], axis),
                 "depth_only_probe_bearing_error_deg": _angle_deg(dep["pts"][probe], prob["pts"][probe])}
        for tag, init in (("isotropic", iso), ("depth_only", dep)):
            entry[tag] = {}
            for chart in CH:
                res = sce.solve(init, chart, method="lm", max_iter=150)
                d = {"status": res["status"], "iter": res["iter"]}
                if res.get("X") is not None:
                    dhat = float(np.linalg.norm(res["X"][probe]))
                    d["rmse_px"] = sce.FOCAL * math.sqrt(2.0 * res["cost"] / n_res)
                    d["depth_rel_error"] = abs(dhat - dtrue) / dtrue
                    d["exit_gradient_max"] = res.get("gmax")
                    d["exit_lambda"] = res.get("lam")
                entry[tag][chart] = d
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------
# O3: exact alignment
# --------------------------------------------------------------------------
def experiment_o3():
    direction = np.array([0.0, 0.0, 1.0])
    prob = sce.scene_forward(probe_dir=direction, probe_depth=12.0, noise=sce.NOISE)
    probe = prob["probe"]
    n = sce.unit(prob["pts"][probe])
    layout = sce.cam_layout(len(prob["cams"]))
    rows = {}
    for chart in CH:
        c = sce.crb(chart, prob, layout, probe, n)
        rows[chart] = c if c is not None else {"status": "chart_undefined",
                                               "reason": "sin(omega)=0 at exact alignment"}
    return rows


# --------------------------------------------------------------------------
# O4: Simulation 2 re-creation
# --------------------------------------------------------------------------
def experiment_o4(charts):
    scene = sim2.build_scene("forward")
    truth = np.array(scene["pts"])
    init = sim2.triangulate(scene)
    out = {"R1_conditioning": {}, "R2_completed_table": {}, "R3_tolerance": {}, "R4_lateral": {}}
    for chart in charts:
        out["R1_conditioning"][chart] = sim2.conditioning(scene, chart)
        for method in ("gn", "lm"):
            out["R2_completed_table"].setdefault(chart, {})[method] = sim2.summarize(
                sim2.run(sim2.make_state(scene, init), chart, method), scene)
        out["R3_tolerance"][chart] = {}
        for s in (1.0, 1.02, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 2.0, 3.0, 5.0):
            st = sim2.make_state(scene, sim2.scale_axial(scene, truth, s))
            out["R3_tolerance"][chart]["%g" % s] = sim2.summarize(sim2.run(st, chart, "gn"), scene)
    lat = sim2.build_scene("lateral")
    lat_truth = np.array(lat["pts"])
    for chart in charts:
        out["R4_lateral"][chart] = {}
        for s in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0):
            st = sim2.make_state(lat, sim2.scale_axial(lat, lat_truth, s))
            out["R4_lateral"][chart]["%g" % s] = sim2.summarize(sim2.run(st, chart, "gn"), lat)
    return out


# --------------------------------------------------------------------------
# O6: one damped step, shared lambda versus the pullback metric
# --------------------------------------------------------------------------
def experiment_o6(charts=("id", "ir", "pa", "po")):
    """The experiment S4 of synthetic_chart_experiment.py, with the original coordinates added.

    From one shared initialization at beta = 1 degree, take one damped step in each chart
    with (i) the same lambda in chart coordinates and (ii) lambda interpreted through the
    pullback metric T^{-T} T^{-1}; compare the physical step with the XYZ step (i) and with
    the physical reference H_x + lambda T^{-T}T^{-1} (ii)."""
    b = math.radians(1.0)
    direction = np.array([math.sin(b), 0.0, math.cos(b)])
    prob = sce.scene_forward(probe_dir=direction, probe_depth=12.0, noise=sce.NOISE)
    init = sce.perturb_shared(prob, np.random.default_rng(404), rel=0.04)
    layout = sce.cam_layout(len(init["cams"]))
    px = sce.init_params(init, "xyz")
    Hx, gx, _ = sce.hessian_at("xyz", init, px, layout)
    rows = []
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        dx_naive = -np.linalg.solve(Hx + lam * np.eye(Hx.shape[0]), gx)
        rec = {"lambda": lam, "same_lambda_everywhere": {}, "pullback_metric": {},
               "step_norm": {"xyz": float(np.linalg.norm(dx_naive))}}
        for chart in charts:
            p = sce.init_params(init, chart)
            Hq, gq, _ = sce.hessian_at(chart, init, p, layout)
            T = sce.build_T(chart, p, init["cams"], init["Rs"], layout, len(p))
            dq = -np.linalg.solve(Hq + lam * np.eye(Hq.shape[0]), gq)
            dx_q = T @ dq
            rec["step_norm"][chart] = float(np.linalg.norm(dx_q))
            rec["same_lambda_everywhere"][chart] = rel(dx_q, dx_naive)
            Ti = np.linalg.inv(T)
            dx_ref = -np.linalg.solve(Hx + lam * (Ti.T @ Ti), gx)
            rec["pullback_metric"][chart] = rel(dx_q, dx_ref)
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# O5: the aligned limit of the chart Jacobian
# --------------------------------------------------------------------------
def experiment_o5():
    """Jacobian of each chart at points close to the baseline axis.

    The point sits at distance 12 m from camera 0 on a cone of half-angle phi
    about the baseline (z axis, associate camera 2 m ahead).  For every phi the
    approach azimuth a about the baseline is varied.  Baseline-adapted chart:
    singular values (sigma_alpha = d sin(beta) -> 0, rank loss, independent of a).
    Original chart: singular values stay bounded away from zero but the Jacobian
    itself depends on a at O(1) however small phi is: phi = arccos(n.b) is the
    geodesic distance from the baseline direction, a cone.
    """
    cams = [np.array([0.0, 0.0, 0.4 * k]) for k in range(6)]
    Rs = [np.eye(3) for _ in range(6)]
    rows = []
    for phi_deg in (10.0, 1.0, 0.1, 0.01):
        phi = math.radians(phi_deg)
        per_chart = {}
        for chart in ("pa", "po"):
            Js, svs = [], []
            for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
                n = np.array([math.sin(phi) * math.cos(a), math.sin(phi) * math.sin(a), math.cos(phi)])
                X = 12.0 * n
                p = sce.chart_inverse(chart, X, cams, Rs)
                _, Jx, _ = sce.chart_jacobian(chart, p, cams, Rs)
                Js.append(Jx)
                svs.append(np.linalg.svd(Jx, compute_uv=False))
            spread = max(np.linalg.norm(Js[i] - Js[0]) for i in range(1, 4)) / np.linalg.norm(Js[0])
            per_chart[chart] = {"singular_values_a0": [float(x) for x in svs[0]],
                                "cond_D_a0": float(svs[0][0] / svs[0][-1]),
                                "jacobian_variation_over_azimuth": float(spread)}
        rows.append({"phi_deg": phi_deg, **per_chart})
    return rows


# --------------------------------------------------------------------------
def consistency(o2, o4, o6):
    """The four charts of the manuscript must be reproduced exactly."""
    out = {}
    try:
        with open(SHIPPED_SYN, encoding="utf-8") as f:
            s4 = json.load(f)["S4_damping_metric"]
        worst = 0.0
        for new, old in zip(o6, s4):
            for key in ("same_lambda_everywhere", "pullback_metric"):
                for chart in ("id", "ir", "pa"):
                    a, b = new[key][chart], old[key][chart]
                    worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
        out["damping_rel_diff_vs_shipped"] = worst
    except (OSError, KeyError, ValueError):
        out["damping"] = "shipped synthetic_chart_experiment.json not found"
    try:
        with open(SHIPPED_SYN, encoding="utf-8") as f:
            syn = json.load(f)
        worst_k, worst_s = 0.0, 0.0
        for new, old in zip(o2, syn["S2_angle_sweep"]):
            for chart in OLD:
                a, b = new[chart], old[chart]
                if "kappa_H" in a and "kappa_H" in b:
                    worst_k = max(worst_k, abs(a["kappa_H"] - b["kappa_H"]) / b["kappa_H"])
                if a.get("sigma_depth_m") and b.get("sigma_depth_m"):
                    worst_s = max(worst_s, abs(a["sigma_depth_m"] - b["sigma_depth_m"]) / b["sigma_depth_m"])
        out["sweep_kappa_rel_diff_vs_shipped"] = worst_k
        out["sweep_sigma_rel_diff_vs_shipped"] = worst_s
    except (OSError, KeyError, ValueError):
        out["sweep"] = "shipped synthetic_chart_experiment.json not found"
    try:
        with open(SHIPPED_SIM2, encoding="utf-8") as f:
            s2 = json.load(f)
        worst = 0.0
        for chart in OLD:
            if chart in o4["R1_conditioning"]:
                a = o4["R1_conditioning"][chart]["cond_H"]
                b = s2["R1_conditioning"][chart]["cond_H"]
                worst = max(worst, abs(a - b) / b)
        out["sim2_cond_H_rel_diff_vs_shipped"] = worst
    except (OSError, KeyError, ValueError):
        out["sim2"] = "shipped repro_sim2_ablation.json not found"
    return out


def main():
    for tag, fn in (("O1 regular rig, undamped GN (cameras at the truth)", experiment_o1),
                    ("O1b regular rig, undamped GN (points and cameras perturbed)", experiment_o1b)):
        print(tag)
        rows = fn()
        if tag.startswith("O1 "):
            o1 = rows
        else:
            o1b = rows
        for ch in CH:
            r = rows[ch]
            print("  %-4s it=%2d cost=%.1e  1st-step physical diff %.1e (cam norm %.1e, pt norm %.1e)"
                  "  2nd-order mismatch %.2e"
                  % (ch, r["iter"], r["final_cost"] or float("nan"), r["first_step_physical_rel_diff"],
                     r["first_step_camera_norm"], r["first_step_point_norm"], r["second_order_mismatch"]))
    print("O2 angle sweep")
    o2 = experiment_o2()
    for r in o2:
        print("  beta=%5.1f" % r["beta_deg"] + "".join(
            "  %s k=%.1e sd=%s" % (ch, r[ch].get("kappa_H", float("nan")),
                                   "None" if not r[ch].get("sigma_depth_m") else "%.3e" % r[ch]["sigma_depth_m"])
            for ch in ("pa", "po")))
    print("O2b LM from isotropic and depth-only initializations")
    o2b = experiment_o2b()
    for r in o2b:
        print("  beta=%5.1f  probe bearing error (isotropic) %.2f deg, angle to axis %.2f deg" % (
            r["beta_deg"], r["isotropic_probe_bearing_error_deg"], r["isotropic_probe_angle_to_axis_deg"]))
        for tag in ("isotropic", "depth_only"):
            print("    %-10s" % tag + "".join(
                " | %s %s rmse=%s |g|=%s" % (ch, r[tag][ch]["status"][:4],
                                             "%.3f" % r[tag][ch]["rmse_px"] if "rmse_px" in r[tag][ch] else "-",
                                             "%.1e" % r[tag][ch]["exit_gradient_max"]
                                             if r[tag][ch].get("exit_gradient_max") is not None else "-")
                for ch in CH))
    print("O3 exact alignment")
    o3 = experiment_o3()
    print("  ", {k: v.get("status", "ok") for k, v in o3.items()})
    print("O5 aligned limit of the chart Jacobian")
    o5 = experiment_o5()
    for r in o5:
        print("  phi=%6.2f  pa: cond_D=%.2e var=%.2e | po: cond_D=%.2e var=%.2e"
              % (r["phi_deg"], r["pa"]["cond_D_a0"], r["pa"]["jacobian_variation_over_azimuth"],
                 r["po"]["cond_D_a0"], r["po"]["jacobian_variation_over_azimuth"]))
    print("O6 one damped step, shared lambda versus pullback metric")
    o6 = experiment_o6()
    for r in o6:
        print("  lam=%.0e  same lambda %s | pullback %s" % (
            r["lambda"], {c: "%.2e" % v for c, v in r["same_lambda_everywhere"].items()},
            {c: "%.1e" % v for c, v in r["pullback_metric"].items()}))
    print("O4 Simulation 2 (this takes a while)")
    o4 = experiment_o4(CH)
    for ch in CH:
        c = o4["R1_conditioning"][ch]
        print("  %-4s cond_H=%.3e cond_S=%.6f rank_S=%s/%s" % (ch, c.get("cond_H", float("nan")),
                                                                c.get("cond_S", float("nan")), c.get("rank_S"), c.get("n_S")))
    result = {
        "protocol": {"charts": list(CH),
                     "po": "original coordinates: (psi, theta) azimuth/elevation of the main bearing in the frame "
                           "of camera 0, plus omega; phi = arccos(n.b_hat) derived (Zhao et al. 2015, Eqs. 2-3, 7-9)",
                     "pa": "baseline-adapted coordinates of the manuscript",
                     "anchors": "camera 0 and the last camera",
                     "gauge": "camera 0 fixed, camera 1 translation fixed"},
        "O1_regular_rig": o1, "O1b_regular_rig_cameras_perturbed": o1b,
        "O2_angle_sweep": o2, "O2b_small_angle_lm": o2b, "O3_exact_alignment": o3,
        "O4_sim2": o4, "O5_aligned_jacobian": o5, "O6_damping_metric": o6,
        "consistency_with_shipped": consistency(o2, o4, o6),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("consistency:", result["consistency_with_shipped"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
