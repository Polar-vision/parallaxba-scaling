"""Static consistency checks for the LaTeX manuscript and generated evidence."""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_PATH = os.path.join(ROOT, "main.tex")
BIB_PATH = os.path.join(ROOT, "references.bib")

with open(TEX_PATH, "r", encoding="utf-8") as f:
    tex = f.read()
with open(BIB_PATH, "r", encoding="utf-8") as f:
    bib = f.read()

errors = []

# Check braces while ignoring escaped literal braces.
balance = 0
for i, char in enumerate(tex):
    if char in "{}" and (i == 0 or tex[i - 1] != "\\"):
        balance += 1 if char == "{" else -1
        if balance < 0:
            errors.append("Closing brace before opening brace at offset %d" % i)
            break
if balance != 0:
    errors.append("Unbalanced brace count: %d" % balance)

# Check nested environments.
env_stack = []
for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
    action, env = match.group(1), match.group(2)
    if action == "begin":
        env_stack.append(env)
    elif not env_stack or env_stack[-1] != env:
        errors.append("Environment mismatch at \\end{%s}" % env)
    else:
        env_stack.pop()
if env_stack:
    errors.append("Unclosed environments: %r" % env_stack)

# Check citations and references against the local BibTeX database.
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
cite_keys = set()
for group in re.findall(r"\\cite\{([^}]+)\}", tex):
    cite_keys.update(k.strip() for k in group.split(","))
missing = sorted(cite_keys - bib_keys)
if missing:
    errors.append("Missing bibliography keys: %r" % missing)
labels = re.findall(r"\\label\{([^}]+)\}", tex)
if len(labels) != len(set(labels)):
    errors.append("Duplicate labels")
refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
missing_refs = sorted(refs - set(labels))
if missing_refs:
    errors.append("Missing labels: %r" % missing_refs)

# Check simple tabular column counts used by this manuscript.
def _tabular_blocks(text):
    """Yield (spec, body) for each tabular, honouring nested braces in the spec."""
    for match in re.finditer(r"\\begin\{tabular\}", text):
        i = match.end()
        while i < len(text) and text[i] in " \t":
            i += 1
        if i < len(text) and text[i] == "[":
            i = text.index("]", i) + 1
        while i < len(text) and text[i] in " \t":
            i += 1
        if i >= len(text) or text[i] != "{":
            continue
        depth, start = 0, i + 1
        while i < len(text):
            ch = text[i]
            if ch in "{}" and text[i - 1] != "\\":
                depth += 1 if ch == "{" else -1
                if depth == 0:
                    break
            i += 1
        body_end = text.find("\\end{tabular}", i)
        yield text[start:i], text[i + 1:body_end]


def _top_level_ampersands(line):
    depth, count = 0, 0
    for j, ch in enumerate(line):
        if ch in "{}" and (j == 0 or line[j - 1] != "\\"):
            depth += 1 if ch == "{" else -1
        elif ch == "&" and depth == 0 and (j == 0 or line[j - 1] != "\\"):
            count += 1
    return count


for spec, body in _tabular_blocks(tex):
    depth, expected = 0, 0
    for ch in spec:
        if ch in "{}":
            depth += 1 if ch == "{" else -1
        elif depth == 0 and ch in "lcrpmbX":
            expected += 1
    for line in body.splitlines():
        line = line.strip()
        line = re.sub(r"\\begin\{smallmatrix\}.*?\\end\{smallmatrix\}", "", line)
        if not line or (line.startswith("\\") and "&" not in line):
            continue
        if "\\multicolumn" in line:
            continue
        if "&" in line:
            actual = _top_level_ampersands(line) + 1
            if actual != expected:
                errors.append("Tabular has %d columns, row has %d: %s" %
                              (expected, actual, line[:100]))

if re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b", tex, re.I):
    errors.append("Unresolved TODO/TBD/PLACEHOLDER token")

# Trace the synthetic-experiment tables back to the generated JSON.
with open(os.path.join(ROOT, "results", "synthetic_chart_experiment.json"),
          "r", encoding="utf-8") as f:
    syn = json.load(f)

s1 = syn["S1_gn_invariance"]
if s1["max_abs_position_difference_m"] > 1e-12:
    errors.append("Chart invariance of the GN iterates exceeds tolerance: %g"
                  % s1["max_abs_position_difference_m"])
for chart in ("xyz", "id", "ir", "pa"):
    if s1["charts"][chart]["status"] != "converged":
        errors.append("GN invariance run did not converge for chart %s" % chart)

sweep = syn["S2_angle_sweep"]
worst_spread = 0.0
kappa_min, kappa_max = float("inf"), 0.0
for row in sweep:
    vals = [row[c]["sigma_depth_m"] for c in ("xyz", "id", "ir", "pa")]
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    worst_spread = max(worst_spread, spread)
    for c in ("xyz", "id", "ir", "pa"):
        kappa_min = min(kappa_min, row[c]["kappa_H"])
        kappa_max = max(kappa_max, row[c]["kappa_H"])
if worst_spread > 1e-8:
    errors.append("Depth CRB is not chart invariant: spread %g" % worst_spread)
if kappa_max / kappa_min < 1e5:
    errors.append("Chart condition numbers do not separate as claimed")
for row in sweep:
    if row["id"]["kappa_H"] >= row["pa"]["kappa_H"]:
        errors.append("Inverse depth is not the better conditioned chart at beta=%s"
                      % row["beta_deg"])

# Where all four charts converge to the same basin they must reach the same
# attainable minimum.  A run that stops while its probe has diverged is a
# false positive (see the parallax 1-degree case below) and is excluded here.
agreeing = 0
for row in sweep:
    chans = [row[c] for c in ("xyz", "id", "ir", "pa")]
    if not all(c["lm_status"] == "converged" and c["lm_depth_rel_error"] < 10.0
               for c in chans):
        continue
    depth = [c["lm_depth_rel_error"] for c in chans]
    rmse = [c["lm_rmse_px"] for c in chans]
    if (max(depth) - min(depth)) / max(depth) > 1e-6:
        errors.append("Converged charts disagree on depth error at beta=%s"
                      % row["beta_deg"])
    if (max(rmse) - min(rmse)) / max(rmse) > 1e-6:
        errors.append("Converged charts disagree on RMSE at beta=%s" % row["beta_deg"])
    if max(depth) / max(1e-300, min(depth)) < 1.5:
        agreeing += 1
if agreeing < 5:
    errors.append("Too few angles where all four charts converge: %d" % agreeing)

# The manuscript claims a false convergence for the parallax chart at 1 degree.
one = [r for r in sweep if abs(r["beta_deg"] - 1.0) < 1e-9]
if one:
    pa = one[0]["pa"]
    if not (pa["lm_status"] == "converged" and pa["lm_depth_rel_error"] > 1e10
            and pa["lm_rmse_px"] < 0.5):
        errors.append("Parallax false-convergence example no longer holds at 1 degree")

align = syn["S3_exact_alignment"]["rows"]
for chart in ("xyz", "id", "ir"):
    if align[chart]["nullity_S"] != 0:
        errors.append("Reduced camera system lost rank at alignment (%s)" % chart)
if align["pa"].get("status") != "chart_undefined":
    errors.append("Parallax chart is reported as defined at exact alignment")

damp = syn["S4_damping_metric"]
if max(max(r["pullback_metric"].values()) for r in damp) > 1e-10:
    errors.append("Pullback damping identity exceeds tolerance")
if max(max(r["same_lambda_everywhere"].values()) for r in damp) < 1.0:
    errors.append("Same-lambda comparison does not separate the charts as claimed")
# As lambda -> 0 the shared-lambda comparison must collapse onto the undamped
# Gauss-Newton step; this is an independent check of the invariance theorem.
tiny = min(damp, key=lambda r: r["lambda"])
spread = (max(tiny["step_norm"].values()) - min(tiny["step_norm"].values())) \
    / min(tiny["step_norm"].values())
if spread > 1e-3:
    errors.append("Damped steps do not converge to the GN step as lambda -> 0: %g" % spread)
if max(r["step_norm"]["pa"] / r["step_norm"]["xyz"] for r in damp) < 4.0:
    errors.append("Unit-damping step ratio is smaller than the manuscript states")

blind = syn["S5_depth_blindness"]
five = [r for r in blind if r["depth_ratio"] == 5.0]
if min(r["rmse_px"] for r in five) > 0.5:
    errors.append("A fivefold depth error is not sub-pixel at small beta")

for literal in ("$3.7\\times10^{-14}$", "$5.3\\times10^{10}$", "$4.60$",
                "$2900$", "$4.88$", "$9.3\\times10^{-12}$",
                "$0.84$", "$2.8\\times10^{-10}$"):
    if literal not in tex:
        errors.append("Synthetic-experiment value missing from manuscript: %s" % literal)

with open(os.path.join(ROOT, "results", "controlled_results.json"), "r", encoding="utf-8") as f:
    controlled = json.load(f)
normalized_checks = controlled["normalized_parallax_atlas"]
if normalized_checks["max_point_equivalence_error"] > 1e-12:
    errors.append("Normalized-chart point equivalence check exceeds tolerance")
if normalized_checks["max_raw_determinant_relative_error"] > 1e-7:
    errors.append("Raw determinant finite-difference check exceeds tolerance")
if normalized_checks["max_normalized_determinant_relative_error"] > 1e-7:
    errors.append("Normalized determinant finite-difference check exceeds tolerance")


def _literal(x):
    """Format a float the way the manuscript writes it: 2.18e-15 -> 2.18\\times10^{-15}."""
    mantissa, exponent = ("%.2e" % x).split("e")
    return "%s\\times10^{%d}" % (mantissa, int(exponent))


for field in ("max_point_equivalence_error", "max_transition_nu_error",
              "max_raw_determinant_relative_error",
              "max_normalized_determinant_relative_error"):
    if _literal(normalized_checks[field]) not in tex:
        errors.append("Normalized-chart value missing from manuscript: %s"
                      % _literal(normalized_checks[field]))

with open(os.path.join(ROOT, "results", "ijrr_3d_ba_theorem.json"), "r", encoding="utf-8") as f:
    theorem = json.load(f)
exp_a = theorem["experiment_A_generic_ba"]
exp_b = theorem["experiment_B_collinear_point"]
exp_c = theorem["experiment_C_sparsity"]
exp_d = theorem["experiment_D_damped_lm"]

for key, tol in (("max_reconstruction_error", 1e-13),
                 ("max_Hq_minus_TtHxT", 1e-13),
                 ("max_Sq_minus_Sx", 1e-13),
                 ("max_gradient_congruence_error", 1e-14),
                 ("max_reduced_gradient_difference", 1e-13),
                 ("damped_full_ba_metric_identity_error", 1e-11)):
    if exp_a[key] > tol:
        errors.append("BA-theorem identity %s exceeds tolerance: %g" % (key, exp_a[key]))
if exp_a["rank_Hx"] != exp_a["rank_Hq"]:
    errors.append("Congruence does not preserve rank of H")
if exp_a["rank_Sx"] != exp_a["rank_Sq"]:
    errors.append("Reduced camera matrix is not rank invariant across charts")
if exp_a["rank_identity_Hx_minus_rank_JX"] != exp_a["rank_Sx"]:
    errors.append("Rank identity rank(S) = rank(H) - rank(J_X) fails")
if exp_a["inertia_Hx"] != exp_a["inertia_Hq"]:
    errors.append("Congruence does not preserve inertia")

base_b = exp_b["base"]
aug_b = exp_b["augmented_with_collinear_point"]
if aug_b["rank_S"] != base_b["rank_S"] or aug_b["nullity_S"] != base_b["nullity_S"]:
    errors.append("Collinear point changed the reduced camera system rank")
if aug_b["nullity_H"] != base_b["nullity_H"] + 1:
    errors.append("Collinear point did not add exactly one null direction")
if aug_b["point_block_rank"] != 2:
    errors.append("Collinear point block is not rank two")

for case in ("full_visibility_5cam", "partial_visibility_6cam"):
    if not exp_c[case]["block_patterns_identical"]:
        errors.append("Schur block pattern differs between charts (%s)" % case)

if exp_d["metric_identity_max_error"] > 1e-10:
    errors.append("Damping metric identity exceeds tolerance")

for literal in ("2.66\\times10^{-15}", "1.78\\times10^{-15}", "7.11\\times10^{-15}",
                "2.22\\times10^{-16}", "1.67\\times10^{-13}", "4.42\\times10^{-12}",
                "68-45=23", "-1.9999", "8.0002"):
    if literal not in tex:
        errors.append("Theorem verification value missing from manuscript: %s" % literal)
for row in ("$\\rank H$ & 32 & 34 & $+2$",
            "$\\rank\\mathcal S$ & 17 & 17 & $0$",
            "$\\rank J_X$ & 15 & 17 & $+2$"):
    if row not in tex:
        errors.append("Missing or mismatched rank-bookkeeping row: %s" % row)

# --------------------------------------------------------------------------
# Reproduction of the original Simulation 2 with the missing XYZ control
# --------------------------------------------------------------------------
with open(os.path.join(ROOT, "results", "repro_sim2_ablation.json"),
          "r", encoding="utf-8") as f:
    repro = json.load(f)

cond = repro["R1_conditioning"]
cs = [cond[c]["cond_S"] for c in ("xyz", "id", "ir", "pa")]
if (max(cs) - min(cs)) / min(cs) > 1e-6:
    errors.append("Reduced camera system is not chart invariant on the "
                  "Simulation 2 geometry: spread %g"
                  % ((max(cs) - min(cs)) / min(cs)))
ch = [cond[c]["cond_H"] for c in ("xyz", "id", "ir", "pa")]
if max(ch) / min(ch) < 100.0:
    errors.append("Chart Hessian condition numbers do not separate on the "
                  "Simulation 2 geometry")
ranks = {cond[c]["rank_S"] for c in ("xyz", "id", "ir", "pa")}
if len(ranks) != 1 or ranks.pop() != cond["xyz"]["n_S"]:
    errors.append("Reduced system rank differs across charts")

for chart in ("xyz", "id", "ir", "pa"):
    for method in ("gn", "lm"):
        row = repro["R2_completed_table"][chart][method]
        if row["status"] != "converged":
            errors.append("Chart %s did not converge under %s from the shared "
                          "initialization" % (chart, method))
    if repro["R3_tolerance"][chart]["1.1"]["status"] != "converged":
        errors.append("Chart %s does not survive a 1.10 depth scale" % chart)

for chart in ("xyz", "id", "ir"):
    if repro["R3_tolerance"][chart]["1.15"]["status"] == "converged":
        errors.append("Chart %s survives 1.15 under forward motion, "
                      "contradicting the shared-threshold claim" % chart)
if repro["R3_tolerance"]["pa"]["5"]["status"] != "converged":
    errors.append("Parallax chart does not survive a factor of five forward")
if repro["R4_lateral"]["xyz"]["2"]["status"] == "converged":
    errors.append("XYZ survives a factor of two laterally, contradicting the claim")
for chart in ("id", "ir", "pa"):
    if repro["R4_lateral"][chart]["5"]["status"] != "converged":
        errors.append("Chart %s does not survive a factor of five laterally" % chart)

# ---------------------------------------------------------------- real data
real_path = os.path.join(ROOT, "results", "real_invariance.json")
if os.path.exists(real_path):
    with open(real_path, encoding="utf-8") as fh:
        real = json.load(fh)
    rw = [r for r in real["windows"] if r.get("status") == "ok"]
    if len(rw) < 9:
        errors.append("Real-sequence experiment has only %d usable windows" % len(rw))
    s_rel = [c["S_rel"] for r in rw for c in r["charts"].values()
             if c.get("status") == "ok"]
    step_rel = [c["step_rel"] for r in rw for c in r["charts"].values()
                if c.get("status") == "ok"]
    if max(s_rel) > 1e-12:
        errors.append("Reduced camera system is not chart-invariant on real data")
    if max(step_rel) > 1e-8:
        errors.append("Physical Gauss-Newton step is not chart-invariant on real data")
    spreads = []
    for r in rw:
        k = [c["kappa_H"] for c in r["charts"].values() if c.get("status") == "ok"]
        spreads.append(max(k) / min(k))
    if max(spreads) < 100:
        errors.append("Real-data condition numbers do not separate the charts")
    real_summary = "S %.1e, step %.1e, kappa spread %.0fx, %d windows" % (
        max(s_rel), max(step_rel), max(spreads), len(rw))
else:
    real_summary = "skipped (results/real_invariance.json absent)"

# ------------------------------------------------------- real depth (Sec. 12)
depth_path = os.path.join(ROOT, "results", "real_depth_collapse.json")
if os.path.exists(depth_path):
    with open(depth_path, encoding="utf-8") as fh:
        depth = json.load(fh)
    dp = depth["solvers"]["pts_only"]
    if dp["n_windows_ok"] < 15:
        errors.append("Depth experiment has only %d usable windows"
                      % dp["n_windows_ok"])
    if dp["n_points"]["xyz"] < 1500:
        errors.append("Depth experiment retains only %d features"
                      % dp["n_points"]["xyz"])
    # paired difference of the returned depths, normalised by the depth itself.
    # The thresholds are tight because the quantity is: normalising by the error
    # against the sensor instead inflates the same runs by three decades.
    worst_pair = max(dp["pair"][c]["max"] for c in ("id", "ir", "pa"))
    if worst_pair > 1e-6:
        errors.append("Charts disagree on real depth by %.2e" % worst_pair)
    med_pair = max(dp["pair"][c]["median"] for c in ("id", "ir", "pa"))
    if med_pair > 1e-9:
        errors.append("Median paired depth difference is %.2e" % med_pair)
    # the residual disagreement must grow with the conditioning, as round-off
    # in an ill-conditioned normal matrix does
    bin_pair = [max((dp["curves_pred"][c]["pairmax"][k] or 0.0)
                    for c in ("xyz", "id", "ir", "pa"))
                for k in range(len(dp["bins_pred_decades"]) - 1)]
    present = [v for v, n in zip(bin_pair, dp["curves_mc"]["xyz"]["n"]) if n]
    if len(present) >= 3 and not all(b >= a for a, b in zip(present, present[1:])):
        errors.append("Paired depth difference is not monotone in the bound: %r"
                      % present)
    # the Monte-Carlo ratio must be one for every chart, and the same for all
    rr = [dp["mc_ratio"][c]["median"] for c in ("xyz", "id", "ir", "pa")]
    if max(rr) - min(rr) > 1e-6:
        errors.append("Monte-Carlo sensitivity ratio differs across charts: %r" % rr)
    if not (0.4 < rr[0] < 1.5):
        errors.append("Measured depth sensitivity is %.2f times the Fisher bound"
                      % rr[0])
    # and the sensitivity must follow the bound rather than sit still
    sens = [v for v in dp["curves_mc"]["xyz"]["median"] if v]
    if len(sens) < 4 or sens[-1] / sens[0] < 10.0:
        errors.append("Depth sensitivity does not follow the geometric bound "
                      "across the bins: %r" % sens)
    if dp["cross_chart_spread_max"] > 1e-6:
        errors.append("Binned depth sensitivities differ across charts by %.2e"
                      % dp["cross_chart_spread_max"])
    # solver ablation: the residual disagreement must follow the stopping rule,
    # so a tighter solver on the same windows has to lower it by an order of
    # magnitude, and a looser one to raise it.  All three use the same blocks.
    ab = {s: max(depth["solvers"][s]["pair"][c]["max"] for c in ("id", "ir", "pa"))
          for s in ("lm_phys", "lm_chart", "gn")}
    if len({depth["solvers"][s]["n_points"]["xyz"]
            for s in ("lm_phys", "lm_chart", "gn")}) != 1:
        errors.append("The solver ablation does not run on identical point blocks")
    if ab["lm_phys"] > 1e-8 or ab["lm_chart"] > 1e-8:
        errors.append("Full LM does not lower the paired difference: %r" % ab)
    if ab["gn"] < 10 * ab["lm_phys"]:
        errors.append("Undamped GN does not show a coarser stopping rule: %r" % ab)
    depth_summary = ("pair %.1e (median %.1e), sensitivity/bound %.2f, %d windows, "
                     "%d features" % (worst_pair, med_pair, rr[0],
                                      dp["n_windows_ok"], dp["n_points"]["xyz"]))
else:
    depth_summary = "skipped (results/real_depth_collapse.json absent)"


if errors:
    print("MANUSCRIPT CHECK FAILED")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

print("Manuscript static checks passed")
print("Citations:", len(cite_keys), "BibTeX entries:", len(bib_keys))
print("Labels:", len(labels), "references:", len(refs))
print("Synthetic GN invariance: %.1e m" % s1["max_abs_position_difference_m"])
print("Depth CRB chart spread: %.1e over %d angles" % (worst_spread, len(sweep)))
print("Condition-number span: %.1e" % (kappa_max / kappa_min))
print("Pullback damping identity: %.1e" % max(max(r["pullback_metric"].values()) for r in damp))
print("Normalized-chart algebraic checks: within tolerance")
print("BA-theorem identities: within tolerance; rank bookkeeping traced")
print("Real-sequence invariance:", real_summary)
print("Real-sequence depth:", depth_summary)
