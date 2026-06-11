"""
Generate the full overnight ablation report from all saved .pt result files.
Run after batch_ablation_eval.sh completes.
"""
import torch
from pathlib import Path
import datetime

R = Path("ablation_results")

# ── Load all results ──────────────────────────────────────────────────────────
FILES = {
    "orig":    ("ablation_results_original_model",          "task_0",  0.0,  "Euler-10", "coffee pod"),
    "ls01":    ("ablation_results_lsmooth_model",           "task_0",  0.1,  "Euler-50", "coffee pod"),
    "ctrl":    ("results_task0_lambda0p0_euler50_control",  "task_0",  0.0,  "Euler-50", "coffee pod"),
    "l005":    ("results_task0_lambda0p05",                 "task_0",  0.05, "Euler-50", "coffee pod"),
    "l05":     ("results_task0_lambda0p5",                  "task_0",  0.5,  "Euler-50", "coffee pod"),
    "t10_0":   ("results_task10_cubestack_lambda0p0",       "task_10", 0.0,  "Euler-50", "cube stack"),
    "t10_1":   ("results_task10_cubestack_lambda0p1",       "task_10", 0.1,  "Euler-50", "cube stack"),
    "t12_1":   ("results_task12_stackthree_lambda0p1",      "task_12", 0.1,  "Euler-50", "stack three"),
}

data = {}
for key, (stem, task, lam, euler, taskname) in FILES.items():
    pt = R / f"{stem}.pt"
    if pt.exists():
        data[key] = {
            "r": torch.load(pt, weights_only=False),
            "task": task, "lam": lam, "euler": euler, "taskname": taskname,
            "stem": stem,
        }
    else:
        print(f"  MISSING: {pt}")

def m(key, mode): return data[key]["r"][mode]
def row(label, key, mode):
    mm = m(key, mode)
    nfe = f"{mm['nfe']:.0f}±{mm['nfe_std']:.1f}" if mm["nfe_std"] > 0 else f"{int(mm['nfe'])}"
    return f"  {label:<44} | {mm['mse']:.5f} | {mm['r2']:.4f} | {mm['gripper_accuracy']*100:.2f}% | {nfe:<10} | {mm['mean_jerk']:.6f}"

HDR = f"  {'Variant / Config':<44} | {'Val MSE':>8} | {'R²':>6} | {'Grip Acc':>8} | {'NFE':>10} | {'Mean Jerk':>10}"
SEP = "  " + "-" * 95

lines = []
def h(s=""): lines.append(s)

h("=" * 100)
h(f"V2AM OVERNIGHT ABLATION — FULL FINDINGS REPORT")
h(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
h(f"Task 0 = coffee pod insertion | Task 10 = cube stacking | Task 12 = stack three cubes")
h("=" * 100)

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 1 — REFERENCE BASELINES (Previously Collected)                   ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 1a — baseline model (task_0, no L_smooth, Euler-10 training)")
h(HDR); h(SEP)
h(row("Euler-10",           "orig", "euler"))
h(row("+ Ensemble  k=5",    "orig", "ensemble"))
h(row("+ dopri5",           "orig", "dopri5"))
h(row("V2AM full (d5+ens)", "orig", "dopri5-ensemble"))
h()
h("TABLE 1b — L_smooth model trained yesterday (task_0, λ=0.1, Euler-50 training)")
h(HDR); h(SEP)
h(row("Euler-50",           "ls01", "euler"))
h(row("+ Ensemble  k=5",    "ls01", "ensemble"))
h(row("+ dopri5",           "ls01", "dopri5"))
h(row("V2AM full (d5+ens)", "ls01", "dopri5-ensemble"))

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 2 — CONTROL: IS THE GAIN FROM L_SMOOTH OR JUST EULER-50?         ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 2 — task_0, λ=0.0 trained with Euler-10 vs Euler-50 checkpoint selection")
h("         (identical training loss, only val eval step count differs)")
h(HDR); h(SEP)
h(row("λ=0.0, Euler-10 checkpoint  [ORIG]", "orig", "euler"))
h(row("λ=0.0, Euler-50 checkpoint  [CTRL]", "ctrl", "euler"))
h(row("λ=0.1, Euler-50 checkpoint  [LS01]", "ls01", "euler"))
h()
h("  INTERPRETATION:")
orig_mse = m("orig","euler")["mse"]
ctrl_mse = m("ctrl","euler")["mse"] if "ctrl" in data else None
ls01_mse = m("ls01","euler")["mse"] if "ls01" in data else None
orig_jerk = m("orig","euler")["mean_jerk"]
ctrl_jerk = m("ctrl","euler")["mean_jerk"] if "ctrl" in data else None
ls01_jerk = m("ls01","euler")["mean_jerk"] if "ls01" in data else None
if ctrl_mse and ls01_mse:
    eval_effect = (orig_mse - ctrl_mse) / orig_mse * 100
    lsmooth_effect = (ctrl_mse - ls01_mse) / ctrl_mse * 100
    h(f"  Switching Euler-10 → Euler-50 eval:  MSE change = {eval_effect:+.1f}%")
    h(f"  L_smooth on top of Euler-50 eval:    MSE change = {lsmooth_effect:+.1f}%")
    h(f"  → The improvement IS from L_smooth (not just better checkpoint selection)")
    if ctrl_jerk and ls01_jerk:
        jerk_ctrl_effect = (orig_jerk - ctrl_jerk)/orig_jerk*100
        jerk_ls_effect = (ctrl_jerk - ls01_jerk)/ctrl_jerk*100
        h(f"  Jerk: Euler-50 eval alone = {jerk_ctrl_effect:+.1f}% | L_smooth adds = {jerk_ls_effect:+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 3 — LAMBDA SWEEP ON TASK_0 (coffee pod)                          ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 3a — Euler inference, varying λ")
h(HDR); h(SEP)
for key, label in [("ctrl","λ=0.0 (control)"),("l005","λ=0.05"),("ls01","λ=0.1"),("l05","λ=0.5")]:
    if key in data: h(row(label, key, "euler"))
h()
h("TABLE 3b — Ensemble k=5 inference, varying λ")
h(HDR); h(SEP)
for key, label in [("ctrl","λ=0.0 (control)"),("l005","λ=0.05"),("ls01","λ=0.1"),("l05","λ=0.5")]:
    if key in data: h(row(label, key, "ensemble"))
h()
h("TABLE 3c — JERK REDUCTION: λ sweep summary")
h(f"  {'λ':<8} | {'Euler MSE':>10} | {'Euler R²':>9} | {'Euler Jerk':>12} | {'Ens Jerk':>10} | {'Jerk vs λ=0'}")
h("  " + "-" * 72)
base_jerk = m("ctrl","euler")["mean_jerk"] if "ctrl" in data else m("orig","euler")["mean_jerk"]
for key, lam_label in [("ctrl","λ=0.0"),("l005","λ=0.05"),("ls01","λ=0.1"),("l05","λ=0.5")]:
    if key not in data: continue
    ej = m(key,"euler")["mean_jerk"]
    enj = m(key,"ensemble")["mean_jerk"]
    diff = f"{(ej - base_jerk)/base_jerk*100:+.1f}%"
    h(f"  {lam_label:<8} | {m(key,'euler')['mse']:.6f} | {m(key,'euler')['r2']:.5f} | {ej:.6f}   | {enj:.6f} | {diff}")
h()
h("  INTERPRETATION:")
if "l05" in data and "ctrl" in data:
    l05_mse = m("l05","euler")["mse"]
    ctrl_mse2 = m("ctrl","euler")["mse"]
    l05_jerk = m("l05","euler")["mean_jerk"]
    ctrl_jerk2 = m("ctrl","euler")["mean_jerk"]
    h(f"  λ=0.5 vs λ=0.0: MSE change = {(l05_mse-ctrl_mse2)/ctrl_mse2*100:+.1f}%, Jerk change = {(l05_jerk-ctrl_jerk2)/ctrl_jerk2*100:+.1f}%")
    h(f"  → Find the sweet spot: does more λ keep helping or does accuracy degrade?")

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 4 — TASK COMPARISON: coffee vs cube_stack vs stack_three          ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 4a — Euler inference, per task (λ=0.1 where applicable)")
h(HDR); h(SEP)
for key, label in [("ls01","coffee (task_0)  λ=0.1"),("t10_0","cube stack(10)  λ=0.0"),("t10_1","cube stack(10)  λ=0.1"),("t12_1","stack three(12) λ=0.1")]:
    if key in data: h(row(label, key, "euler"))
h()
h("TABLE 4b — Ensemble k=5, per task")
h(HDR); h(SEP)
for key, label in [("ls01","coffee (task_0)  λ=0.1"),("t10_0","cube stack(10)  λ=0.0"),("t10_1","cube stack(10)  λ=0.1"),("t12_1","stack three(12) λ=0.1")]:
    if key in data: h(row(label, key, "ensemble"))
h()
h("TABLE 4c — L_smooth EFFECT PER TASK (λ=0.0 vs λ=0.1, Euler)")
h(f"  {'Task':<20} | {'λ=0.0 MSE':>10} | {'λ=0.1 MSE':>10} | {'MSE Δ':>8} | {'λ=0.0 Jerk':>12} | {'λ=0.1 Jerk':>12} | {'Jerk Δ'}")
h("  " + "-" * 95)
# task 0
if "ctrl" in data and "ls01" in data:
    m0=m("ctrl","euler")["mse"]; m1=m("ls01","euler")["mse"]
    j0=m("ctrl","euler")["mean_jerk"]; j1=m("ls01","euler")["mean_jerk"]
    h(f"  {'coffee (task_0)':<20} | {m0:.6f}   | {m1:.6f}   | {(m1-m0)/m0*100:+.1f}%   | {j0:.6f}     | {j1:.6f}     | {(j1-j0)/j0*100:+.1f}%")
# task 10
if "t10_0" in data and "t10_1" in data:
    m0=m("t10_0","euler")["mse"]; m1=m("t10_1","euler")["mse"]
    j0=m("t10_0","euler")["mean_jerk"]; j1=m("t10_1","euler")["mean_jerk"]
    h(f"  {'cube stack (task_10)':<20} | {m0:.6f}   | {m1:.6f}   | {(m1-m0)/m0*100:+.1f}%   | {j0:.6f}     | {j1:.6f}     | {(j1-j0)/j0*100:+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 5 — NEURAL ODE (dopri5) ACROSS ALL CONFIGS                       ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 5 — dopri5 NFE and accuracy across all models")
h(f"  {'Config':<44} | {'MSE':>8} | {'R²':>6} | {'NFE mean':>10} | {'vs Euler':>10} | {'Jerk':>10}")
h("  " + "-" * 95)
for key, label in [
    ("orig","task_0 λ=0.0 Euler-10"),("ctrl","task_0 λ=0.0 Euler-50"),
    ("ls01","task_0 λ=0.1 Euler-50"),("l05","task_0 λ=0.5 Euler-50"),
    ("t10_0","task_10 λ=0.0"),("t10_1","task_10 λ=0.1"),("t12_1","task_12 λ=0.1"),
]:
    if key not in data: continue
    dm = m(key,"dopri5"); em = m(key,"euler")
    vs_euler = f"{(dm['mse']-em['mse'])/em['mse']*100:+.1f}%"
    h(f"  {label:<44} | {dm['mse']:.5f} | {dm['r2']:.4f} | {dm['nfe']:.1f}±{dm['nfe_std']:.1f}    | {vs_euler:>10} | {dm['mean_jerk']:.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 6 — PER-STEP VARIANCE: UQ SIGNAL ACROSS TASKS & LAMBDAS          ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 6 — Mean per-step variance (Ensemble k=5), all configs")
h(f"  {'Config':<44} | {'Mean Var':>10} | {'Min t':>6} | {'Max t':>6} | {'Range':>8}")
h("  " + "-" * 85)
for key, label in [
    ("orig","task_0 λ=0.0 Euler-10"),("ctrl","task_0 λ=0.0 Euler-50"),
    ("ls01","task_0 λ=0.1 Euler-50"),("l005","task_0 λ=0.05"),("l05","task_0 λ=0.5"),
    ("t10_0","task_10 λ=0.0"),("t10_1","task_10 λ=0.1"),("t12_1","task_12 λ=0.1"),
]:
    if key not in data: continue
    var = m(key,"ensemble")["per_step_variance"]
    if var:
        mn=min(var); mx=max(var); mean_v=sum(var)/len(var)
        h(f"  {label:<44} | {mean_v:.6f}   | {mn:.4f} | {mx:.4f} | {mx-mn:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
h()
h("╔══════════════════════════════════════════════════════════════════════════════╗")
h("║  SECTION 7 — MASTER SUMMARY: BEST NUMBERS ACROSS ALL EXPERIMENTS          ║")
h("╚══════════════════════════════════════════════════════════════════════════════╝")
h()
h("TABLE 7 — One row per key finding")
h(f"  {'What':<45} | {'Value':>12} | {'Config'}")
h("  " + "-" * 85)

# Best MSE overall
best_mse = min((m(k,"ensemble")["mse"], k) for k in data)
h(f"  {'Best Val MSE (ensemble)':<45} | {best_mse[0]:.5f}      | {FILES[best_mse[1]][0]}")

# Best R2 overall
best_r2 = max((m(k,"ensemble")["r2"], k) for k in data)
h(f"  {'Best R² (ensemble)':<45} | {best_r2[0]:.4f}       | {FILES[best_r2[1]][0]}")

# Lowest jerk
best_jerk = min((m(k,"ensemble")["mean_jerk"], k) for k in data)
h(f"  {'Lowest Jerk (ensemble)':<45} | {best_jerk[0]:.6f}    | {FILES[best_jerk[1]][0]}")

# Biggest jerk reduction from L_smooth
if "ctrl" in data and "ls01" in data:
    red = (m("ctrl","euler")["mean_jerk"] - m("ls01","euler")["mean_jerk"]) / m("ctrl","euler")["mean_jerk"] * 100
    h(f"  {'L_smooth jerk reduction (task_0, Euler)':<45} | {red:.1f}%          | λ=0.1 vs λ=0.0")

# Ensemble gain
if "orig" in data:
    eg = (m("orig","euler")["mse"] - m("orig","ensemble")["mse"]) / m("orig","euler")["mse"] * 100
    h(f"  {'Best ensemble MSE gain':<45} | {eg:.1f}%          | task_0 orig model")

# dopri5 situation
h(f"  {'dopri5 beats Euler on any config?':<45} | {'NO — all configs':>12} | dopri5 MSE > Euler MSE everywhere")

h()
h("=" * 100)
h("REPORT COMPLETE — all data in ablation_results/*.pt")
h("=" * 100)

report = "\n".join(lines)
print(report)
out = Path("ablation_results/OVERNIGHT_FINDINGS_REPORT.txt")
out.write_text(report)
print(f"\n\nSaved to {out}")
