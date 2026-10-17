import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG_DIR, SCORES, out

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"APACHE IVa": "#2a78d6", "SAPS II": "#e34948", "OASIS": "#1f9e89"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.8})
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load(name):
    with open(out(name)) as f:
        return json.load(f)


def style(ax):
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=300, facecolor="white")
    plt.close(fig)


flow, perf, belts = load("flow.json"), load("performance.json"), load("calibration_belt.json")
pop = pd.read_parquet(out("population.parquet"))
excluded = flow["excluded_age"] + flow["excluded_readmission"] + flow["excluded_icu_under_4h"]
n_admission = int((pop.criterion == "admission").sum())

boxes = {
    "b1": f"eICU Collaborative Research Database\n{flow['icu_stays']:,} ICU stays, {flow['hospitals']} hospitals",
    "b2": f"ICU stays with ischemic heart disease\n{flow['ihd_stays']:,}",
    "b3": f"Adults with ischemic heart disease\n{flow['cohort']:,} patients",
    "b4": f"Acute or decompensated HF at admission\n{len(pop):,} patients\n"
          f"(HF admission: {n_admission}; diagnosis within 24 hours: {len(pop) - n_admission})",
    "b5": f"Performance analysis\n{perf['n']} patients, {perf['deaths']} deaths, {perf['hospitals']} hospitals",
}
sides = {
    "x1": f"Without ischemic heart disease:\n{flow['icu_stays'] - flow['ihd_stays']:,}",
    "x2": f"Excluded: {excluded:,}\nAge under 18 years: {flow['excluded_age']}\n"
          f"Readmission in the same hospitalization: {flow['excluded_readmission']:,}\n"
          f"ICU stay under 4 hours: {flow['excluded_icu_under_4h']}",
    "x3": f"Without acute or decompensated HF at admission:\n{flow['cohort'] - len(pop):,}",
    "x4": f"No hospital discharge status: {perf['no_discharge_status']}\nNo APACHE IVa prediction: {perf['no_apache']}",
}
fig, ax = plt.subplots(figsize=(7.0, 7.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
box = dict(boxstyle="square,pad=0.6", facecolor="white", edgecolor=INK2, linewidth=0.9)
arrow = dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0)
ys = {"b1": 95, "b2": 77, "b3": 56, "b4": 33, "b5": 8}
for k, yv in ys.items():
    ax.text(30, yv, boxes[k], ha="center", va="center", fontsize=8.4, bbox=box, linespacing=1.35)
for k, yv in (("x1", 86), ("x2", 66.5), ("x3", 44.5), ("x4", 20.5)):
    ax.text(74, yv, sides[k], ha="center", va="center", fontsize=7.8, bbox=box, linespacing=1.35)
    ax.annotate("", xy=(52, yv), xytext=(30, yv), arrowprops=arrow)
for upper, lower in (("b1", "b2"), ("b2", "b3"), ("b3", "b4"), ("b4", "b5")):
    ax.annotate("", xy=(30, ys[lower] + 5.5), xytext=(30, ys[upper] - 5.5), arrowprops=arrow)
save(fig, "figure1_flowchart.png")

fig, ax = plt.subplots(figsize=(5.0, 4.8))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)))
for name in SCORES:
    r = perf["roc"][name]
    ax.plot([0] + r["fpr"] + [1], [0] + r["tpr"] + [1], color=COLORS[name], lw=1.8,
            label=f"{name}: AUC {perf['scores'][name]['auc']:.2f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
style(ax)
ax.set_xlabel("1 - specificity")
ax.set_ylabel("Sensitivity")
ax.legend(loc="lower right", frameon=False, fontsize=9)
save(fig, "figure2_roc.png")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, (name, col), letter in zip(axes, SCORES.items(), "ABC"):
    b = belts[col]
    x = 100 * np.array(b["seqP"])
    for level, alpha, label in ((1, 0.22, "95% belt"), (0, 0.45, "80% belt")):
        ax.fill_between(x, 100 * np.array(b["bounds"][level]["L"]), 100 * np.array(b["bounds"][level]["U"]),
                        color=COLORS[name], alpha=alpha, linewidth=0, label=label)
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"{letter}. {name}", fontsize=9.5, color=INK)
    style(ax)
    p = "p < 0.001" if b["p_value"] < 0.001 else f"p = {b['p_value']:.3f}"
    at95 = b["inters"][1]
    deviations = [f"below {100 * lo:.0f}-{100 * hi:.0f}%" for lo, hi in at95["under"]]
    deviations += [f"above {100 * lo:.0f}-{100 * hi:.0f}%" for lo, hi in at95["over"]]
    text = "\n".join([f"Calibration test: {p}", "Deviation from bisector (95%):", *(deviations or ["none"])])
    ax.text(3, 97, text, ha="left", va="top", fontsize=6.6, color=INK2, linespacing=1.3,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.set_xlabel("Predicted mortality (%)", fontsize=8.5)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Observed mortality (%)", fontsize=8.5)
axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)
save(fig, "figure3_calibration_belts.png")
print(f"figures saved in {FIG_DIR}")
