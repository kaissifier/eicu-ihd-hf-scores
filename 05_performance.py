import json

import numpy as np
import pandas as pd
from scipy import stats

from common import SCORES, SEED, out

rng = np.random.default_rng(SEED)
d = pd.read_parquet(out("scores.parquet"))
known = d[d.hospitaldischargestatus.notna()]
a = known[known.p_apache.notna()].copy()
a["p_apache"] = a.p_apache.clip(1e-4, 1 - 1e-4)
y = a.died_hospital.values
names = list(SCORES)


def midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out_ranks = np.empty(n)
    out_ranks[order] = ranks
    return out_ranks


def delong(y, preds):
    """AUCs and their covariance matrix (DeLong et al., 1988) for predictions in the rows of preds."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    auc, v10, v01 = np.zeros(k), np.zeros((k, m)), np.zeros((k, n))
    for r in range(k):
        tx, ty = midrank(pos[r]), midrank(neg[r])
        tz = midrank(np.concatenate([pos[r], neg[r]]))
        auc[r] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10[r] = (tz[:m] - tx) / n
        v01[r] = 1 - (tz[m:] - ty) / m
    return auc, np.cov(v10) / m + np.cov(v01) / n


auc, cov = delong(y, np.vstack([a[c].values for c in SCORES.values()]))
res = {"population": len(d), "known_outcome": len(known), "no_discharge_status": int(d.hospitaldischargestatus.isna().sum()),
       "no_apache": int(known.p_apache.isna().sum()), "n": len(a), "deaths": int(y.sum()),
       "hospitals": int(a.hospitalid.nunique()), "scores": {}}

for i, name in enumerate(names):
    p = a[SCORES[name]]
    se = cov[i, i] ** 0.5
    by_hospital = a.groupby("hospitalid").agg(o=("died_hospital", "sum"), e=(SCORES[name], "sum"))
    draw = rng.integers(0, len(by_hospital), size=(1000, len(by_hospital)))
    boot = by_hospital.o.values[draw].sum(1) / by_hospital.e.values[draw].sum(1)
    res["scores"][name] = {
        "auc": auc[i], "auc_ci": [auc[i] - 1.96 * se, auc[i] + 1.96 * se],
        "predicted_pct": 100 * p.mean(), "observed_pct": 100 * y.mean(),
        "smr": y.sum() / p.sum(), "smr_ci": list(np.percentile(boot, [2.5, 97.5])),
        "brier": ((p - y) ** 2).mean(),
    }

res["auc_comparisons"] = {}
for i in range(3):
    for j in range(i + 1, 3):
        diff = auc[i] - auc[j]
        se = (cov[i, i] + cov[j, j] - 2 * cov[i, j]) ** 0.5
        res["auc_comparisons"][f"{names[i]} vs {names[j]}"] = {
            "difference": diff, "ci": [diff - 1.96 * se, diff + 1.96 * se], "p": 2 * stats.norm.sf(abs(diff / se))}

res["roc"] = {}
for name in names:
    s = a[SCORES[name]].values
    thresholds = np.unique(np.quantile(s, np.linspace(0, 1, 201)))[::-1]
    res["roc"][name] = {"fpr": [((s >= t) & (y == 0)).sum() / (y == 0).sum() for t in thresholds],
                        "tpr": [((s >= t) & (y == 1)).sum() / (y == 1).sum() for t in thresholds]}

# patients with urine output recorded and, if ventilated, a PaO2/FiO2 ratio
complete = a[a.uo.notna() & ~((a.vent_support == 1) & a.pf_min.isna())]
cc_auc, _ = delong(complete.died_hospital.values, np.vstack([complete[c].values for c in SCORES.values()]))
res["complete_case"] = {"n": len(complete), "deaths": int(complete.died_hospital.sum()),
                        "auc": dict(zip(names, cc_auc))}

a[["died_hospital", *SCORES.values()]].rename(columns={"died_hospital": "death"}).to_csv(out("predictions.csv"), index=False)
with open(out("performance.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)

print(f"{res['n']} patients, {res['deaths']} deaths, {res['hospitals']} hospitals")
for name, r in res["scores"].items():
    print(f"{name:<11} AUC {r['auc']:.3f} ({r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f})  "
          f"SMR {r['smr']:.2f} ({r['smr_ci'][0]:.2f}-{r['smr_ci'][1]:.2f})  Brier {r['brier']:.3f}")
for k, r in res["auc_comparisons"].items():
    p = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
    print(f"{k:<22} difference {r['difference']:+.3f} ({r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}), {p}")
