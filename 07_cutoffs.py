import json
import math

import numpy as np
import pandas as pd

from common import out, table

d = pd.read_parquet(out("scores.parquet"))
a = d[d.hospitaldischargestatus.notna() & d.p_apache.notna()].copy()
apache = pd.read_csv(table("apachePatientResult"), usecols=["patientunitstayid", "apacheversion", "apachescore"])
apache = apache[apache.apacheversion == "IVa"].drop_duplicates("patientunitstayid")
a = a.merge(apache[["patientunitstayid", "apachescore"]], on="patientunitstayid", how="left")
y = a.died_hospital.values.astype(int)


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def proportion(k, n):
    return {"value": k / n if n else float("nan"), "ci": wilson(k, n)}


def accuracy(positive):
    tp = int((positive & (y == 1)).sum())
    fn = int((~positive & (y == 1)).sum())
    tn = int((~positive & (y == 0)).sum())
    fp = int((positive & (y == 0)).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": proportion(tp, tp + fn),
            "specificity": proportion(tn, tn + fp), "ppv": proportion(tp, tp + fp), "npv": proportion(tn, tn + fn)}


COLUMNS = {"APACHE IVa": ("p_apache", "apachescore"), "SAPS II": ("sapsii_prob", "sapsii"), "OASIS": ("oasis_prob", "oasis")}
res = {"n": len(a), "deaths": int(y.sum())}
for name, (prob, points) in COLUMNS.items():
    p = a[prob].values
    best_j, best_t = None, None
    for t in np.unique(p):
        positive = p >= t
        j = (positive & (y == 1)).sum() / y.sum() + (~positive & (y == 0)).sum() / (y == 0).sum() - 1
        if best_j is None or j > best_j:
            best_j, best_t = j, t
    positive = p >= best_t
    min_points = a.loc[positive, points].min()
    res[name] = {"cutoff_prob": float(best_t), "cutoff_points": None if pd.isna(min_points) else float(min_points),
                 "youden": float(best_j), "positives": int(positive.sum()), **accuracy(positive),
                 "cutoff_50": accuracy(p >= 0.5)}

with open(out("cutoffs.json"), "w") as f:
    json.dump(res, f, indent=1)


def fmt(r):
    return f"{100 * r['value']:.1f}% ({100 * r['ci'][0]:.1f}-{100 * r['ci'][1]:.1f})"


for name in COLUMNS:
    r = res[name]
    print(f"{name}: >= {r['cutoff_points']:.0f} points ({100 * r['cutoff_prob']:.1f}% predicted) | "
          f"sensitivity {fmt(r['sensitivity'])}, specificity {fmt(r['specificity'])}")
