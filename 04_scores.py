import numpy as np
import pandas as pd

from common import EXTRACT_DIR, out


def read(name):
    return pd.read_parquet(EXTRACT_DIR / f"{name}.parquet")


def num(s):
    return pd.to_numeric(s, errors="coerce")


pop = pd.read_parquet(out("population.parquet"))
d = pd.DataFrame({"id": pop.patientunitstayid})

# vital signs and GCS: minimum and maximum
vp, va, nc = read("vitalPeriodic"), read("vitalAperiodic"), read("nurseCharting")
long = []
for col, var in (("heartrate", "hr"), ("respiration", "rr"), ("systemicsystolic", "sbp"), ("systemicmean", "map"),
                 ("temperature", "temp")):
    long.append(pd.DataFrame({"id": vp.patientunitstayid, "var": var, "v": num(vp[col])}))
for col, var in (("noninvasivesystolic", "sbp"), ("noninvasivemean", "map")):
    long.append(pd.DataFrame({"id": va.patientunitstayid, "var": var, "v": num(va[col])}))
NURSING = {"Heart Rate": "hr", "Respiratory Rate": "rr", "Non-Invasive BP Systolic": "sbp",
           "Invasive BP Systolic": "sbp", "Non-Invasive BP Mean": "map", "Invasive BP Mean": "map",
           "Temperature (C)": "temp", "GCS Total": "gcs"}
charted = nc[nc.nursingchartcelltypevalname.isin(NURSING)]
long.append(pd.DataFrame({"id": charted.patientunitstayid, "var": charted.nursingchartcelltypevalname.map(NURSING),
                          "v": num(charted.nursingchartvalue)}))
vitals = pd.concat(long).dropna()
LIMITS = {"hr": (20, 250), "rr": (1, 70), "sbp": (40, 300), "map": (20, 200), "temp": (25, 45), "gcs": (3, 15)}
low = vitals["var"].map({k: lo for k, (lo, hi) in LIMITS.items()})
high = vitals["var"].map({k: hi for k, (lo, hi) in LIMITS.items()})
vitals = vitals[(vitals.v >= low) & (vitals.v <= high)]
agg = vitals.groupby(["id", "var"])["v"].agg(["min", "max"]).unstack()
agg.columns = [f"{var}_{stat}" for stat, var in agg.columns]
d = d.merge(agg, left_on="id", right_index=True, how="left")

# laboratory
lab = read("lab")
lab["v"] = num(lab.labresult)
LABS = {"BUN": "bun", "WBC x 1000": "wbc", "potassium": "k", "sodium": "na", "bicarbonate": "hco3", "HCO3": "hco3",
        "total bilirubin": "bili", "platelets x 1000": "plt", "creatinine": "creat"}
lb = lab[lab.labname.isin(LABS)].assign(var=lambda z: z.labname.map(LABS)).dropna(subset=["v"])
lb = lb[lb.v >= 0]
agg = lb.groupby(["patientunitstayid", "var"])["v"].agg(["min", "max"]).unstack()
agg.columns = [f"{var}_{stat}" for stat, var in agg.columns]
d = d.merge(agg, left_on="id", right_index=True, how="left")

# invasive and noninvasive ventilation
aps, pred, rc = read("apacheApsVar"), read("apachePredVar"), read("respiratoryCharting")
invasive = (set(aps.loc[(num(aps.vent) == 1) | (num(aps.intubated) == 1), "patientunitstayid"])
            | set(pred.loc[(num(pred.oobventday1) == 1) | (num(pred.ventday1) == 1), "patientunitstayid"]))
device = nc[nc.nursingchartcelltypevalname == "O2 Admin Device"]
device_name = device.nursingchartvalue.fillna("").str.strip().str.lower()
invasive |= set(device.loc[device_name.isin(["ventilator", "vent"]), "patientunitstayid"])
noninvasive = set(device.loc[device_name.str.contains("bipap|cpap|bi-pap"), "patientunitstayid"])
resp_value = rc.respchartvalue.fillna("").str.lower()
resp_label = rc.respchartvaluelabel.fillna("")
noninvasive |= set(rc.loc[resp_value.str.contains("bipap|bi-pap|cpap") | resp_label.str.contains(r"^NIV|IPAP|EPAP"),
                          "patientunitstayid"])
d["invasive_vent"] = d.id.isin(invasive).astype(int)
d["niv"] = (d.id.isin(noninvasive) & ~d.id.isin(invasive)).astype(int)
d["vent_support"] = ((d.invasive_vent == 1) | d.id.isin(noninvasive)).astype(int)

# PaO2/FiO2 from 0 to 24 h, paired with the nearest FiO2 within 3 hours
pao2 = lab[(lab.labname == "paO2") & (lab.labresultoffset >= 0)][["patientunitstayid", "labresultoffset", "v"]].dropna()
fio2 = pd.concat([
    lab[lab.labname == "FiO2"][["patientunitstayid", "labresultoffset", "v"]].rename(columns={"labresultoffset": "t"}),
    rc[rc.respchartvaluelabel.isin(["FiO2", "FIO2 (%)"])].assign(v=lambda z: num(z.respchartvalue))
    [["patientunitstayid", "respchartoffset", "v"]].rename(columns={"respchartoffset": "t"}),
]).dropna()
fio2["v"] = np.where(fio2.v <= 1, fio2.v * 100, fio2.v)
fio2 = fio2[(fio2.v >= 21) & (fio2.v <= 100)]
pao2 = pao2.rename(columns={"labresultoffset": "t"}).sort_values("t")
fio2 = fio2.sort_values("t").rename(columns={"v": "fio2"})
pf = pd.merge_asof(pao2, fio2, on="t", by="patientunitstayid", direction="nearest", tolerance=180).dropna()
pf["pf"] = pf.v / (pf.fio2 / 100)
d = d.merge(pf.groupby("patientunitstayid")["pf"].min().rename("pf_min"), left_on="id", right_index=True, how="left")

# urine output in the first 24 h
apache_uo = aps[["patientunitstayid", "urine"]].assign(u=lambda z: num(z.urine))
apache_uo = apache_uo[apache_uo.u >= 0]
io = read("intakeOutput")
io_label = io.celllabel.fillna("").str.lower()
io_path = io.cellpath.fillna("").str.lower()
urine = io[io_path.str.contains("output") & io_label.str.contains("urin|foley|catheter")
           & ~io_label.str.contains("count|stool|unmeasured")].copy()
urine["val"] = num(urine.cellvaluenumeric)
urine = urine[urine.val >= 0]
charted_uo = urine.groupby("patientunitstayid").agg(total=("val", "sum"), t0=("intakeoutputoffset", "min"),
                                                    t1=("intakeoutputoffset", "max"))
charted_uo = charted_uo[(charted_uo.t1 - charted_uo.t0) >= 20 * 60]
d["uo"] = d.id.map(apache_uo.set_index("patientunitstayid").u)
d["uo_source"] = np.where(d.uo.notna(), "apache", np.where(d.id.isin(charted_uo.index), "intakeOutput", "missing"))
d["uo"] = d.uo.fillna(d.id.map(charted_uo.total))

# GCS from nursing; APACHE components when not charted
gcs = aps.assign(e=num(aps.eyes), m=num(aps.motor), v=num(aps.verbal))
gcs = gcs[(gcs.e > 0) & (gcs.m > 0) & (gcs.v > 0)].assign(g=lambda z: z.e + z.m + z.v).set_index("patientunitstayid").g
d["gcs_source"] = np.where(d.gcs_min.notna(), "nursing", np.where(d.id.isin(gcs.index), "apache", "missing"))
d["gcs_min"] = d.gcs_min.fillna(d.id.map(gcs))

# vasopressors and inotropes (SOFA only); mcg/min doses are divided by weight
inf = read("infusionDrug")
drug_name = inf.drugname.fillna("").str.lower()
rate = num(inf.drugrate)
pt = read("patient").set_index("patientunitstayid")
weight = num(inf.patientweight)
weight = weight.where(weight.between(30, 300), inf.patientunitstayid.map(num(pt.admissionweight)))
drug = np.select([drug_name.str.contains("norepinephrine"), drug_name.str.contains("epinephrine"),
                  drug_name.str.contains("dopamine"), drug_name.str.contains("dobutamine")],
                 ["nor", "epi", "dopa", "dobu"], None)
dose = np.where(drug_name.str.contains("mcg/kg/min"), rate,
                np.where(drug_name.str.contains(r"\(mcg/min\)"), rate / weight, np.nan))
vaso = pd.DataFrame({"id": inf.patientunitstayid, "drug": drug, "rate": rate, "dose": dose})
vaso = vaso[vaso.drug.notna() & (vaso.rate > 0)]
vaso.loc[vaso.dose > 50, "dose"] = np.nan
vaso = vaso.groupby(["id", "drug"]).agg(dose=("dose", "max"), n=("rate", "size")).reset_index()


def sofa_cardio(stay, map_min):
    r = vaso[vaso.id == stay]
    doses = {k: r.loc[r.drug == k, "dose"] for k in ("nor", "epi", "dopa", "dobu")}
    given = {k: len(v) > 0 for k, v in doses.items()}
    top = {k: (v.max() if len(v) and v.notna().any() else np.nan) for k, v in doses.items()}
    if top["dopa"] > 15 or top["epi"] > 0.1 or top["nor"] > 0.1:
        return 4
    if top["dopa"] > 5 or given["epi"] or given["nor"]:
        return 3
    if given["dopa"] or given["dobu"]:
        return 2
    if map_min < 70:
        return 1
    return 0


# demographics, admission type and chronic diseases
d["age"] = d.id.map(num(pt.age.replace({"> 89": "90"})))
d["pre_icu_min"] = d.id.map(-num(pt.hospitaladmitoffset))
surgical = pt.unitadmitsource.fillna("").isin(["Operating Room", "Recovery Room", "PACU"]).astype(int)
d["surgical"] = d.id.map(surgical).fillna(0)
pr = pred.set_index("patientunitstayid")
d["elective"] = d.id.map(num(pr.electivesurgery) == 1).fillna(False).astype(int)
d["aids"] = d.id.map(num(pr.aids) == 1).fillna(False).astype(int)
d["hematologic"] = d.id.map((num(pr.lymphoma) == 1) | (num(pr.leukemia) == 1)).fillna(False).astype(int)
d["metastatic"] = d.id.map(num(pr.metastaticcancer) == 1).fillna(False).astype(int)


def first(conditions, default=0):
    """Points of the first condition met, in order; default when none is met or data are missing."""
    points = np.full(len(d), np.nan)
    for cond, p in conditions:
        points = np.where(np.isnan(points) & cond.fillna(False).values, p, points)
    return np.nan_to_num(points, nan=default)


vent = d.vent_support == 1

saps = {
    "age": first([(d.age < 40, 0), (d.age < 60, 7), (d.age < 70, 12), (d.age < 75, 15), (d.age < 80, 16),
                  (d.age >= 80, 18)]),
    "hr": first([(d.hr_min < 40, 11), (d.hr_max >= 160, 7), (d.hr_max >= 120, 4), (d.hr_min < 70, 2)]),
    "sbp": first([(d.sbp_min < 70, 13), (d.sbp_min < 100, 5), (d.sbp_max >= 200, 2)]),
    "temp": first([(d.temp_max >= 39, 3)]),
    "pf": first([(vent & (d.pf_min < 100), 11), (vent & (d.pf_min < 200), 9), (vent & (d.pf_min >= 200), 6)]),
    "uo": first([(d.uo < 500, 11), (d.uo < 1000, 4)]),
    "bun": first([(d.bun_max >= 84, 10), (d.bun_max >= 28, 6)]),
    "wbc": first([(d.wbc_min < 1, 12), (d.wbc_max >= 20, 3)]),
    "k": first([(d.k_min < 3, 3), (d.k_max >= 5, 3)]),
    "na": first([(d.na_min < 125, 5), (d.na_max >= 145, 1)]),
    "hco3": first([(d.hco3_min < 15, 6), (d.hco3_min < 20, 3)]),
    "bili": first([(d.bili_max >= 6, 9), (d.bili_max >= 4, 4)]),
    "gcs": first([(d.gcs_min < 3, 0), (d.gcs_min < 6, 26), (d.gcs_min < 9, 13), (d.gcs_min < 11, 7),
                  (d.gcs_min < 14, 5)]),
    "chronic": first([(d.aids == 1, 17), (d.hematologic == 1, 10), (d.metastatic == 1, 9)]),
    "admission": first([((d.surgical == 1) & (d.elective == 1), 0), (d.surgical == 1, 8)], default=6),
}
d["sapsii"] = sum(saps.values())
d["sapsii_prob"] = 1 / (1 + np.exp(-(-7.7631 + 0.0737 * d.sapsii + 0.9971 * np.log(d.sapsii + 1))))

oasis = {
    "pre_icu": first([(d.pre_icu_min < 10.2, 5), (d.pre_icu_min < 297, 3), (d.pre_icu_min < 1440, 0),
                      (d.pre_icu_min < 18708, 2), (d.pre_icu_min >= 18708, 1)]),
    "age": first([(d.age < 24, 0), (d.age <= 53, 3), (d.age <= 77, 6), (d.age <= 89, 9), (d.age >= 90, 7)]),
    "gcs": first([(d.gcs_min <= 7, 10), (d.gcs_min < 14, 4), (d.gcs_min == 14, 3)]),
    "hr": first([(d.hr_max > 125, 6), (d.hr_min < 33, 4), (d.hr_max.between(107, 125), 3),
                 (d.hr_max.between(89, 106), 1)]),
    "map": first([(d.map_min < 20.65, 4), (d.map_min < 51, 3), (d.map_max > 143.44, 3),
                  ((d.map_min >= 51) & (d.map_min < 61.33), 2)]),
    "rr": first([(d.rr_min < 6, 10), (d.rr_max > 44, 9), (d.rr_max > 30, 6), (d.rr_max > 22, 1), (d.rr_min < 13, 1)]),
    "temp": first([(d.temp_max > 39.88, 6), (d.temp_min.between(33.22, 35.93), 4),
                   (d.temp_max.between(33.22, 35.93), 4), (d.temp_min < 33.22, 3),
                   ((d.temp_min > 35.93) & (d.temp_min <= 36.39), 2), (d.temp_max.between(36.89, 39.88), 2)]),
    "uo": first([(d.uo < 671.09, 10), (d.uo > 6896.80, 8), (d.uo.between(671.09, 1426.99), 5),
                 (d.uo.between(1427.00, 2544.14), 1)]),
    "vent": np.where(d.invasive_vent == 1, 9, 0),
    "elective": np.where((d.elective == 1) & (d.surgical == 1), 0, 6),
}
d["oasis"] = sum(oasis.values())
d["oasis_prob"] = 1 / (1 + np.exp(-(-6.1746 + 0.1275 * d.oasis)))

# SOFA: ventilatory support means invasive or noninvasive ventilation
sofa = {
    "resp": first([(vent & (d.pf_min < 100), 4), (vent & (d.pf_min < 200), 3), (d.pf_min < 300, 2),
                   (d.pf_min < 400, 1)]),
    "coag": first([(d.plt_min < 20, 4), (d.plt_min < 50, 3), (d.plt_min < 100, 2), (d.plt_min < 150, 1)]),
    "liver": first([(d.bili_max >= 12, 4), (d.bili_max >= 6, 3), (d.bili_max >= 2, 2), (d.bili_max >= 1.2, 1)]),
    "cardio": np.array([sofa_cardio(i, m) for i, m in zip(d.id, d.map_min.fillna(999))]),
    "cns": first([(d.gcs_min < 6, 4), (d.gcs_min <= 9, 3), (d.gcs_min <= 12, 2), (d.gcs_min <= 14, 1)]),
    "renal": first([(d.creat_max >= 5, 4), (d.uo < 200, 4), (d.creat_max >= 3.5, 3), (d.uo < 500, 3),
                    (d.creat_max >= 2, 2), (d.creat_max >= 1.2, 1)]),
}
d["sofa"] = sum(sofa.values())

for prefix, parts in (("saps", saps), ("oasis", oasis), ("sofa", sofa)):
    for k, v in parts.items():
        d[f"{prefix}_{k}"] = v

d = d.merge(pop, left_on="id", right_on="patientunitstayid").drop(columns="id")
d.to_parquet(out("scores.parquet"), index=False)
print(f"{len(d)} patients | median SAPS II {d.sapsii.median():.0f}, OASIS {d.oasis.median():.0f}, SOFA {d.sofa.median():.0f}")
