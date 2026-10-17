import json
import re

import pandas as pd

from common import EXTRACT_DIR, out, table


def num(s):
    return pd.to_numeric(s, errors="coerce")


def median_iqr(s):
    return [float(v) for v in s.dropna().quantile([0.5, 0.25, 0.75]).values]


def n_pct(flag):
    flag = flag.fillna(0).astype(int)
    return [int(flag.sum()), 100 * float(flag.mean())]


scores = pd.read_parquet(out("scores.parquet"))
cohort = pd.read_parquet(out("cohort.parquet"))
pt = pd.read_parquet(EXTRACT_DIR / "patient.parquet")
pt["patientunitstayid"] = pt.patientunitstayid.astype("int64")
pt = pt.assign(died_icu=(pt.unitdischargestatus == "Expired").astype(int),
               hospital_los_d=(num(pt.hospitaldischargeoffset) - num(pt.hospitaladmitoffset)) / 1440,
               pre_icu_d=-num(pt.hospitaladmitoffset) / 1440)

CONDITIONS = ["diabetes", "htn", "ckd", "afib", "copd", "stroke", "pvd"]
t = scores[scores.hospitaldischargestatus.notna()]
t = t.merge(cohort[["patientunitstayid", "age_num", "gender", "unitadmitsource", "apacheadmissiondx", "icu_los_h",
                    *CONDITIONS]], on="patientunitstayid", how="left")
t = t.merge(pt[["patientunitstayid", "died_icu", "hospital_los_d", "pre_icu_d"]], on="patientunitstayid", how="left")

# admission diagnosis groups, first match wins
DX_GROUPS = [
    ("Congestive heart failure", r"chf|congestive heart failure"),
    ("Acute myocardial infarction", r"\bmi\b|myocardial infarction|ami"),
    ("Cardiogenic shock", r"cardiogenic shock"),
    ("Cardiac arrest", r"arrest"),
    ("Unstable angina or chest pain", r"angina|chest pain"),
    ("Arrhythmia", r"rhythm|arrhythm|block|tachy|brady"),
    ("Respiratory failure or pulmonary edema", r"pulmonary|respir|edema"),
    ("Sepsis", r"sepsis"),
    ("Cardiac surgery", r"cabg|valve|surgery|bypass"),
]


def dx_group(text):
    for label, pattern in DX_GROUPS:
        if re.search(pattern, text):
            return label
    return "Other diagnoses"


t["dx_group"] = t.apacheadmissiondx.fillna("").str.lower().map(dx_group)

care = pd.read_csv(table("carePlanGeneral"), usecols=["patientunitstayid", "cplitemoffset", "cplgroup", "cplitemvalue"])
care = care[(care.cplgroup == "Care Limitation") & care.patientunitstayid.isin(t.patientunitstayid)]
first_entry = care[care.cplitemoffset == care.groupby("patientunitstayid").cplitemoffset.transform("min")]
first_entry = first_entry[first_entry.cplitemoffset <= 1440]
limited = first_entry.loc[~first_entry.cplitemvalue.str.contains("Full therapy", case=False, na=False), "patientunitstayid"]
t["limitation_of_support"] = t.patientunitstayid.isin(set(limited)).astype(int)

table1 = {
    "n": len(t),
    "hospitals": int(t.hospitalid.nunique()),
    "age": median_iqr(t.age_num),
    "female": n_pct(t.gender == "Female"),
    "from_emergency_department": n_pct(t.unitadmitsource == "Emergency Department"),
    "pre_icu_days": median_iqr(t.pre_icu_d),
    "hf_admission_diagnosis": n_pct(t.criterion == "admission"),
    "admission_diagnosis": [[k, int(v), 100 * v / len(t)] for k, v in t.dx_group.value_counts().items()],
    **{c: n_pct(t[c]) for c in CONDITIONS},
    "invasive_ventilation": n_pct(t.invasive_vent),
    "noninvasive_ventilation_only": n_pct(t.niv),
    "vasoactive_support": n_pct(t.sofa_cardio >= 2),
    "sofa": median_iqr(t.sofa),
    **{f"sofa_{k}_2_or_more": n_pct(t[f"sofa_{k}"] >= 2) for k in ("resp", "cardio", "renal", "cns", "liver", "coag")},
    "limitation_of_support": n_pct(t.limitation_of_support),
    "care_limitation_recorded": int(first_entry.patientunitstayid.nunique()),
    "apache_predicted_pct": median_iqr(100 * t.p_apache),
    "sapsii": median_iqr(t.sapsii),
    "sapsii_predicted_pct": median_iqr(100 * t.sapsii_prob),
    "oasis": median_iqr(t.oasis),
    "oasis_predicted_pct": median_iqr(100 * t.oasis_prob),
    "icu_los_h": median_iqr(t.icu_los_h),
    "hospital_los_d": median_iqr(t.hospital_los_d),
    "icu_death": n_pct(t.died_icu),
    "hospital_death": n_pct(t.died_hospital),
}

a = t[t.p_apache.notna()]
vent = a.vent_support == 1
COMPONENTS = {"age": "age", "heart_rate": "hr_min", "systolic_pressure": "sbp_min", "mean_pressure": "map_min",
              "respiratory_rate": "rr_min", "temperature": "temp_max", "gcs": "gcs_min", "urine_output": "uo",
              "bun": "bun_max", "wbc": "wbc_max", "potassium": "k_max", "sodium": "na_max", "bicarbonate": "hco3_min",
              "bilirubin": "bili_max", "pre_icu_time": "pre_icu_min"}
missing = {
    "n": len(a),
    "pct": {k: 100 * float(a[c].isna().mean()) for k, c in COMPONENTS.items()},
    "urine_output": int(a.uo.isna().sum()),
    "bilirubin": int(a.bili_max.isna().sum()),
    "ventilated": int(vent.sum()),
    "pf_ratio_ventilated": int((vent & a.pf_min.isna()).sum()),
}
missing["pct"]["pf_ratio_ventilated"] = 100 * float(a.loc[vent, "pf_min"].isna().mean())

predicted = {name: [100 * float(a[c].mean()), 100 * float(a[c].std())]
             for name, c in (("APACHE IVa", "p_apache"), ("SAPS II", "sapsii_prob"), ("OASIS", "oasis_prob"))}
spearman = a[["p_apache", "sapsii", "oasis"]].corr(method="spearman")["p_apache"].drop("p_apache").to_dict()

with open(out("descriptives.json"), "w") as f:
    json.dump({"table1": table1, "missing": missing, "predicted_mean_sd": predicted,
               "spearman_with_apache": spearman}, f, indent=1)

print(f"Table 1: {table1['n']} patients | hospital death {table1['hospital_death'][1]:.1f}%, "
      f"ICU death {table1['icu_death'][1]:.1f}%")
print(f"missing in the analysis set (n = {len(a)}): urine output {missing['urine_output']}, bilirubin "
      f"{missing['bilirubin']}, PaO2/FiO2 {missing['pf_ratio_ventilated']} of {missing['ventilated']} ventilated")
print("Spearman with APACHE IVa:", {k: round(v, 2) for k, v in spearman.items()})
